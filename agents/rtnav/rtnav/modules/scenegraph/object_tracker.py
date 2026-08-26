"""Multi-view object tracker: associates detections with scene graph nodes
using 3D bbox IoU and OWLv2 feature similarity."""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from rtnav.modules.scenegraph.cpp_accel import (
    batch_bbox_iou_containment_cpp,
    batch_voxel_iou_cpp,
    find_merge_candidates_cpp,
    score_candidates_cpp,
)
from rtnav.modules.scenegraph.merge_policy import same_label_pairs
from rtnav.modules.scenegraph.scene_graph import ObjectDetection, ObjectNode, SceneGraph, VoxelGrid


class ObjectTracker:
    """Associates detections with scene-graph nodes across views, and merges
    overlapping nodes. Requires the sg_accel C++ extension."""

    MAX_VIS_CHECKS = 30  # cap visibility checks per merge pass
    MERGE_BASE_INTERVAL = 5  # base cadence (frames) for the merge pass
    MERGE_MAX_FRAME_MS = 200  # skip merge when the frame already took this long
    MERGE_BBOX_IOU_MIN = 0.1  # min bbox IoU to merge two nodes
    MERGE_VOXEL_IOU_MIN = 0.05  # min voxel IoU to merge two nodes

    def __init__(
        self,
        config=None,
        voting_window: int = 10,
    ):
        self.cfg = config
        from rtnav.config.modules.scenegraph_cfg import SceneGraphConfig

        sg_cfg = config if config is not None else SceneGraphConfig()
        self.voxel_size_m = float(sg_cfg.voxel_size_m)
        self.merge_max_distance_m = float(sg_cfg.merge_max_distance_m)
        self.voting_window = voting_window
        # Min camera-position separation for two detections to count as distinct confirmation locations.
        ObjectNode._confirm_min_separation_m = float(sg_cfg.confirm_min_separation_m)

        # Node data cache (avoids rebuilding arrays every frame).
        self._node_cache_version = -1
        self._cached_node_ids = None
        self._cached_node_centroids = None
        self._cached_node_bboxes = None
        self._cached_node_features = None
        self._cached_kdtree = None
        self._cached_node_id_to_idx = {}
        self._cached_node_label_strs = None
        self._cached_node_label_set = None

    def _rebuild_node_cache(self, scene_graph: SceneGraph, feat_dim: int = 768):
        """Refresh cached node arrays. Uses incremental add/remove based on
        scene_graph._added_nodes / _removed_nodes; full rebuild only on cold start."""
        added = list(scene_graph._added_nodes)
        removed = set(scene_graph._removed_nodes)
        scene_graph._added_nodes = []
        scene_graph._removed_nodes = set()

        has_cache = self._cached_node_ids is not None and len(self._cached_node_ids) > 0

        if has_cache:
            changed = False

            if removed:
                keep_mask = np.array(
                    [nid not in removed for nid in self._cached_node_ids], dtype=bool
                )
                self._cached_node_ids = [
                    nid for nid, k in zip(self._cached_node_ids, keep_mask) if k
                ]
                self._cached_node_centroids = self._cached_node_centroids[keep_mask]
                self._cached_node_bboxes = self._cached_node_bboxes[keep_mask]
                self._cached_node_features = self._cached_node_features[keep_mask]
                if self._cached_node_label_strs is not None:
                    self._cached_node_label_strs = [
                        lbl for lbl, k in zip(self._cached_node_label_strs, keep_mask) if k
                    ]
                changed = True

            new_ids = [nid for nid in added if nid in scene_graph.nodes and nid not in removed]
            if new_ids:
                new_centroids = np.array(
                    [scene_graph.nodes[nid].centroid for nid in new_ids],
                    dtype=np.float64,
                )
                if len(new_centroids) > 0:
                    valid_mask = np.isfinite(new_centroids).all(axis=1)
                    if not valid_mask.all():
                        new_ids = [nid for nid, v in zip(new_ids, valid_mask) if v]
                        new_centroids = new_centroids[valid_mask]

                if new_ids:
                    self._cached_node_ids.extend(new_ids)
                    self._cached_node_centroids = np.concatenate(
                        [self._cached_node_centroids, new_centroids]
                    )
                    new_bboxes = np.array(
                        [scene_graph.nodes[nid].bbox_3d for nid in new_ids],
                        dtype=np.float64,
                    )
                    self._cached_node_bboxes = np.concatenate(
                        [self._cached_node_bboxes, new_bboxes]
                    )
                    new_features = np.zeros((len(new_ids), feat_dim), dtype=np.float64)
                    for i, nid in enumerate(new_ids):
                        node = scene_graph.nodes[nid]
                        if node.owl_features is not None:
                            new_features[i] = node.owl_features[:feat_dim]
                    self._cached_node_features = np.concatenate(
                        [self._cached_node_features, new_features]
                    )
                    if self._cached_node_label_strs is not None:
                        self._cached_node_label_strs.extend(
                            [scene_graph.nodes[nid].label for nid in new_ids]
                        )
                    changed = True

            if changed:
                num_nodes = len(self._cached_node_ids)
                if num_nodes > 0:
                    self._cached_kdtree = cKDTree(self._cached_node_centroids)
                else:
                    self._cached_kdtree = None
                self._cached_node_id_to_idx = {
                    nid: idx for idx, nid in enumerate(self._cached_node_ids)
                }
                if self._cached_node_label_strs is not None:
                    self._cached_node_label_set = set(self._cached_node_label_strs)
                else:
                    self._cached_node_label_set = None

            self._node_cache_version = scene_graph._version
            return

        all_node_ids = list(scene_graph.nodes.keys())
        all_centroids = np.array(
            [scene_graph.nodes[nid].centroid for nid in all_node_ids], dtype=np.float64
        )

        if len(all_centroids) > 0:
            valid_mask = np.isfinite(all_centroids).all(axis=1)
            if not valid_mask.all():
                invalid_count = (~valid_mask).sum()
                print(
                    f"[ObjectTracker] WARNING: Filtering {invalid_count} cached nodes with NaN/inf centroids"
                )
                self._cached_node_ids = [
                    nid for nid, valid in zip(all_node_ids, valid_mask) if valid
                ]
                self._cached_node_centroids = all_centroids[valid_mask]
            else:
                self._cached_node_ids = all_node_ids
                self._cached_node_centroids = all_centroids
        else:
            self._cached_node_ids = all_node_ids
            self._cached_node_centroids = all_centroids

        num_nodes = len(self._cached_node_ids)

        self._cached_node_bboxes = (
            np.array(
                [scene_graph.nodes[nid].bbox_3d for nid in self._cached_node_ids],
                dtype=np.float64,
            )
            if num_nodes > 0
            else np.zeros((0, 2, 3), dtype=np.float64)
        )

        self._cached_node_features = np.zeros((num_nodes, feat_dim), dtype=np.float64)
        for i, nid in enumerate(self._cached_node_ids):
            node = scene_graph.nodes[nid]
            if node.owl_features is not None:
                self._cached_node_features[i] = node.owl_features[:feat_dim]

        if num_nodes > 0:
            self._cached_kdtree = cKDTree(self._cached_node_centroids)
        else:
            self._cached_kdtree = None

        self._cached_node_id_to_idx = {nid: idx for idx, nid in enumerate(self._cached_node_ids)}

        self._cached_node_label_strs = [
            scene_graph.nodes[nid].label for nid in self._cached_node_ids
        ]
        self._cached_node_label_set = set(self._cached_node_label_strs)

        self._node_cache_version = scene_graph._version

    def _associate_detections_cpp(
        self, scene_graph: SceneGraph, detections: List[ObjectDetection], frame_idx: int
    ) -> Tuple[List[Tuple[int, ObjectDetection]], List[ObjectDetection]]:
        """KDTree radius search → C++ batch scoring → greedy match."""

        num_dets = len(detections)
        num_nodes = len(scene_graph.nodes)

        if num_nodes == 0:
            return [], detections

        feat_dim = 768
        for det in detections:
            if det.owl_features is not None:
                feat_dim = det.owl_features.shape[0]
                break

        cache_valid = (
            self._node_cache_version == scene_graph._version
            and self._cached_node_ids is not None
            and len(self._cached_node_ids) == num_nodes
        )
        if not cache_valid:
            self._rebuild_node_cache(scene_graph, feat_dim)

        det_label_set = {det.label for det in detections}
        if self._cached_node_label_set is None:
            node_label_set = {scene_graph.nodes[nid].label for nid in self._cached_node_ids}
            self._cached_node_label_set = node_label_set
        all_labels = det_label_set | self._cached_node_label_set
        label_list = sorted(all_labels)
        label_to_idx = {label: idx for idx, label in enumerate(label_list)}

        det_centroids = np.array([det.centroid for det in detections], dtype=np.float64)
        det_bboxes = np.array([det.bbox_3d for det in detections], dtype=np.float64)
        det_features = np.zeros((num_dets, feat_dim), dtype=np.float64)
        det_labels = np.zeros(num_dets, dtype=np.int32)
        for i, det in enumerate(detections):
            if det.owl_features is not None:
                det_features[i] = det.owl_features[:feat_dim]
            det_labels[i] = label_to_idx.get(det.label, -1)

        if self._cached_node_label_strs is None:
            self._cached_node_label_strs = [
                scene_graph.nodes[nid].label for nid in self._cached_node_ids
            ]
        node_labels_arr = np.array(
            [label_to_idx.get(lbl, -1) for lbl in self._cached_node_label_strs],
            dtype=np.int32,
        )

        candidate_det_indices = []
        candidate_node_indices = []
        det_candidate_ranges = []

        for det_idx in range(num_dets):
            start = len(candidate_det_indices)
            nearby = self._cached_kdtree.query_ball_point(det_centroids[det_idx], r=2.0)
            for node_idx in nearby:
                candidate_det_indices.append(det_idx)
                candidate_node_indices.append(node_idx)
            det_candidate_ranges.append((start, len(candidate_det_indices)))

        if not candidate_det_indices:
            return [], detections

        candidate_det_arr = np.array(candidate_det_indices, dtype=np.int32)
        candidate_node_arr = np.array(candidate_node_indices, dtype=np.int32)

        scores = score_candidates_cpp(
            candidate_det_arr,
            candidate_node_arr,
            det_bboxes,
            det_features,
            det_labels,
            det_centroids,
            self._cached_node_bboxes,
            self._cached_node_features,
            node_labels_arr,
            self._cached_node_centroids,
        )

        matched = []
        unmatched = []
        used_nodes = set()

        for det_idx, det in enumerate(detections):
            start, end = det_candidate_ranges[det_idx]
            if start == end:
                unmatched.append(det)
                continue

            det_label_idx = det_labels[det_idx]
            best_score = -1.0
            best_node_idx = -1
            best_label_match = False

            for i in range(start, end):
                node_idx = candidate_node_indices[i]
                if node_idx in used_nodes:
                    continue

                score = scores[i]
                if score < 0:
                    continue

                labels_match = node_labels_arr[node_idx] == det_label_idx

                # Prioritize label matches over higher scores.
                if (not best_label_match and labels_match) or (
                    best_label_match == labels_match and score > best_score
                ):
                    best_score = score
                    best_node_idx = node_idx
                    best_label_match = labels_match

            if best_node_idx >= 0:
                matched.append((self._cached_node_ids[best_node_idx], det))
                used_nodes.add(best_node_idx)
            else:
                unmatched.append(det)

        return matched, unmatched

    def associate_detections(
        self,
        scene_graph: SceneGraph,
        detections: List[ObjectDetection],
        frame_idx: int,
        debug: bool = False,
    ) -> Tuple[List[Tuple[int, ObjectDetection]], List[ObjectDetection]]:
        """Associate detections to existing nodes; returns (matched, unmatched)."""
        if len(scene_graph.nodes) == 0:
            if debug:
                print(
                    f"[ObjectTracker] No existing nodes, all {len(detections)} detections are new"
                )
            return [], detections

        return self._associate_detections_cpp(scene_graph, detections, frame_idx)

    def update_scene_graph(
        self,
        scene_graph: SceneGraph,
        detections: List[ObjectDetection],
        frame_idx: int,
        camera_params_list: Optional[List[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]] = None,
    ) -> Tuple[List[int], List[int]]:
        """Run association → dedup → update → create → visibility lifecycle.

        Returns (matched_node_ids, new_node_ids)."""

        t_usg_start = time.perf_counter()

        # Phase 1: associate (read-only).
        matched, unmatched = self.associate_detections(scene_graph, detections, frame_idx)
        # Phase 2: dedup — highest confidence wins per node.
        node_to_best_detection = {}
        for node_id, detection in matched:
            if node_id not in node_to_best_detection:
                node_to_best_detection[node_id] = detection
            else:
                if detection.confidence > node_to_best_detection[node_id].confidence:
                    node_to_best_detection[node_id] = detection
        # Phase 3: update matched nodes.
        matched_node_ids = []
        for node_id, detection in node_to_best_detection.items():
            scene_graph.update_node(node_id, detection, frame_idx)
            matched_node_ids.append(node_id)
        # Phase 3.5: incremental cache refresh for matched nodes only.
        if self._cached_node_id_to_idx and matched_node_ids:
            feat_dim = (
                self._cached_node_features.shape[1]
                if self._cached_node_features is not None and len(self._cached_node_features) > 0
                else 768
            )
            label_strs = self._cached_node_label_strs
            for node_id in matched_node_ids:
                idx = self._cached_node_id_to_idx.get(node_id)
                if idx is not None:
                    node = scene_graph.nodes[node_id]
                    self._cached_node_centroids[idx] = node.centroid
                    self._cached_node_bboxes[idx] = node.bbox_3d
                    if node.owl_features is not None:
                        self._cached_node_features[idx] = node.owl_features[:feat_dim]
                    if label_strs is not None and label_strs[idx] != node.label:
                        old_label = label_strs[idx]
                        label_strs[idx] = node.label
                        if self._cached_node_label_set is not None:
                            self._cached_node_label_set.add(node.label)
                            # Only drop old_label if no other node still holds it.
                            if old_label not in label_strs:
                                self._cached_node_label_set.discard(old_label)

        # Phase 4: create nodes for unmatched detections.
        new_node_ids = []
        for detection in unmatched:
            top_k = list(detection.top_k_label_probs or [])
            new_node = ObjectNode(
                node_id=0,
                label=detection.label,
                bbox_3d=detection.bbox_3d.copy(),
                centroid=detection.centroid.copy(),
                point_cloud=detection.point_cloud.copy(),
                colors=detection.colors.copy(),
                owl_features=(
                    detection.owl_features.copy() if detection.owl_features is not None else None
                ),
                confidence=detection.confidence,
                view_count=1,
                first_seen=frame_idx,
                last_seen=frame_idx,
                label_history=(
                    [(detection.label, detection.confidence, frame_idx, top_k)]
                    if detection.label
                    else []
                ),
                label_counts={detection.label: 1} if detection.label else {},
                label_prob_sums={label: float(prob) for label, prob in top_k},
                label_prob_count=1 if detection.label else 0,
                frames_since_detection=0,
                is_confirmed=False,
                voting_window=self.voting_window,
                number_to_be_confirmed=int(self.cfg.min_confirmation_views)
                if self.cfg is not None
                else 2,
                # Seed the first observation vantage (x, y, yaw) so confirmation counts it.
                observed_locations=(
                    [
                        (
                            float(detection.observed_from_xy[0]),
                            float(detection.observed_from_xy[1]),
                            float(detection.observed_from_yaw or 0.0),
                        )
                    ]
                    if detection.observed_from_xy is not None
                    else []
                ),
            )
            new_node._raw_obs = 1
            new_node._voxel_grid = VoxelGrid(voxel_size=self.voxel_size_m)
            new_node._voxel_grid.add_points(detection.point_cloud, detection.colors)
            node_id = scene_graph.add_node(new_node)
            new_node_ids.append(node_id)
        # Phase 4.5: extend cache with new nodes so the subsequent merge can use it.
        if new_node_ids and self._cached_node_ids is not None:
            feat_dim = (
                self._cached_node_features.shape[1]
                if self._cached_node_features is not None and len(self._cached_node_features) > 0
                else 768
            )
            new_centroids = np.array(
                [scene_graph.nodes[nid].centroid for nid in new_node_ids],
                dtype=np.float64,
            )
            new_bboxes = np.array(
                [scene_graph.nodes[nid].bbox_3d for nid in new_node_ids],
                dtype=np.float64,
            )
            new_features = np.zeros((len(new_node_ids), feat_dim), dtype=np.float64)
            for i, nid in enumerate(new_node_ids):
                node = scene_graph.nodes[nid]
                if node.owl_features is not None:
                    new_features[i] = node.owl_features[:feat_dim]
            self._cached_node_ids.extend(new_node_ids)
            self._cached_node_centroids = np.concatenate(
                [self._cached_node_centroids, new_centroids]
            )
            self._cached_node_bboxes = np.concatenate([self._cached_node_bboxes, new_bboxes])
            self._cached_node_features = np.concatenate([self._cached_node_features, new_features])
            for nid in new_node_ids:
                self._cached_node_id_to_idx[nid] = len(self._cached_node_id_to_idx)
            if self._cached_node_label_strs is not None:
                self._cached_node_label_strs.extend(
                    [scene_graph.nodes[nid].label for nid in new_node_ids]
                )
            if self._cached_node_label_set is not None:
                for nid in new_node_ids:
                    self._cached_node_label_set.add(scene_graph.nodes[nid].label)

        # Phase 5: visibility-based strikes for unmatched nodes.
        if camera_params_list is not None:
            matched_set = set(matched_node_ids)

            candidates = []
            for node_id, node in scene_graph.nodes.items():
                if node_id in matched_set:
                    continue
                if node.is_confirmed and node.frames_since_detection > 5:
                    continue
                if node.frames_since_detection > 8:
                    continue
                priority = (
                    0 if not node.is_confirmed else 1,
                    node.frames_since_detection,
                )
                candidates.append((priority, node_id, node))

            candidates.sort(key=lambda x: x[0])

            # Pre-compute inverse extrinsics once per camera.
            precomputed_cameras = []
            for intrinsics, extrinsics, image_shape in camera_params_list:
                world_to_cam = np.linalg.inv(extrinsics)
                precomputed_cameras.append((intrinsics, world_to_cam, image_shape))

            for _, node_id, node in candidates[: self.MAX_VIS_CHECKS]:
                is_visible = self._check_visibility_batch(node, precomputed_cameras)
                if is_visible:
                    node.frames_since_detection += 1
        t_phase5 = time.perf_counter()

        total_ms = (t_phase5 - t_usg_start) * 1000

        # Phase 6: periodic merge. Scale interval with node count; skip if over time budget.
        n_nodes = len(scene_graph.nodes)
        mult = 4 if n_nodes > 300 else 2 if n_nodes > 150 else 1
        merge_interval = self.MERGE_BASE_INTERVAL * mult

        if frame_idx % merge_interval == 0 and n_nodes > 1 and total_ms < self.MERGE_MAX_FRAME_MS:
            self.merge_overlapping_nodes(scene_graph, debug=False)

        return matched_node_ids, new_node_ids

    def _check_visibility_batch(
        self,
        node: ObjectNode,
        precomputed_cameras: List[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]],
    ) -> bool:
        """Visible in any of the precomputed cameras (≥3 of 8 bbox corners in view)."""
        if node.bbox_3d is None:
            return False

        min_xyz, max_xyz = node.bbox_3d[0], node.bbox_3d[1]
        corners = np.array(
            [
                [min_xyz[0], min_xyz[1], min_xyz[2]],
                [max_xyz[0], min_xyz[1], min_xyz[2]],
                [min_xyz[0], max_xyz[1], min_xyz[2]],
                [max_xyz[0], max_xyz[1], min_xyz[2]],
                [min_xyz[0], min_xyz[1], max_xyz[2]],
                [max_xyz[0], min_xyz[1], max_xyz[2]],
                [min_xyz[0], max_xyz[1], max_xyz[2]],
                [max_xyz[0], max_xyz[1], max_xyz[2]],
            ]
        )

        ones = np.ones((8, 1))
        corners_hom = np.concatenate([corners, ones], axis=1).T

        for intrinsics, world_to_cam, image_shape in precomputed_cameras:
            points_cam = (world_to_cam @ corners_hom)[:3, :].T

            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]

            z_cam = points_cam[:, 2]
            z_safe = np.where(np.abs(z_cam) < 1e-6, 1e-6, z_cam)

            u = (points_cam[:, 0] * fx / z_safe) + cx
            v = (points_cam[:, 1] * fy / z_safe) + cy

            height, width = image_shape
            in_front = z_cam > 0.1
            within_depth = z_cam < 10.0
            within_x = (u >= 0) & (u < width)
            within_y = (v >= 0) & (v < height)

            visible_mask = in_front & within_depth & within_x & within_y
            if np.sum(visible_mask) >= 3:
                return True

        return False

    def remove_stale_objects(
        self, scene_graph: SceneGraph, max_frames_without_detection: int = 10
    ) -> List[int]:
        """Drop nodes that have been visible-but-undetected for too long."""
        removed_node_ids = []

        for node_id, node in list(scene_graph.nodes.items()):
            if node.frames_since_detection >= max_frames_without_detection:
                scene_graph.remove_node(node_id)
                removed_node_ids.append(node_id)

        return removed_node_ids

    def merge_overlapping_nodes(
        self,
        scene_graph: SceneGraph,
        debug: bool = True,
    ) -> List[Tuple[int, int]]:
        """Merge same-label nodes that overlap (both bbox and voxel)."""
        if len(scene_graph.nodes) < 2:
            return []

        t_merge_start = time.perf_counter()

        node_ids = list(scene_graph.nodes.keys())
        num_nodes = len(node_ids)

        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        candidates = self._find_merge_candidates_cpp(scene_graph, node_ids)

        same_label_pairs = len(candidates)
        t_candidates = time.perf_counter()

        if len(candidates) == 0:
            if debug:
                print(f"[Merge] Stats: {same_label_pairs} same-label pairs, 0 within 3m, 0 merged")
            return []

        if self._cached_node_bboxes is not None and len(self._cached_node_bboxes) == num_nodes:
            all_bboxes = self._cached_node_bboxes
        else:
            all_bboxes = np.array(
                [scene_graph.nodes[nid].bbox_3d for nid in node_ids],
                dtype=np.float64,
            )
        pairs_arr = np.array(
            [[id_to_idx[c[0]], id_to_idx[c[1]]] for c in candidates], dtype=np.int32
        )
        bbox_ious = batch_bbox_iou_containment_cpp(all_bboxes, pairs_arr)

        bbox_pass_mask = bbox_ious >= self.MERGE_BBOX_IOU_MIN
        candidates_after_bbox = [c for c, passed in zip(candidates, bbox_pass_mask) if passed]
        bbox_ious_filtered = bbox_ious[bbox_pass_mask]

        close_pairs = len(candidates_after_bbox)
        t_bbox = time.perf_counter()

        if len(candidates_after_bbox) == 0:
            t_voxel = t_bbox
            return []

        voxel_ious = self._compute_voxel_ious_cpp(
            scene_graph, candidates_after_bbox, node_ids, id_to_idx
        )
        t_voxel = time.perf_counter()

        merged_pairs = []
        nodes_to_remove = set()
        for i, (node_id_a, node_id_b) in enumerate(candidates_after_bbox):
            if node_id_a in nodes_to_remove or node_id_b in nodes_to_remove:
                continue

            bbox_iou = bbox_ious_filtered[i]
            voxel_iou = voxel_ious[i]

            if voxel_iou < self.MERGE_VOXEL_IOU_MIN:
                continue

            node_a = scene_graph.nodes[node_id_a]
            node_b = scene_graph.nodes[node_id_b]

            # Keep the node seen from more views; merge the other into it.
            if node_a.view_count >= node_b.view_count:
                keep_node, remove_node = node_a, node_b
                keep_id, remove_id = node_id_a, node_id_b
            else:
                keep_node, remove_node = node_b, node_a
                keep_id, remove_id = node_id_b, node_id_a

            if debug:
                print(
                    f"[Merge] node {remove_id} ({remove_node.label}, {remove_node.view_count} views) "
                    f"-> node {keep_id} ({keep_node.label}, {keep_node.view_count} views) "
                    f"| bbox_iou={bbox_iou:.2f}, voxel_iou={voxel_iou:.2f}"
                )

            self._merge_node_data(keep_node, remove_node)

            nodes_to_remove.add(remove_id)
            merged_pairs.append((keep_id, remove_id))

        for node_id in nodes_to_remove:
            if node_id in scene_graph.nodes:
                del scene_graph.nodes[node_id]
                scene_graph._removed_nodes.add(node_id)
        if nodes_to_remove:
            scene_graph._version += 1

        t_merge_end = time.perf_counter()
        merge_total_ms = (t_merge_end - t_merge_start) * 1000
        if debug:
            print(
                f"[Merge] Stats: {same_label_pairs} same-label pairs, {close_pairs} within 3m, {len(merged_pairs)} merged"
            )
            if merged_pairs:
                print(f"[Merge] {len(scene_graph.nodes)} nodes remaining after merge")
            if merge_total_ms > self.MERGE_MAX_FRAME_MS:
                cand_ms = (t_candidates - t_merge_start) * 1000
                bbox_ms = (t_bbox - t_candidates) * 1000
                voxel_ms = (t_voxel - t_bbox) * 1000
                rest_ms = (t_merge_end - t_voxel) * 1000
                print(
                    f"[Merge] SLOW ({merge_total_ms:.0f}ms): candidates={cand_ms:.0f}ms bbox={bbox_ms:.0f}ms voxel={voxel_ms:.0f}ms rest={rest_ms:.0f}ms"
                )

        return merged_pairs

    def _find_merge_candidates_cpp(
        self, scene_graph: SceneGraph, node_ids: List[int]
    ) -> List[Tuple[int, int]]:
        """Find merge candidates via C++ spatial hashing; uses cache when valid."""
        if (
            self._cached_node_id_to_idx
            and self._cached_node_centroids is not None
            and len(self._cached_node_ids) == len(node_ids)
        ):
            centroids = self._cached_node_centroids.astype(np.float64)
            label_strs = self._cached_node_label_strs
            if label_strs is not None:
                label_to_idx = {}
                label_ints = np.empty(len(node_ids), dtype=np.int32)
                for i, ls in enumerate(label_strs):
                    if ls not in label_to_idx:
                        label_to_idx[ls] = len(label_to_idx)
                    label_ints[i] = label_to_idx[ls]
            else:
                label_to_idx = {}
                label_ints = np.empty(len(node_ids), dtype=np.int32)
                for i, nid in enumerate(node_ids):
                    label = scene_graph.nodes[nid].label
                    if label not in label_to_idx:
                        label_to_idx[label] = len(label_to_idx)
                    label_ints[i] = label_to_idx[label]
        else:
            label_to_idx = {}
            centroids = np.array(
                [scene_graph.nodes[nid].centroid for nid in node_ids], dtype=np.float64
            )
            label_ints = np.empty(len(node_ids), dtype=np.int32)
            for i, nid in enumerate(node_ids):
                label = scene_graph.nodes[nid].label
                if label not in label_to_idx:
                    label_to_idx[label] = len(label_to_idx)
                label_ints[i] = label_to_idx[label]

        pairs_idx = find_merge_candidates_cpp(
            centroids, label_ints, max_distance=self.merge_max_distance_m
        )

        # Map pair indices back via the ID list that matches the centroids array.
        ids = (
            self._cached_node_ids
            if (
                self._cached_node_centroids is not None
                and len(self._cached_node_ids) == len(node_ids)
            )
            else node_ids
        )
        pairs = [(ids[p[0]], ids[p[1]]) for p in pairs_idx]
        return same_label_pairs(pairs, scene_graph.nodes)

    def _compute_voxel_ious_cpp(
        self,
        scene_graph: SceneGraph,
        candidates: List[Tuple[int, int]],
        node_ids: List[int],
        id_to_idx: Dict[int, int],
    ) -> np.ndarray:
        """Voxel IoU via C++ batch op; only builds keys for nodes in candidate pairs."""
        candidate_nids = set()
        for c in candidates:
            candidate_nids.add(c[0])
            candidate_nids.add(c[1])

        compact_nids = sorted(candidate_nids)
        compact_id_to_idx = {nid: i for i, nid in enumerate(compact_nids)}

        all_voxel_keys = []
        for nid in compact_nids:
            node = scene_graph.nodes[nid]
            voxel_grid = node.get_voxel_grid()
            if voxel_grid is not None and len(voxel_grid.voxels) > 0:
                keys = voxel_grid.get_keys_np()
            else:
                keys = np.zeros((0, 3), dtype=np.int32)
            all_voxel_keys.append(keys)

        pairs = np.array(
            [[compact_id_to_idx[c[0]], compact_id_to_idx[c[1]]] for c in candidates],
            dtype=np.int32,
        )

        return batch_voxel_iou_cpp(all_voxel_keys, pairs)

    def _merge_node_data(self, keep_node, remove_node):
        """Fold remove_node's data into keep_node."""
        # Merge voxel dicts directly (avoid export/re-import); defer get_points().
        if remove_node._voxel_grid is not None and len(remove_node._voxel_grid.voxels) > 0:
            if keep_node._voxel_grid is None:
                keep_node._voxel_grid = VoxelGrid(voxel_size=self.voxel_size_m)
            keep_node._voxel_grid.voxels.update(remove_node._voxel_grid.voxels)
            keep_node._voxel_grid._cache_valid = False
            keep_node._voxel_grid._cached_keys_np = None

        # bbox = union.
        keep_node.bbox_3d[0] = np.minimum(keep_node.bbox_3d[0], remove_node.bbox_3d[0])
        keep_node.bbox_3d[1] = np.maximum(keep_node.bbox_3d[1], remove_node.bbox_3d[1])

        total_views = keep_node.view_count + remove_node.view_count
        keep_node.view_count = total_views

        keep_node.label_history.extend(remove_node.label_history)
        keep_node.label_history.sort(key=lambda entry: int(entry[2]) if len(entry) > 2 else -1)
        keep_node.label_history = keep_node.label_history[-50:]
        for label, count in remove_node.label_counts.items():
            keep_node.label_counts[label] = keep_node.label_counts.get(label, 0) + count
        keep_node.label_prob_count += remove_node.label_prob_count
        for label, probability in remove_node.label_prob_sums.items():
            keep_node.label_prob_sums[label] = (
                keep_node.label_prob_sums.get(label, 0.0) + probability
            )

        if remove_node.owl_features is not None:
            if keep_node.owl_features is None:
                keep_node.owl_features = remove_node.owl_features.copy()
            else:
                keep_node.owl_features = (
                    keep_node.owl_features * (total_views - remove_node.view_count)
                    + remove_node.owl_features * remove_node.view_count
                ) / total_views
                keep_node.owl_features /= np.linalg.norm(keep_node.owl_features) + 1e-8

        # Fold in distinct observation vantages, then confirm on the combined count.
        for loc in remove_node.observed_locations:
            keep_node._register_location(loc[:2], loc[2] if len(loc) > 2 else None)
        if keep_node.observed_locations:
            keep_node.view_count = len(keep_node.observed_locations)
            if len(keep_node.observed_locations) >= keep_node.number_to_be_confirmed:
                keep_node.is_confirmed = True
        elif keep_node.view_count >= keep_node.number_to_be_confirmed:
            keep_node.is_confirmed = True
        keep_node.first_seen = min(keep_node.first_seen, remove_node.first_seen)
        keep_node.last_seen = max(keep_node.last_seen, remove_node.last_seen)
