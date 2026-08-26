"""Scene Graph Thread — online 3D object mapping driven by detection results.

Consumes ``shared_state.perception.detection_result`` (multi-camera detection
output from the OWLv2 detector thread), back-projects 2-D boxes
into 3-D using the camera intrinsics and depth, runs an ObjectTracker to
associate hits across frames, and publishes a ``SceneGraphOutput`` to
``shared_state.scenegraph.scene_graph`` for target selection and visualization.
"""

import threading
import time
from typing import Optional

import numpy as np

from rtnav.core.data_types import (
    MultiCameraDetectionResult,
    SceneGraphNode,
    SceneGraphOutput,
)
from rtnav.modules.scenegraph.det3d import process_all_cameras_parallel
from rtnav.modules.scenegraph.object_tracker import ObjectTracker
from rtnav.modules.scenegraph.scene_graph import SceneGraph


class SceneGraphThread(threading.Thread):
    """Real-time scene graph builder."""

    def __init__(
        self,
        shared_state,
        shutdown_event,
        cfg,
        update_rate: Optional[float] = 0.5,
    ):
        super().__init__(name="SceneGraphThread", daemon=True)
        self.shared_state = shared_state
        self.shutdown_event = shutdown_event
        self.update_rate = update_rate

        scenegraph_cfg = cfg.scenegraph
        self.scenegraph_cfg = scenegraph_cfg
        self.remove_stale_objects = scenegraph_cfg.remove_stale_objects
        self.max_frames_without_detection = scenegraph_cfg.max_frames_without_detection

        # Pruning safety valves — kicks in only at extreme node counts.
        self.SOFT_LIMIT = 2000
        self.HARD_LIMIT = 2500

        self.scene_graph = SceneGraph()
        self.object_tracker = ObjectTracker(config=scenegraph_cfg)

        self.frame_counter = 0
        self._last_processed_timestamp: Optional[float] = None

        if self.update_rate is None:
            print("SceneGraphThread initialized (unlimited rate)")
        else:
            print(f"SceneGraphThread initialized (update_rate={update_rate}Hz)")
        if self.remove_stale_objects:
            print(
                f"  Stale-object removal ENABLED (max_strikes={self.max_frames_without_detection})"
            )

        with self.shared_state.lock:
            self._last_episode_index = getattr(self.shared_state.system, "episode_index", 0)

    def reset_episode(self) -> None:
        """Synchronous episode-boundary reset, called by the runner."""
        self._reset()

    def _reset(self):
        # Atomic check-and-update so the runner's synchronous teardown call
        # and this thread's own polling can't both rebuild on the same
        # counter bump.
        with self.shared_state.lock:
            counter = getattr(self.shared_state.system, "episode_index", 0)
            if counter <= self._last_episode_index:
                return
            self._last_episode_index = counter
        print("[SceneGraphThread] Episode reset: creating fresh scene graph + tracker")
        self.scene_graph = SceneGraph()
        self.object_tracker = ObjectTracker(config=self.scenegraph_cfg)
        self.frame_counter = 0
        self._last_processed_timestamp = None
        with self.shared_state.lock:
            self.shared_state.scenegraph.scene_graph = None
            self.shared_state.scenegraph.pending_verified_detections = []

    def _consume_verified_target_requests(self) -> None:
        with self.shared_state.lock:
            requests = list(
                getattr(self.shared_state.scenegraph, "pending_verified_detections", []) or []
            )
            if not requests:
                return
            self.shared_state.scenegraph.pending_verified_detections = []

        assigned: dict[str, int] = {}
        for request in requests:
            detection = request.get("detection")
            request_id = request.get("request_id")
            if detection is None or request_id is None:
                continue
            matched_ids, new_ids = self.object_tracker.update_scene_graph(
                self.scene_graph,
                [detection],
                self.frame_counter,
            )
            node_ids = list(matched_ids or []) + list(new_ids or [])
            if node_ids:
                assigned[str(request_id)] = int(node_ids[0])

        if not assigned:
            return
        output = self._convert_to_output_format(time.time())
        with self.shared_state.lock:
            targets = list(getattr(self.shared_state.target, "target_goals", []) or [])
            assigned_target_node_ids: set[int] = set()
            for target in targets:
                request_id = target.get("sg_request_id")
                if request_id is not None and str(request_id) in assigned:
                    node_id = int(assigned[str(request_id)])
                    target["node_id"] = node_id
                    assigned_target_node_ids.add(node_id)
            if assigned_target_node_ids:
                self.shared_state.target.target_node_blacklist_ids.difference_update(
                    assigned_target_node_ids
                )
            self.shared_state.target.target_goals = targets
            self.shared_state.scenegraph.scene_graph = output
        print(
            f"[SceneGraph] Added {len(assigned)} VLM-confirmed target node(s): "
            f"{sorted(assigned.values())}"
        )

    def run(self):
        print("SceneGraphThread started")
        from rtnav.utils.task_gate import wait_for_task_ready

        while not self.shutdown_event.is_set():
            self._reset()
            # Per-iteration task_ready gate: pause for the whole reset
            # window so we don't grow the new episode's scene graph using
            # the previous episode's tail detections.
            if not wait_for_task_ready(self.shared_state, "SceneGraph", self.shutdown_event):
                break
            start_time = time.time()
            self._consume_verified_target_requests()

            try:
                detection_result = self._get_detection_result()

                if detection_result is not None:
                    self._process(detection_result)
                    self.frame_counter += 1
                    continue
                else:
                    time.sleep(0.01)
                    continue

            except Exception as e:
                print(f"[SceneGraphThread] error: {e}")
                import traceback

                traceback.print_exc()

            if self.update_rate is not None:
                elapsed = time.time() - start_time
                sleep_time = max(0.0, 1.0 / self.update_rate - elapsed)
                time.sleep(sleep_time)
            else:
                time.sleep(0.01)

        print("SceneGraphThread stopped")

    def _get_detection_result(self) -> Optional[MultiCameraDetectionResult]:
        with self.shared_state.lock:
            result = self.shared_state.perception.detection_result
            if result is None:
                return None
            current_reset_counter = getattr(self.shared_state.system, "episode_index", 0)
            result_reset_counter = getattr(result, "episode_index", current_reset_counter)
            if result_reset_counter != current_reset_counter:
                self.shared_state.perception.detection_result = None
                print(
                    "[SceneGraphThread] Dropped stale detection_result "
                    f"(result_reset={result_reset_counter}, "
                    f"current_reset={current_reset_counter})"
                )
                return None
            if (
                self._last_processed_timestamp is not None
                and result.timestamp == self._last_processed_timestamp
            ):
                return None
            return result

    def _process(self, detection_result: MultiCameraDetectionResult):
        detections_3d = process_all_cameras_parallel(
            camera_results=detection_result.camera_results
        )

        camera_params_list = []
        for cam_name, cam_result in detection_result.camera_results.items():
            camera_params_list.append(
                (
                    cam_result.intrinsics,
                    cam_result.T_world_camera,
                    cam_result.rgb_image.shape[:2],
                )
            )

        if detections_3d:
            self.object_tracker.update_scene_graph(
                self.scene_graph,
                detections_3d,
                self.frame_counter,
                camera_params_list=camera_params_list,
            )

        if self.remove_stale_objects:
            self.object_tracker.remove_stale_objects(
                self.scene_graph,
                max_frames_without_detection=self.max_frames_without_detection,
            )

        if len(self.scene_graph.nodes) >= self.HARD_LIMIT:
            self._prune_scene_graph(target=self.SOFT_LIMIT)

        scene_graph_output = self._convert_to_output_format(detection_result.timestamp)

        with self.shared_state.lock:
            self.shared_state.scenegraph.scene_graph = scene_graph_output

        self._last_processed_timestamp = detection_result.timestamp

    def _prune_scene_graph(self, target=None):
        if target is None:
            target = self.SOFT_LIMIT
        nodes_dict = self.scene_graph.nodes
        if len(nodes_dict) <= target:
            return
        n_to_remove = len(nodes_dict) - target
        if n_to_remove <= 0:
            return

        REDUNDANCY_DIST = 1.5
        from collections import defaultdict

        label_groups = defaultdict(list)
        for nid, n in nodes_dict.items():
            if n.centroid is not None:
                label_groups[n.label].append((nid, n))

        protected_ids = set()
        for label, group in label_groups.items():
            confirmed_in_group = [(nid, n) for nid, n in group if n.is_confirmed]
            if len(confirmed_in_group) == 1:
                protected_ids.add(confirmed_in_group[0][0])
            elif len(confirmed_in_group) == 0 and len(group) == 1:
                protected_ids.add(group[0][0])

        unconfirmed = []
        for nid, n in nodes_dict.items():
            if nid in protected_ids:
                continue
            if not n.is_confirmed:
                unconfirmed.append((nid, n.confidence, n.view_count))

        confirmed_redundant = []
        for label, group in label_groups.items():
            confirmed = [
                (nid, n) for nid, n in group if n.is_confirmed and nid not in protected_ids
            ]
            if len(confirmed) <= 1:
                continue
            confirmed.sort(key=lambda x: x[1].confidence, reverse=True)
            for i, (nid, n) in enumerate(confirmed):
                for j in range(i):
                    better_n = confirmed[j][1]
                    dist = float(np.linalg.norm(n.centroid[:2] - better_n.centroid[:2]))
                    if dist < REDUNDANCY_DIST:
                        confirmed_redundant.append((nid, n.confidence, n.view_count))
                        break

        unconfirmed.sort(key=lambda x: (x[1], x[2]))
        confirmed_redundant.sort(key=lambda x: (x[1], x[2]))
        candidates = unconfirmed + confirmed_redundant

        removed_unconfirmed = 0
        removed_confirmed = 0
        removed_count = 0
        for nid, _, _ in candidates:
            if removed_count >= n_to_remove:
                break
            if nid in nodes_dict:
                is_confirmed = nodes_dict[nid].is_confirmed
                del nodes_dict[nid]
                removed_count += 1
                if is_confirmed:
                    removed_confirmed += 1
                else:
                    removed_unconfirmed += 1

        if removed_count:
            print(
                f"[SceneGraph] Pruned {removed_count} nodes "
                f"({removed_unconfirmed} unconfirmed, {removed_confirmed} redundant) "
                f"→ {len(nodes_dict)} remaining (target={target}, "
                f"{len(protected_ids)} protected)"
            )

    def _convert_to_output_format(self, timestamp: float):
        _empty_3 = np.zeros(3)
        _empty_bbox = np.zeros((2, 3))
        _empty_pc = np.zeros((0, 3))

        nodes = []
        for node_id, node in self.scene_graph.nodes.items():
            # Publish label history as a flat list of label strings (with
            # repetition) so consumers can compute frequency-weighted
            # supermajority.  A plain set of distinct labels loses all
            # frequency information — a node seen as "chair" 18× and
            # "sofa" 1× would look like a 50/50 split with a set.
            if node.label_history:
                past_labels = [entry[0] for entry in node.label_history]
            else:
                past_labels = list(node.all_labels) if node.all_labels else []

            point_cloud = node.point_cloud if node.point_cloud is not None else _empty_pc
            colors = node.colors if node.colors is not None else _empty_pc
            voxel_grid = node.get_voxel_grid()
            if voxel_grid is not None:
                point_cloud, colors = voxel_grid.get_points()

            scene_node = SceneGraphNode(
                node_id=node.node_id,
                past_labels=past_labels,
                chosen_label=node.label,
                confidence=node.confidence,
                centroid=node.centroid if node.centroid is not None else _empty_3,
                bbox_3d=node.bbox_3d if node.bbox_3d is not None else _empty_bbox,
                point_cloud=point_cloud,
                colors=colors,
                view_count=node.view_count,
                is_confirmed=node.is_confirmed,
                observed_from_xy=getattr(node, "observed_from_xy", None),
                label_counts=dict(node.label_counts),  # copy: uncapped lifetime tally
                label_history=list(node.label_history),
                label_prob_sums=dict(node.label_prob_sums),
                label_prob_count=int(node.label_prob_count),
            )
            nodes.append(scene_node)

        return SceneGraphOutput(
            nodes=nodes,
            timestamp=timestamp,
        )
