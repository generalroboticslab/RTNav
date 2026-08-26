"""VLM verification for detector target candidates."""

from __future__ import annotations

import base64
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Optional

import cv2
import numpy as np

from rtnav.modules.decision.target_utils import (
    label_matches_any,
    normalize_target_names,
)
from rtnav.utils.vllm_utils import get_vllm_base_url, get_vllm_model_id

YES_THRESHOLD = 0.90
NODE_REJECTION_BLACKLIST_COUNT = 3
NODE_REJECTION_MAX_YES_SCORE = 0.50
NODE_MATCH_RADIUS_M = 1.0
MOBILE_SAM_EROSION_SIZE = 5
TEMPORARY_TARGET_DISTANCE_M = 5.0
MAX_RELIABLE_TARGET_DISTANCE_M = 4.95


@dataclass
class TargetCandidate:
    signature: tuple
    camera_name: str
    label: str
    canonical: str
    bbox: tuple
    confidence: float
    timestamp: float
    boxed_rgb: np.ndarray
    camera_result: Any
    detection_entity: Any
    object_detection: Optional[Any]
    node_id: Optional[int]
    x: Optional[float]
    y: Optional[float]
    surface_x: Optional[float]
    surface_y: Optional[float]
    distance: float
    temporary_target: bool
    competing_label: Optional[str] = None
    sam_mask: Optional[np.ndarray] = None

    @property
    def has_geometry(self) -> bool:
        return (
            self.object_detection is not None
            and self.x is not None
            and self.y is not None
            and self.surface_x is not None
            and self.surface_y is not None
        )

    @property
    def has_temporary_goal(self) -> bool:
        return (
            self.temporary_target
            and self.x is not None
            and self.y is not None
            and self.surface_x is not None
            and self.surface_y is not None
        )


@dataclass
class VerificationResult:
    candidate: TargetCandidate
    yes_prob: float
    no_prob: float
    yes_norm: float
    inference_time_s: float

    @property
    def accepted(self) -> bool:
        return self.yes_norm > YES_THRESHOLD


def _choice_top_logprobs(choice):
    if choice.logprobs is None:
        raise RuntimeError("vLLM response did not include token logprobs")
    if hasattr(choice.logprobs, "top_logprobs") and choice.logprobs.top_logprobs:
        return choice.logprobs.top_logprobs[0].items()
    if hasattr(choice.logprobs, "content") and choice.logprobs.content:
        top_logprobs = choice.logprobs.content[0].top_logprobs
        return ((entry.token, entry.logprob) for entry in top_logprobs)
    raise RuntimeError("vLLM response did not include first-token top logprobs")


def _yes_no_token_probs(choice) -> tuple[float, float]:
    yes_prob = 0.0
    no_prob = 0.0
    for token, logprob in _choice_top_logprobs(choice):
        prob = math.exp(float(logprob))
        normalized = str(token).strip().lower()
        if normalized == "yes":
            yes_prob += prob
        elif normalized == "no":
            no_prob += prob
    return yes_prob, no_prob


def _normalized_yes(yes_prob: float, no_prob: float) -> float:
    denom = yes_prob + no_prob
    return yes_prob / denom if denom > 0.0 else 0.0


def _boxed_rgb(rgb: np.ndarray, bbox: tuple) -> np.ndarray:
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    image = np.ascontiguousarray(image.copy())
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    cx = 0.5 * (left + right)
    cy = 0.5 * (top + bottom)
    half_w = 0.6 * (right - left)
    half_h = 0.6 * (bottom - top)
    p1 = (
        max(0, min(w - 1, int(round(cx - half_w)))),
        max(0, min(h - 1, int(round(cy - half_h)))),
    )
    p2 = (
        max(0, min(w - 1, int(round(cx + half_w)))),
        max(0, min(h - 1, int(round(cy + half_h)))),
    )
    thickness = max(2, int(round(min(h, w) * 0.005)))
    cv2.rectangle(image, p1, p2, (255, 0, 0), thickness=thickness)
    return image


def _image_data_url(rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("[TargetVLM] failed to JPEG-encode boxed RGB frame")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _verification_figure(result: VerificationResult) -> np.ndarray:
    """Visual evidence sent to the VLM, annotated with its result."""
    candidate = result.candidate
    image = np.ascontiguousarray(np.asarray(candidate.boxed_rgb).copy())
    if candidate.sam_mask is not None:
        mask = (np.asarray(candidate.sam_mask) > 0).astype(np.uint8)
        h, w = image.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 0, 0), 2, cv2.LINE_AA)

    verdict = "YES" if result.accepted else "NO"
    lines = [
        f"{verdict}  p(yes)={result.yes_norm:.3f}",
        f"raw yes={result.yes_prob:.3f}  no={result.no_prob:.3f}",
        f"time={result.inference_time_s:.2f}s",
    ]
    if candidate.competing_label:
        lines.append(f"competing: {candidate.competing_label}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(image.shape[:2]) / 900.0)
    thickness = max(1, round(scale * 2))
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    pad, gap = 10, 6
    width = max(size[0] for size in sizes) + 2 * pad
    line_height = max(size[1] for size in sizes)
    height = len(lines) * line_height + (len(lines) - 1) * gap + 2 * pad
    x0 = max(0, image.shape[1] - width - pad)
    y0 = pad
    cv2.rectangle(image, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
    y = y0 + pad + line_height
    for line in lines:
        cv2.putText(
            image,
            line,
            (x0 + pad, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height + gap
    return image


def _mask_too_offset(mask: np.ndarray) -> bool:
    """VLFM's provisional-evidence test for masks cropped by a side edge."""
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    x, _, width, _ = cv2.boundingRect(mask)
    image_width = mask.shape[1]
    third = image_width // 3
    return (x + width <= third and x <= int(0.05 * image_width)) or (
        x >= 2 * third and x + width >= int(0.95 * image_width)
    )


def _surface_xy_from_points(point_cloud, reference_xyz: Optional[np.ndarray]):
    if point_cloud is None:
        return None
    pts = np.asarray(point_cloud, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return None
    pts = _finite_points(pts)
    if len(pts) == 0:
        return None
    if reference_xyz is None:
        return np.median(pts[:, :2], axis=0)
    ref = np.asarray(reference_xyz, dtype=float)
    if pts.shape[1] >= 3 and ref.shape[0] >= 3:
        dists = np.linalg.norm(pts[:, :3] - ref[:3][None, :], axis=1)
    else:
        dists = np.linalg.norm(pts[:, :2] - ref[:2][None, :], axis=1)
    valid_d = np.isfinite(dists)
    pts = pts[valid_d]
    dists = dists[valid_d]
    if len(pts) == 0:
        return None
    return pts[int(np.argmin(dists)), :2]


def _finite_points(pts: np.ndarray) -> np.ndarray:
    cols = 3 if pts.shape[1] >= 3 else 2
    return pts[np.isfinite(pts[:, :cols]).all(axis=1)]


class TargetVerifier:
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self._queried_signatures: set[tuple] = set()
        self._node_rejection_counts: dict[int, int] = {}
        self._mobile_sam = None
        self._request_counter = 0

    def reset(self) -> None:
        self._queried_signatures.clear()
        self._node_rejection_counts.clear()
        self._request_counter = 0

    def target_name(self, param: Any) -> str:
        names = normalize_target_names(param)
        with self.shared_state.lock:
            goal_category = str(getattr(self.shared_state.task, "goal_category", "") or "").strip()
        if goal_category:
            return goal_category
        if names:
            return names[0]
        raise ValueError("[TargetVLM] cannot verify target without a target name")

    def collect_candidates(
        self,
        param: Any,
        include_queried: bool = False,
    ) -> list[TargetCandidate]:
        names = normalize_target_names(param)
        with self.shared_state.lock:
            detection_result = getattr(self.shared_state.perception, "detection_result", None)
            synonym_to_canonical = dict(
                getattr(self.shared_state.task, "synonym_to_canonical", {}) or {}
            )
            goal_category = (
                str(getattr(self.shared_state.task, "goal_category", "") or "").lower().strip()
            )
            reset_counter = int(getattr(self.shared_state.system, "episode_index", 0))
            sg = getattr(self.shared_state.scenegraph, "scene_graph", None)
            robot_odom = getattr(self.shared_state.sensor, "latest_odom", None)
            blacklisted_nodes = set(
                getattr(self.shared_state.target, "target_node_blacklist_ids", set()) or set()
            )

        if detection_result is None:
            return []
        result_reset = int(getattr(detection_result, "episode_index", reset_counter))
        if result_reset != reset_counter:
            return []

        lookup = synonym_to_canonical
        if not lookup:
            return []
        target_set = {n.lower().strip() for n in names} | set(lookup.keys()) | {goal_category}
        target_set.discard("")
        robot_xy = (
            np.array([float(robot_odom[0]), float(robot_odom[1])], dtype=float)
            if robot_odom is not None
            else None
        )

        out: list[TargetCandidate] = []
        for cam_name, cam in detection_result.camera_results.items():
            for det in list(getattr(cam, "detections", []) or []):
                det_label = str(det.name).lower().strip()
                canonical = lookup.get(det_label)
                if canonical is None and not label_matches_any(det_label, target_set):
                    continue
                canonical = canonical or goal_category or str(det.name)
                bbox_key = tuple(int(round(float(v))) for v in det.bbox)
                signature = (
                    reset_counter,
                    str(cam_name),
                    det_label,
                    bbox_key,
                )
                if signature in self._queried_signatures and not include_queried:
                    continue
                projected = self._project_detection(cam_name, cam, det)
                object_detection = None
                node_id = None
                x = y = surface_x = surface_y = None
                distance = float("inf")
                temporary_target = False
                competing_label = None
                if projected is not None:
                    if float(getattr(projected, "robot_distance", float("inf"))) < 0.5:
                        continue
                    node_id = self._nearest_scene_graph_node_id(sg, projected.centroid)
                    if node_id is not None and int(node_id) in blacklisted_nodes:
                        continue
                    competing_label = self._competing_scene_graph_label(
                        sg,
                        node_id,
                        (det_label, canonical, goal_category),
                        det,
                    )
                    if node_id is not None:
                        node_label = self._scene_graph_node_label(sg, node_id)
                        print(
                            f"[TargetVLM] candidate node={node_id} "
                            f"det='{det_label}' node_label='{node_label or 'none'}' "
                            f"competing='{competing_label or 'none'}'"
                        )
                    T_world_camera = getattr(cam, "T_world_camera", None)
                    camera_xyz = (
                        np.asarray(T_world_camera[:3, 3], dtype=float)
                        if T_world_camera is not None
                        else None
                    )
                    surface_xy = _surface_xy_from_points(projected.point_cloud, camera_xyz)
                    if surface_xy is None:
                        surface_xy = np.asarray(projected.centroid[:2], dtype=float)
                    distance = (
                        float(
                            np.linalg.norm(
                                np.asarray(projected.centroid[:2], dtype=float) - robot_xy
                            )
                        )
                        if robot_xy is not None
                        else 0.0
                    )
                    object_detection = projected
                    x = float(projected.centroid[0])
                    y = float(projected.centroid[1])
                    surface_x = float(surface_xy[0])
                    surface_y = float(surface_xy[1])
                else:
                    temp_xy = self._temporary_boundary_xy(cam, det.bbox)
                    if temp_xy is not None:
                        temporary_target = True
                        x = surface_x = float(temp_xy[0])
                        y = surface_y = float(temp_xy[1])
                        distance = (
                            float(np.linalg.norm(np.asarray(temp_xy, dtype=float) - robot_xy))
                            if robot_xy is not None
                            else TEMPORARY_TARGET_DISTANCE_M
                        )
                out.append(
                    TargetCandidate(
                        signature=signature,
                        camera_name=str(cam_name),
                        label=str(det.name),
                        canonical=str(canonical),
                        bbox=tuple(float(v) for v in det.bbox),
                        confidence=float(det.confidence),
                        timestamp=float(detection_result.timestamp),
                        boxed_rgb=_boxed_rgb(cam.rgb_image, det.bbox),
                        camera_result=cam,
                        detection_entity=det,
                        object_detection=object_detection,
                        node_id=node_id,
                        x=x,
                        y=y,
                        surface_x=surface_x,
                        surface_y=surface_y,
                        distance=distance,
                        temporary_target=temporary_target,
                        competing_label=competing_label,
                    )
                )
        return out

    @staticmethod
    def _temporary_boundary_xy(cam, bbox: tuple) -> Optional[tuple]:
        rgb = getattr(cam, "rgb_image", None)
        depth = getattr(cam, "depth_image", None)
        T = getattr(cam, "T_world_camera", None)
        if rgb is None or depth is None or T is None:
            return None
        rgb_h, rgb_w = np.asarray(rgb).shape[:2]
        depth_h, depth_w = np.asarray(depth).shape[:2]
        K = getattr(cam, "intrinsics_rgb", None)
        if K is None:
            K = getattr(cam, "intrinsics_depth", None)
            if K is None:
                raise ValueError(
                    "[TargetVLM] cannot create temporary target without camera intrinsics"
                )
            K = np.asarray(K, dtype=float).copy()
            if (depth_h, depth_w) != (rgb_h, rgb_w):
                K[0, 0] *= float(rgb_w) / float(depth_w)
                K[1, 1] *= float(rgb_h) / float(depth_h)
                K[0, 2] *= float(rgb_w) / float(depth_w)
                K[1, 2] *= float(rgb_h) / float(depth_h)
        else:
            K = np.asarray(K, dtype=float)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError("[TargetVLM] camera intrinsics must be a finite 3x3 matrix")
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            raise ValueError("[TargetVLM] camera intrinsics have zero focal length")
        x1, y1, x2, y2 = [float(v) for v in bbox]
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)
        depth_m = MAX_RELIABLE_TARGET_DISTANCE_M
        p_cam = np.array(
            [
                (u - cx) / fx * depth_m,
                (v - cy) / fy * depth_m,
                depth_m,
            ],
            dtype=float,
        )
        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise ValueError("[TargetVLM] camera transform must be a finite 4x4 matrix")
        p_world = T[:3, :3] @ p_cam + T[:3, 3]
        if not np.isfinite(p_world[:2]).all():
            return None
        return (float(p_world[0]), float(p_world[1]))

    @staticmethod
    def _project_detection(cam_name: str, cam, det):
        from rtnav.modules.scenegraph.det3d import process_all_cameras_parallel

        cam_one = replace(cam, detections=[det])
        detections_3d, _ = process_all_cameras_parallel(
            camera_results={cam_name: cam_one},
            frame_idx=0,
        )
        if not detections_3d:
            return None
        obj = detections_3d[0]
        centroid = getattr(obj, "centroid", None)
        if centroid is None or not np.isfinite(centroid).all():
            return None
        return obj

    def _mobile_sam_model(self):
        if self._mobile_sam is None:
            from rtnav.modules.decision.mobile_sam import MobileSAM

            print("[TargetVLM] Loading MobileSAM for confirmed target masks")
            self._mobile_sam = MobileSAM()
        return self._mobile_sam

    @staticmethod
    def _without_confirmed_geometry(
        candidate: TargetCandidate,
        keep_temporary: bool = False,
    ) -> TargetCandidate:
        if keep_temporary and candidate.temporary_target:
            return replace(candidate, object_detection=None, node_id=None)
        return replace(
            candidate,
            object_detection=None,
            node_id=None,
            x=None,
            y=None,
            surface_x=None,
            surface_y=None,
            distance=float("inf"),
            temporary_target=False,
        )

    def _refine_confirmed_candidate(self, candidate: TargetCandidate) -> TargetCandidate:
        from rtnav.modules.scenegraph.det3d import project_mask_detection_to_3d

        cam = candidate.camera_result
        det = candidate.detection_entity

        def _temporary_boundary_target(reason: str) -> TargetCandidate:
            with self.shared_state.lock:
                robot_odom = getattr(self.shared_state.sensor, "latest_odom", None)
            temp_xy = self._temporary_boundary_xy(cam, det.bbox)
            if temp_xy is None:
                return self._without_confirmed_geometry(candidate, keep_temporary=True)
            robot_xy = (
                np.array([float(robot_odom[0]), float(robot_odom[1])], dtype=float)
                if robot_odom is not None
                else None
            )
            distance = (
                float(np.linalg.norm(np.asarray(temp_xy, dtype=float) - robot_xy))
                if robot_xy is not None
                else TEMPORARY_TARGET_DISTANCE_M
            )
            print(f"[TargetVLM] {reason}; using temporary boundary goal")
            return replace(
                candidate,
                object_detection=None,
                node_id=None,
                x=float(temp_xy[0]),
                y=float(temp_xy[1]),
                surface_x=float(temp_xy[0]),
                surface_y=float(temp_xy[1]),
                distance=distance,
                temporary_target=True,
            )

        def _temporary_detector_target(reason: str) -> TargetCandidate:
            if not candidate.has_geometry:
                return _temporary_boundary_target(reason)
            print(f"[TargetVLM] {reason}; using provisional detector geometry")
            return replace(
                candidate,
                object_detection=None,
                node_id=None,
                temporary_target=True,
            )

        observation_distance = float(
            getattr(candidate.object_detection, "robot_distance", float("inf"))
        )
        if observation_distance >= MAX_RELIABLE_TARGET_DISTANCE_M:
            return _temporary_boundary_target(
                f"confirmed target at {observation_distance:.2f}m "
                f">= {MAX_RELIABLE_TARGET_DISTANCE_M:.2f}m"
            )
        current_distance = float(candidate.distance)
        if current_distance > TEMPORARY_TARGET_DISTANCE_M:
            return _temporary_detector_target(
                f"confirmed target currently {current_distance:.2f}m away"
            )

        mask = self._mobile_sam_model().segment_bbox(
            cam.rgb_image,
            [float(v) for v in det.bbox],
        )
        candidate = replace(candidate, sam_mask=mask)

        projected = project_mask_detection_to_3d(
            cam,
            det,
            mask,
            frame_idx=0,
            erosion_size=MOBILE_SAM_EROSION_SIZE,
            failure_context=(
                f"camera={candidate.camera_name} canonical={candidate.canonical} "
                f"signature={candidate.signature}"
            ),
        )
        if projected is None:
            return _temporary_detector_target("MobileSAM mask projection failed")
        projected_distance = float(getattr(projected, "robot_distance", float("inf")))
        if not np.isfinite(projected_distance):
            raise RuntimeError("[TargetVLM] mask projection returned non-finite robot_distance")
        if projected_distance < 0.5:
            return self._without_confirmed_geometry(candidate)

        with self.shared_state.lock:
            robot_odom = getattr(self.shared_state.sensor, "latest_odom", None)

        T_world_camera = getattr(cam, "T_world_camera", None)
        camera_xyz = (
            np.asarray(T_world_camera[:3, 3], dtype=float) if T_world_camera is not None else None
        )
        surface_xy = _surface_xy_from_points(projected.point_cloud, camera_xyz)
        if surface_xy is None:
            surface_xy = np.asarray(projected.centroid[:2], dtype=float)
        robot_xy = (
            np.array([float(robot_odom[0]), float(robot_odom[1])], dtype=float)
            if robot_odom is not None
            else None
        )
        distance = (
            float(np.linalg.norm(np.asarray(projected.centroid[:2], dtype=float) - robot_xy))
            if robot_xy is not None
            else 0.0
        )
        temporary_reason = None
        if distance > TEMPORARY_TARGET_DISTANCE_M:
            temporary_reason = (
                f"confirmed target currently {distance:.2f}m away "
                f"> {TEMPORARY_TARGET_DISTANCE_M:.2f}m"
            )
        elif _mask_too_offset(mask):
            temporary_reason = "MobileSAM mask is too offset"
        if temporary_reason:
            print(f"[TargetVLM] {temporary_reason}; using provisional MobileSAM geometry")
            return replace(
                candidate,
                object_detection=None,
                node_id=None,
                x=float(projected.centroid[0]),
                y=float(projected.centroid[1]),
                surface_x=float(surface_xy[0]),
                surface_y=float(surface_xy[1]),
                distance=distance,
                temporary_target=True,
            )

        with self.shared_state.lock:
            sg = getattr(self.shared_state.scenegraph, "scene_graph", None)
            blacklisted_nodes = set(
                getattr(self.shared_state.target, "target_node_blacklist_ids", set()) or set()
            )
        node_id = self._nearest_scene_graph_node_id(sg, projected.centroid)
        if node_id is not None and int(node_id) in blacklisted_nodes:
            return self._without_confirmed_geometry(candidate)

        return replace(
            candidate,
            object_detection=projected,
            node_id=node_id,
            x=float(projected.centroid[0]),
            y=float(projected.centroid[1]),
            surface_x=float(surface_xy[0]),
            surface_y=float(surface_xy[1]),
            distance=distance,
            temporary_target=False,
        )

    @staticmethod
    def _nearest_scene_graph_node_id(sg, centroid) -> Optional[int]:
        if sg is None or not getattr(sg, "nodes", None):
            return None
        c = np.asarray(centroid, dtype=float)
        best_id = None
        best_dist = NODE_MATCH_RADIUS_M
        nodes = getattr(sg, "nodes", None)
        node_iter = nodes.values() if isinstance(nodes, dict) else nodes
        for node in node_iter:
            nc = getattr(node, "centroid", None)
            if nc is None:
                continue
            nc = np.asarray(nc, dtype=float)
            if nc.shape[0] < 2 or not np.isfinite(nc[:2]).all():
                continue
            dist = float(np.linalg.norm(nc[:2] - c[:2]))
            if dist < best_dist:
                best_dist = dist
                best_id = int(getattr(node, "node_id"))
        return best_id

    @staticmethod
    def _scene_graph_node(sg, node_id: Optional[int]):
        if sg is None or node_id is None:
            return None
        if hasattr(sg, "get_node"):
            node = sg.get_node(int(node_id))
            if node is not None:
                return node
        nodes = getattr(sg, "nodes", None)
        if isinstance(nodes, dict):
            return nodes.get(int(node_id))
        if nodes is not None:
            for candidate in nodes:
                if int(getattr(candidate, "node_id", -1)) == int(node_id):
                    return candidate
        return None

    def _scene_graph_node_label(self, sg, node_id: Optional[int]) -> Optional[str]:
        node = self._scene_graph_node(sg, node_id)
        label = str(getattr(node, "label", None) or getattr(node, "chosen_label", "") or "").strip()
        return label or None

    def _scene_graph_node_labels(self, sg, node_id: Optional[int]) -> list[str]:
        node = self._scene_graph_node(sg, node_id)
        if node is None:
            return []

        out: list[str] = []

        def add(label) -> None:
            text = str(label or "").strip()
            if text and text not in out:
                out.append(text)

        add(getattr(node, "label", None))
        add(getattr(node, "chosen_label", None))

        def count_sort_key(item) -> float:
            try:
                return -float(item[1])
            except (TypeError, ValueError):
                return 0.0

        label_counts = getattr(node, "label_counts", None)
        if isinstance(label_counts, dict):
            for label, _count in sorted(label_counts.items(), key=count_sort_key):
                add(label)

        for label in list(getattr(node, "past_labels", []) or []):
            add(label)

        for entry in list(getattr(node, "label_history", []) or []):
            if isinstance(entry, (list, tuple)) and entry:
                add(entry[0])
            else:
                add(entry)

        for label in list(getattr(node, "all_labels", []) or []):
            add(label)

        return out

    def _competing_scene_graph_label(
        self,
        sg,
        node_id: Optional[int],
        matching_labels,
        det=None,
    ) -> Optional[str]:
        labels = [
            label
            for label in self._scene_graph_node_labels(sg, node_id)
            if not label_matches_any(label, matching_labels)
        ]
        second_label = self._detection_second_label(det)
        if (
            second_label
            and not label_matches_any(second_label, matching_labels)
            and not label_matches_any(second_label, labels)
        ):
            labels.append(second_label)
        return ", ".join(labels) if labels else None

    @staticmethod
    def _detection_second_label(det) -> Optional[str]:
        top_k = list(getattr(det, "top_k_label_probs", []) or [])
        if len(top_k) < 2:
            return None
        entry = top_k[1]
        label = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
        text = str(label or "").strip()
        return text or None

    def verify_batch(
        self,
        candidates: list[TargetCandidate],
        target_name: str,
    ) -> list[VerificationResult]:
        if not candidates:
            return []
        model_id = get_vllm_model_id()
        max_workers = min(len(candidates), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(
                executor.map(
                    lambda candidate: self._verify_one(candidate, target_name, model_id),
                    candidates,
                )
            )

    def _verify_one(
        self,
        candidate: TargetCandidate,
        target_name: str,
        model_id: str,
    ) -> VerificationResult:
        from openai import OpenAI

        if candidate.competing_label:
            prompt = (
                "Look at the object inside the red bounding box. "
                f"Question: is this object a {target_name}? "
                f"Note: similar-looking objects may include {candidate.competing_label}. "
                "Answer yes only if the boxed object is clearly the target object. "
                "Return exactly one word: yes or no."
            )
        else:
            prompt = (
                "Look at the object inside the red bounding box. "
                f"Question: is this object a {target_name}? "
                "Answer yes only if the boxed object is clearly the target object. "
                "Return exactly one word: yes or no."
            )
        client = OpenAI(base_url=get_vllm_base_url(), api_key="EMPTY", timeout=120.0)
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(candidate.boxed_rgb)},
                        },
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        inference_time_s = time.perf_counter() - t0
        yes_prob, no_prob = _yes_no_token_probs(response.choices[0])
        if yes_prob + no_prob <= 0.0:
            raise RuntimeError("[TargetVLM] first-token logprobs did not include yes/no tokens")
        result = VerificationResult(
            candidate=candidate,
            yes_prob=yes_prob,
            no_prob=no_prob,
            yes_norm=_normalized_yes(yes_prob, no_prob),
            inference_time_s=inference_time_s,
        )
        return result

    def apply_results(self, results: list[VerificationResult]) -> list[dict]:
        accepted: list[dict] = []
        rejected_node_ids: set[int] = set()
        for result in results:
            self._queried_signatures.add(result.candidate.signature)
            if result.accepted:
                result.candidate = self._refine_confirmed_candidate(result.candidate)
                self._record_result(result)
                target = self._target_dict(result)
                if target is not None:
                    accepted.append(target)
                continue

            self._record_result(result)
            candidate = result.candidate
            if (
                candidate.node_id is None
                or not np.isfinite(result.yes_norm)
                or result.yes_norm > NODE_REJECTION_MAX_YES_SCORE
            ):
                continue

            node_id = int(candidate.node_id)
            count = self._node_rejection_counts.get(node_id, 0) + 1
            self._node_rejection_counts[node_id] = count
            if count >= NODE_REJECTION_BLACKLIST_COUNT:
                rejected_node_ids.add(node_id)
                if count == NODE_REJECTION_BLACKLIST_COUNT:
                    print(
                        f"[TargetVLM] permanently blacklisted node={node_id} after "
                        f"{count} rejections with score <= "
                        f"{NODE_REJECTION_MAX_YES_SCORE:.2f}"
                    )

        if not accepted and not rejected_node_ids:
            return []

        with self.shared_state.lock:
            if rejected_node_ids:
                self.shared_state.target.target_node_blacklist_ids.update(rejected_node_ids)
            blacklisted_nodes = set(self.shared_state.target.target_node_blacklist_ids)
            accepted = [
                target
                for target in accepted
                if target.get("node_id") is None
                or int(target["node_id"]) not in blacklisted_nodes
            ]
            existing = [
                target
                for target in list(getattr(self.shared_state.target, "target_goals", []) or [])
                if target.get("node_id") is None
                or int(target["node_id"]) not in blacklisted_nodes
            ]
            pending = list(
                getattr(self.shared_state.scenegraph, "pending_verified_detections", []) or []
            )
            for target in accepted:
                duplicate_idx = next(
                    (i for i, item in enumerate(existing) if self._same_target(item, target)),
                    None,
                )
                if duplicate_idx is None:
                    existing.append(target)
                    stored = target
                else:
                    stored = self._merge_duplicate_target(existing[duplicate_idx], target)
                    existing[duplicate_idx] = stored
                if (
                    stored.get("sg_request_id") is not None
                    and stored.get("_object_detection") is not None
                ):
                    pending.append(
                        {
                            "request_id": stored["sg_request_id"],
                            "detection": stored["_object_detection"],
                        }
                    )
            for target in existing:
                target.pop("_object_detection", None)
            self.shared_state.target.target_goals = existing
            self.shared_state.scenegraph.pending_verified_detections = pending
        return accepted

    def _record_result(self, result: VerificationResult):
        with self.shared_state.lock:
            logger = self.shared_state.system.vlm_logger
        if logger is None:
            return
        candidate = result.candidate
        logger.record_verification(
            _verification_figure(result),
            {
                "timestamp": time.time(),
                "observation_timestamp": candidate.timestamp,
                "accepted": result.accepted,
                "target": candidate.canonical,
                "detector_label": candidate.label,
                "camera": candidate.camera_name,
                "bbox": candidate.bbox,
                "node_id": candidate.node_id,
                "yes_prob": result.yes_prob,
                "no_prob": result.no_prob,
                "yes_normalized": result.yes_norm,
                "inference_time_s": result.inference_time_s,
                "competing_label": candidate.competing_label,
                "temporary_target": candidate.temporary_target,
            },
        )

    @staticmethod
    def _merge_duplicate_target(existing: dict, target: dict) -> dict:
        merged = dict(existing)
        merged.update(target)
        return merged

    def _target_dict(self, result: VerificationResult) -> Optional[dict]:
        c = result.candidate
        if not c.has_geometry and not c.has_temporary_goal:
            print("[TargetVLM] accepted target has no valid 3D projection; saved confirmation only")
            return None
        request_id = None
        if c.object_detection is not None:
            self._request_counter += 1
            request_id = f"vlm_target_{int(time.time() * 1000)}_{self._request_counter}"
        return {
            "x": c.x,
            "y": c.y,
            "surface_x": c.surface_x,
            "surface_y": c.surface_y,
            "label": c.canonical,
            "canonical": c.canonical,
            "confidence": c.confidence,
            "vlm_yes_norm": result.yes_norm,
            "vlm_confirmed": True,
            "node_id": c.node_id,
            "sg_request_id": request_id,
            "temporary_target": c.temporary_target,
            "_object_detection": c.object_detection,
        }

    @staticmethod
    def _same_target(a: dict, b: dict) -> bool:
        a_node = a.get("node_id")
        b_node = b.get("node_id")
        if a_node is not None and b_node is not None:
            return int(a_node) == int(b_node)
        if a.get("canonical") != b.get("canonical"):
            return False
        return (
            (float(a.get("x", float("inf"))) - float(b.get("x", float("-inf")))) ** 2
            + (float(a.get("y", float("inf"))) - float(b.get("y", float("-inf")))) ** 2
        ) < 0.5**2
