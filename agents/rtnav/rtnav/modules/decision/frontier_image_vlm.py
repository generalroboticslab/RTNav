"""Image-based VLM frontier selection."""

from __future__ import annotations

import base64
import math
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from rtnav.utils.vllm_utils import get_vllm_base_url, get_vllm_model_id


@dataclass
class FrontierImageEntry:
    entry_id: int
    frontier_idx: int
    centroid_xy: Tuple[float, float]
    rgb: Optional[np.ndarray]
    unexplored_direction_xy: Optional[np.ndarray] = None
    camera_name: str = ""
    camera_angle_rad: float = math.inf
    image_timestamp: float = 0.0
    image_centroid_xy: Optional[Tuple[float, float]] = None
    image_camera_xy: Optional[Tuple[float, float]] = None
    last_seen: float = 0.0


def _image_data_url(rgb: np.ndarray) -> str:
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("[FrontierImageVLM] failed to JPEG-encode frontier RGB")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


class FrontierImageVLM:
    """Maintain live frontier images and ask a VLM to choose among them."""

    MATCH_RADIUS_M = 0.5 # considered same frontier is new frontier is within this radius
    MAX_CAMERA_ANGLE_RAD = math.radians(30.0) # keep images where camera is within this angle of frontier
    MAX_IMAGE_FRONTIER_DRIFT_M = 0.5
    PRIMARY_IMAGE_RADIUS_M = 4.0
    # ponytail: single-frame depth can be noisy at edges; use multi-view ray tests if needed.
    DEPTH_VISIBILITY_TOLERANCE_M = 0.25

    def __init__(self, shared_state):
        self.shared_state = shared_state
        self._entries: List[FrontierImageEntry] = []
        self._current_camera_xy: Optional[Tuple[float, float]] = None
        self._next_id = 1
        self._query_counter = 0
        self._last_choice_sheet: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._entries = []
        self._current_camera_xy = None
        self._next_id = 1
        self._query_counter = 0
        self._last_choice_sheet = None
        self._publish_entries()
        self._publish_query_text("")

    def shutdown(self) -> None:
        return

    def drop_near(self, xy: Tuple[float, float], radius_m: float = 1.0) -> None:
        if xy is None:
            return
        x, y = float(xy[0]), float(xy[1])
        kept = []
        for entry in self._entries:
            ex, ey = entry.centroid_xy
            if (ex - x) ** 2 + (ey - y) ** 2 > radius_m * radius_m:
                kept.append(entry)
        if len(kept) != len(self._entries):
            self._entries = kept
            self._publish_entries()

    def sync(
        self,
        centroids_xy: Sequence[Sequence[float]],
        unexplored_directions_xy: Optional[Sequence[Any]] = None,
        cameras: Optional[dict[str, Any]] = None,
        recent_cameras: Optional[dict[str, Any]] = None,
        frontier_points_xy: Optional[Sequence[Any]] = None,
    ) -> List[FrontierImageEntry]:
        """Synchronize image entries to the current live frontier list."""
        now = time.time()
        centroids = [(float(c[0]), float(c[1])) for c in centroids_xy]
        directions = self._normalize_directions(unexplored_directions_xy, len(centroids))
        frontier_points = self._normalize_frontier_points(frontier_points_xy, len(centroids))
        cameras = self._camera_snapshot() if cameras is None else dict(cameras or {})
        if recent_cameras is None:
            recent_cameras = self._newest_camera_frames(cameras)
        else:
            recent_cameras = dict(recent_cameras or {})
        self._current_camera_xy = self._newest_camera_xy(recent_cameras or cameras)
        unmatched = list(self._entries)
        new_entries: List[FrontierImageEntry] = []

        for frontier_idx, centroid_xy in enumerate(centroids):
            unexplored_direction_xy = directions[frontier_idx]
            entry = self._pop_matching_entry(unmatched, centroid_xy)
            if entry is None:
                entry = FrontierImageEntry(
                    entry_id=self._next_id,
                    frontier_idx=frontier_idx,
                    centroid_xy=centroid_xy,
                    rgb=None,
                    unexplored_direction_xy=unexplored_direction_xy,
                    last_seen=now,
                )
                self._next_id += 1
            else:
                entry.frontier_idx = frontier_idx
                entry.centroid_xy = centroid_xy
                entry.unexplored_direction_xy = unexplored_direction_xy
                entry.last_seen = now

            if self._image_is_stale(entry, centroid_xy):
                self._clear_image(entry)

            candidate = self._best_camera_image(
                centroid_xy,
                frontier_points[frontier_idx],
                unexplored_direction_xy,
                cameras,
            )
            if candidate is not None:
                (
                    camera_name,
                    rgb,
                    angle_rad,
                    timestamp,
                    camera_xy,
                    frontier_marker,
                    arrow_direction,
                ) = candidate
                entry.rgb = self._annotate_frontier_overlay(
                    rgb,
                    frontier_marker,
                    arrow_direction,
                )
                entry.camera_name = camera_name
                entry.camera_angle_rad = angle_rad
                entry.image_timestamp = timestamp
                entry.image_centroid_xy = centroid_xy
                entry.image_camera_xy = camera_xy

            new_entries.append(entry)

        self._entries = new_entries
        self._publish_entries()
        return list(self._entries)

    def choose(self) -> Tuple[Optional[int], str]:
        """Return the current frontier index selected by the VLM."""
        ready, reason = self.ready_to_query()
        if not ready:
            return None, reason
        entries = self._selection_entries()
        missing = self._missing_image_entries()
        selected_ids = {entry.entry_id for entry in entries}
        excluded = [entry for entry in self._imaged_entries() if entry.entry_id not in selected_ids]
        if missing or excluded:
            missing_idxs = [int(entry.frontier_idx) for entry in missing]
            excluded_text = ", ".join(
                f"{entry.frontier_idx}:{self._entry_tier(entry)}" for entry in excluded
            )
            print(
                f"[FrontierImageVLM] prompting with {self.selection_scope()} "
                f"frontiers {len(entries)}/{len(self._entries)}; "
                f"excluded=[{excluded_text}], missing image indices={missing_idxs}"
            )
        target = self._target_name()
        prompt = self._build_prompt(target, entries)
        contact_sheet = self._build_contact_sheet(entries)
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(entry.rgb)},
            }
            for entry in entries
        ]
        content.append({"type": "text", "text": prompt})

        from openai import OpenAI

        self._query_counter += 1
        query_id = self._query_counter
        self._publish_query_text(f"Frontier VLM query {query_id}: running")
        t0 = time.perf_counter()
        client = OpenAI(base_url=get_vllm_base_url(), api_key="EMPTY", timeout=120.0)
        response = client.chat.completions.create(
            model=get_vllm_model_id(),
            messages=[
                {
                    "role": "system",
                    "content": "You are a navigation selector. Reply with only one allowed integer and no words.",
                },
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "add_vision_id": True,
                }
            },
        )
        elapsed = time.perf_counter() - t0
        text = (
            response.choices[0].message.content.strip()
            if response.choices[0].message.content
            else ""
        )
        index_logprobs = self._index_logprobs(response.choices[0], entries)
        if not index_logprobs:
            self._publish_query_text(
                f"Frontier VLM query {query_id}: no valid index, t={elapsed:.2f}s"
            )
            self._record_query(
                query_id,
                prompt,
                text,
                elapsed,
                entries,
                missing,
                contact_sheet,
                None,
                response.choices[0],
            )
            return None, "first-token logprobs did not include any valid frontier index"
        prompt_idx = max(index_logprobs, key=index_logprobs.get)
        entry = entries[prompt_idx]
        self._publish_query_text(
            f"Frontier VLM query {query_id}: D{prompt_idx + 1}->idx{entry.frontier_idx}, "
            f"t={elapsed:.2f}s"
        )
        selected_sheet = self._build_contact_sheet(
            entries,
            selected_prompt_idx=prompt_idx,
            elapsed=elapsed,
        )
        self._record_query(
            query_id,
            prompt,
            text,
            elapsed,
            entries,
            missing,
            contact_sheet,
            selected_sheet,
            response.choices[0],
            selected_prompt_idx=prompt_idx,
        )
        if self._recording_enabled():
            self._last_choice_sheet = selected_sheet
        print(
            f"[FrontierImageVLM] chose direction {prompt_idx + 1} -> "
            f"frontier {entry.frontier_idx} "
            f"({entry.centroid_xy[0]:.1f},{entry.centroid_xy[1]:.1f}) "
            f"target='{target}' t={elapsed:.2f}s decoded={text!r}"
        )
        return int(entry.frontier_idx), ""

    def ready_to_query(self) -> Tuple[bool, str]:
        n_frontiers = self.frontier_count()
        n_candidates = self.candidate_count()
        if n_frontiers == 0:
            return False, "no live frontiers"
        if n_candidates == 0:
            return False, "no live frontier images"
        return True, ""

    def frontier_count(self) -> int:
        return len(self._entries)

    def candidate_count(self) -> int:
        return len(self._selection_entries())

    def single_candidate_frontier_idx(self) -> Optional[int]:
        if len(self._entries) == 1:
            return int(self._entries[0].frontier_idx)
        entries = self._selection_entries()
        if len(entries) != 1:
            return None
        return int(entries[0].frontier_idx)

    def stash_single_choice_sheet(self) -> None:
        """Record a no-VLM single-candidate pick and its excluded alternatives."""
        entries = self._selection_entries()
        if len(entries) != 1:
            return
        input_sheet = self._build_contact_sheet(entries)
        selected_sheet = self._build_contact_sheet(entries, selected_prompt_idx=0)
        if self._recording_enabled():
            self._last_choice_sheet = selected_sheet

        selected_entry = entries[0]
        excluded_entries = [
            entry for entry in self._imaged_entries() if entry.entry_id != selected_entry.entry_id
        ]
        excluded = [self._entry_record(entry) for entry in excluded_entries]
        with self.shared_state.lock:
            logger = self.shared_state.system.vlm_logger
        if logger is None:
            return
        selected = {**self._entry_record(selected_entry), "direction": 1}
        logger.record_frontier(
            input_sheet,
            selected_sheet,
            {
                "timestamp": time.time(),
                "bypassed_vlm": True,
                "candidate_scope": self.selection_scope(),
                "frontiers": [selected],
                "excluded_frontier_indices": [item["frontier_idx"] for item in excluded],
                "excluded_frontiers": excluded,
                "missing_frontier_indices": [
                    entry.frontier_idx for entry in self._missing_image_entries()
                ],
                "selected": {"direction": 1, "frontier_idx": selected_entry.frontier_idx},
            },
            self._excluded_contact_sheet(excluded_entries),
        )

    def consume_choice_sheet(self) -> Optional[np.ndarray]:
        """Return and clear the sheet of the most recent image-frontier pick."""
        sheet, self._last_choice_sheet = self._last_choice_sheet, None
        return sheet

    def _imaged_entries(self) -> List[FrontierImageEntry]:
        return [entry for entry in self._entries if entry.rgb is not None]

    def _selection_entries(self) -> List[FrontierImageEntry]:
        """Use nearby-capture images first, then farther viewpoints."""
        imaged = self._imaged_entries()
        primary = [entry for entry in imaged if self._is_primary(entry)]
        return primary or [entry for entry in imaged if not self._is_primary(entry)]

    def selection_scope(self) -> str:
        imaged = self._imaged_entries()
        if any(self._is_primary(entry) for entry in imaged):
            return "primary"
        return "secondary" if imaged else "none"

    def _image_distance_m(self, entry: FrontierImageEntry) -> Optional[float]:
        return self._camera_distance_m(entry.image_camera_xy)

    def _camera_distance_m(self, camera_xy) -> Optional[float]:
        if self._current_camera_xy is None or camera_xy is None:
            return None
        dx = float(camera_xy[0]) - float(self._current_camera_xy[0])
        dy = float(camera_xy[1]) - float(self._current_camera_xy[1])
        return float(math.hypot(dx, dy))

    def _is_primary(self, entry: FrontierImageEntry) -> bool:
        distance = self._image_distance_m(entry)
        return distance is not None and distance <= self.PRIMARY_IMAGE_RADIUS_M

    def _entry_tier(self, entry: FrontierImageEntry) -> str:
        return "primary" if self._is_primary(entry) else "secondary"

    def _recording_enabled(self) -> bool:
        with self.shared_state.lock:
            return self.shared_state.system.decision_recorder is not None

    def _record_query(
        self,
        query_id,
        prompt,
        response,
        elapsed,
        entries,
        missing,
        input_sheet,
        selected_sheet,
        choice,
        selected_prompt_idx=None,
    ):
        with self.shared_state.lock:
            logger = self.shared_state.system.vlm_logger
        if logger is None:
            return
        queried_ids = {entry.entry_id for entry in entries}
        excluded_entries = [
            entry for entry in self._imaged_entries() if entry.entry_id not in queried_ids
        ]
        excluded = [self._entry_record(entry) for entry in excluded_entries]
        top_tokens = [
            {"token": token, "probability": probability}
            for token, probability in self._top_token_probabilities(choice)
        ]
        frontiers = []
        for prompt_idx, entry in enumerate(entries):
            frontiers.append(
                {
                    "direction": prompt_idx + 1,
                    "frontier_idx": entry.frontier_idx,
                    "centroid_xy": entry.centroid_xy,
                    "camera": entry.camera_name,
                    "camera_angle_deg": (
                        math.degrees(entry.camera_angle_rad)
                        if math.isfinite(entry.camera_angle_rad)
                        else None
                    ),
                    "image_distance_m": self._image_distance_m(entry),
                    "tier": self._entry_tier(entry),
                }
            )
        selected = None
        if selected_prompt_idx is not None:
            selected = {
                "direction": selected_prompt_idx + 1,
                "frontier_idx": entries[selected_prompt_idx].frontier_idx,
            }
        logger.record_frontier(
            input_sheet,
            selected_sheet,
            {
                "query_id": query_id,
                "timestamp": time.time(),
                "prompt": prompt,
                "response": response,
                "inference_time_s": elapsed,
                "frontiers": frontiers,
                "candidate_scope": self.selection_scope(),
                "excluded_frontier_indices": [item["frontier_idx"] for item in excluded],
                "excluded_frontiers": excluded,
                "missing_frontier_indices": [entry.frontier_idx for entry in missing],
                "selected": selected,
                "top_tokens": top_tokens,
            },
            self._excluded_contact_sheet(excluded_entries),
        )

    def _missing_image_entries(self) -> List[FrontierImageEntry]:
        return [entry for entry in self._entries if entry.rgb is None]

    def _entry_record(self, entry: FrontierImageEntry) -> dict[str, Any]:
        return {
            "frontier_idx": int(entry.frontier_idx),
            "centroid_xy": tuple(entry.centroid_xy),
            "camera": entry.camera_name,
            "camera_angle_deg": (
                float(math.degrees(entry.camera_angle_rad))
                if math.isfinite(entry.camera_angle_rad)
                else None
            ),
            "image_timestamp": float(entry.image_timestamp),
            "image_camera_xy": entry.image_camera_xy,
            "image_distance_m": self._image_distance_m(entry),
            "tier": self._entry_tier(entry),
        }

    def _excluded_contact_sheet(self, entries) -> Optional[np.ndarray]:
        if not entries:
            return None
        labels = [f"F{entry.frontier_idx} {self._entry_tier(entry)}" for entry in entries]
        return self._build_contact_sheet(entries, labels=labels)

    @classmethod
    def _clear_image(cls, entry: FrontierImageEntry) -> None:
        entry.rgb = None
        entry.camera_name = ""
        entry.camera_angle_rad = math.inf
        entry.image_timestamp = 0.0
        entry.image_centroid_xy = None
        entry.image_camera_xy = None

    @classmethod
    def _image_is_stale(
        cls,
        entry: FrontierImageEntry,
        centroid_xy: Tuple[float, float],
    ) -> bool:
        if entry.rgb is None:
            return False
        if entry.image_centroid_xy is not None:
            dx = float(entry.image_centroid_xy[0]) - float(centroid_xy[0])
            dy = float(entry.image_centroid_xy[1]) - float(centroid_xy[1])
            if dx * dx + dy * dy > cls.MAX_IMAGE_FRONTIER_DRIFT_M**2:
                return True
        return False

    @staticmethod
    def _normalize_directions(unexplored_directions_xy, n_frontiers: int):
        directions = [None] * n_frontiers
        if unexplored_directions_xy is None:
            return directions
        for i in range(min(n_frontiers, len(unexplored_directions_xy))):
            direction = unexplored_directions_xy[i]
            if direction is None:
                continue
            direction = np.asarray(direction, dtype=float).reshape(-1)
            if len(direction) < 2 or not np.all(np.isfinite(direction[:2])):
                continue
            directions[i] = direction[:2]
        return directions

    @staticmethod
    def _normalize_frontier_points(frontier_points_xy, n_frontiers: int):
        points = [None] * n_frontiers
        if frontier_points_xy is None:
            return points
        for i in range(min(n_frontiers, len(frontier_points_xy))):
            candidate = np.asarray(frontier_points_xy[i], dtype=float)
            if candidate.ndim == 1:
                candidate = candidate.reshape(1, -1)
            if candidate.ndim != 2 or candidate.shape[1] < 2:
                continue
            candidate = candidate[:, :2]
            candidate = candidate[np.all(np.isfinite(candidate), axis=1)]
            if len(candidate):
                points[i] = candidate
        return points

    @staticmethod
    def _choice_top_logprobs(choice) -> List[Tuple[str, float]]:
        if choice.logprobs is None:
            raise RuntimeError("vLLM response did not include token logprobs")
        if hasattr(choice.logprobs, "top_logprobs") and choice.logprobs.top_logprobs:
            return [
                (str(token), float(logprob))
                for token, logprob in choice.logprobs.top_logprobs[0].items()
            ]
        if hasattr(choice.logprobs, "content") and choice.logprobs.content:
            top_logprobs = choice.logprobs.content[0].top_logprobs
            return [(str(entry.token), float(entry.logprob)) for entry in top_logprobs]
        raise RuntimeError("vLLM response did not include first-token top logprobs")

    @classmethod
    def _top_token_probabilities(cls, choice, limit: int = 10) -> List[Tuple[str, float]]:
        top_tokens = sorted(
            cls._choice_top_logprobs(choice),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
        return [
            (token, min(1.0, max(0.0, math.exp(logprob))))
            for token, logprob in top_tokens
        ]

    def _index_logprobs(
        self,
        choice,
        entries: Sequence[FrontierImageEntry],
    ) -> dict[int, float]:
        top_logprobs = self._choice_top_logprobs(choice)
        scores: dict[int, float] = {}
        for token, logprob in top_logprobs:
            token_text = str(token).strip()
            try:
                idx = int(token_text)
            except ValueError:
                continue
            if 1 <= idx <= len(entries):
                prompt_idx = idx - 1
                scores[prompt_idx] = max(
                    scores.get(prompt_idx, -math.inf),
                    float(logprob),
                )
        return scores

    def _pop_matching_entry(
        self,
        entries: List[FrontierImageEntry],
        centroid_xy: Tuple[float, float],
    ) -> Optional[FrontierImageEntry]:
        if not entries:
            return None
        cx, cy = centroid_xy
        d2s = [
            (entry.centroid_xy[0] - cx) ** 2 + (entry.centroid_xy[1] - cy) ** 2 for entry in entries
        ]
        best_idx = int(np.argmin(d2s))
        if d2s[best_idx] > self.MATCH_RADIUS_M * self.MATCH_RADIUS_M:
            return None
        return entries.pop(best_idx)

    def _camera_snapshot(self) -> dict[str, Any]:
        with self.shared_state.lock:
            cameras = dict(getattr(self.shared_state.sensor, "latest_cameras", {}) or {})
        return cameras

    @staticmethod
    def _camera_xy(T) -> Optional[Tuple[float, float]]:
        if T is None:
            return None
        T = np.asarray(T, dtype=float)
        if T.shape[0] < 3 or T.shape[1] < 4:
            return None
        return float(T[0, 3]), float(T[1, 3])

    @classmethod
    def _newest_camera_xy(cls, cameras) -> Optional[Tuple[float, float]]:
        newest_timestamp = -math.inf
        newest_xy = None
        for cam in cameras.values():
            camera_xy = cls._camera_xy(getattr(cam, "T_world_cam", None))
            timestamp = float(getattr(cam, "timestamp", 0.0) or 0.0)
            if camera_xy is not None and timestamp >= newest_timestamp:
                newest_timestamp = timestamp
                newest_xy = camera_xy
        return newest_xy

    @staticmethod
    def _newest_camera_frames(cameras) -> dict[str, Any]:
        """Infer the latest snapshot when callers do not identify it explicitly."""
        if not cameras:
            return {}
        timestamps = {
            name: float(getattr(camera, "timestamp", 0.0) or 0.0)
            for name, camera in cameras.items()
        }
        newest_timestamp = max(timestamps.values())
        return {
            name: cameras[name]
            for name, timestamp in timestamps.items()
            if timestamp == newest_timestamp
        }

    def _best_camera_image(
        self,
        centroid_xy,
        frontier_points_xy,
        unexplored_direction_xy,
        cameras,
    ):
        best = None
        best_score = (math.inf, math.inf, math.inf, math.inf)
        cx, cy = float(centroid_xy[0]), float(centroid_xy[1])
        for name, cam in cameras.items():
            rgb = getattr(cam, "rgb", None)
            T = getattr(cam, "T_world_cam", None)
            if rgb is None or T is None:
                continue
            T = np.asarray(T, dtype=float)
            if T.shape[0] < 3 or T.shape[1] < 3:
                continue
            camera_xy = self._camera_xy(T)
            if camera_xy is None:
                continue
            cam_xy = np.asarray(camera_xy, dtype=float)
            to_frontier = np.array([cx - cam_xy[0], cy - cam_xy[1]], dtype=float)
            norm = float(np.linalg.norm(to_frontier))
            if norm <= 1e-6:
                continue
            to_frontier /= norm
            forward = T[:2, 2].astype(float)
            f_norm = float(np.linalg.norm(forward))
            if f_norm <= 1e-6:
                continue
            forward /= f_norm
            cos_angle = float(np.clip(np.dot(to_frontier, forward), -1.0, 1.0))
            angle_rad = float(math.acos(cos_angle))
            if angle_rad > self.MAX_CAMERA_ANGLE_RAD:
                continue
            timestamp = float(getattr(cam, "timestamp", 0.0) or 0.0)
            arrow_direction = self._frontier_screen_direction(unexplored_direction_xy, T)
            marker = self._frontier_image_point(centroid_xy, cam, T)
            item = (
                name,
                rgb,
                angle_rad,
                timestamp,
                camera_xy,
                marker,
                arrow_direction,
            )
            distance = self._camera_distance_m(camera_xy)
            fully_occluded = not self._frontier_has_visible_point(
                frontier_points_xy, centroid_xy, cam, T
            )
            score = (
                fully_occluded,
                distance if distance is not None else math.inf,
                angle_rad,
                -timestamp,
            )
            if score < best_score:
                best = item
                best_score = score
        return best

    @staticmethod
    def _frontier_screen_direction(direction_xy, T: np.ndarray):
        """Reduce the map-derived unexplored direction to screen-left or screen-right."""
        if direction_xy is None:
            return None
        direction = np.asarray(direction_xy, dtype=float).reshape(-1)
        if len(direction) < 2 or not np.all(np.isfinite(direction[:2])):
            return None
        direction = direction[:2]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return None
        direction /= norm

        right = np.asarray(T[:2, 0], dtype=float)
        right_norm = float(np.linalg.norm(right))
        if right_norm <= 1e-6:
            return None
        right /= right_norm
        horizontal = float(direction @ right)
        if abs(horizontal) <= 1e-6:
            return None
        return (1.0 if horizontal > 0 else -1.0, 0.0)

    def _frontier_image_point(self, point_xy, cam, T: np.ndarray):
        projection = self._frontier_point_projection(point_xy, cam, T)
        if projection is None:
            return None
        rgb = getattr(cam, "rgb", None)
        h, w = np.asarray(rgb).shape[:2]
        x_px, y_px, _ = projection
        return (
            float(np.clip(x_px, 0, w - 1)),
            float(np.clip(y_px, 0, h - 1)),
        )

    def _frontier_point_projection(self, point_xy, cam, T: np.ndarray):
        rgb = getattr(cam, "rgb", None)
        if rgb is None:
            return None
        h, w = np.asarray(rgb).shape[:2]
        if h <= 1 or w <= 1:
            return None
        point = np.asarray(point_xy, dtype=float).reshape(-1)
        if len(point) < 2 or not np.all(np.isfinite(point[:2])):
            return None

        K = getattr(cam, "intrinsics", None)
        try:
            K = np.asarray(K, dtype=float)
        except Exception:
            K = None
        if K is not None and K.shape[0] >= 3 and K.shape[1] >= 3:
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            center_x = float(K[0, 2])
            center_y = float(K[1, 2])
        else:
            fx = (w / 2.0) / math.tan(self.MAX_CAMERA_ANGLE_RAD)
            fy = fx
            center_x = w / 2.0
            center_y = h / 2.0
        if not np.all(np.isfinite([fx, fy, center_x, center_y])):
            return None
        if abs(fx) <= 1e-6 or abs(fy) <= 1e-6:
            return None

        relative_world = np.array(
            [point[0] - T[0, 3], point[1] - T[1, 3], -T[2, 3]],
            dtype=float,
        )
        point_cam = relative_world @ T[:3, :3]
        if not np.all(np.isfinite(point_cam)) or point_cam[2] <= 1e-6:
            return None
        x_px = center_x + fx * point_cam[0] / point_cam[2]
        y_px = center_y + fy * point_cam[1] / point_cam[2]
        return float(x_px), float(y_px), float(point_cam[2])

    def _frontier_has_visible_point(
        self,
        points_xy,
        centroid_xy,
        cam,
        T: np.ndarray,
    ) -> bool:
        points = np.asarray(
            [centroid_xy] if points_xy is None else points_xy,
            dtype=float,
        ).reshape(-1, 2)
        h, w = np.asarray(cam.rgb).shape[:2]
        for point in points:
            projection = self._frontier_point_projection(point, cam, T)
            if projection is None:
                continue
            x_px, y_px, expected_depth = projection
            if not (0 <= x_px < w and 0 <= y_px < h):
                continue
            if self._frontier_point_passes_depth(
                cam,
                x_px,
                y_px,
                expected_depth,
                (h, w),
            ):
                return True
        return False

    def _frontier_point_passes_depth(
        self,
        cam,
        x_px: float,
        y_px: float,
        expected_depth: float,
        image_shape,
    ) -> bool:
        depth = getattr(cam, "depth", None)
        if depth is None:
            return True
        depth = np.asarray(depth)
        if depth.ndim < 2 or depth.shape[0] == 0 or depth.shape[1] == 0:
            return True
        if depth.ndim > 2:
            depth = depth[..., 0]

        image_h, image_w = image_shape
        depth_h, depth_w = depth.shape[:2]
        u = int(round((x_px + 0.5) * depth_w / image_w - 0.5))
        v = int(round((y_px + 0.5) * depth_h / image_h - 0.5))
        u = max(0, min(depth_w - 1, u))
        v = max(0, min(depth_h - 1, v))
        patch = depth[
            max(0, v - 1) : min(depth_h, v + 2),
            max(0, u - 1) : min(depth_w, u + 2),
        ]
        min_depth = float(getattr(cam, "min_depth_m", 0.0) or 0.0)
        max_depth = float(getattr(cam, "max_depth_m", math.inf) or math.inf)
        valid = (
            np.isfinite(patch)
            & (patch > min_depth + 1e-3)
            & (patch < max_depth - 1e-3)
        )
        if not np.any(valid):
            return True
        return bool(
            np.max(patch[valid]) + self.DEPTH_VISIBILITY_TOLERANCE_M
            >= expected_depth
        )

    @staticmethod
    def _annotate_frontier_overlay(
        rgb: np.ndarray,
        frontier_marker: Optional[Tuple[float, float]],
        arrow_direction: Optional[Tuple[float, float]],
        label: Optional[str] = None,
    ) -> np.ndarray:
        image = np.asarray(rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        annotated = np.ascontiguousarray(image.copy())
        h, w = annotated.shape[:2]
        if h <= 1 or w <= 1:
            return annotated

        label_x = None
        if frontier_marker is not None:
            x, y = frontier_marker
            post = (
                max(0, min(w - 1, int(round(float(x))))),
                max(0, min(h - 1, int(round(float(y))))),
            )
            label_x = float(post[0])
            line_thickness = max(4, int(round(min(h, w) / 80.0)))

            direction = None
            if arrow_direction is not None:
                direction = np.asarray(arrow_direction, dtype=float)
                norm = float(np.linalg.norm(direction))
                if norm > 1e-6:
                    direction /= norm
                else:
                    direction = None

            for color, thickness in (
                ((0, 0, 0), line_thickness + 2),
                ((255, 0, 0), line_thickness),
            ):
                cv2.line(
                    annotated,
                    (post[0], 0),
                    post,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

            if direction is not None:
                length = max(24, min(64, int(round(min(h, w) * 0.14))))
                edge_margin = min(
                    max(8, int(math.ceil((line_thickness + 4) / 2.0)) + 1),
                    (w - 1) // 2,
                )
                length = min(length, max(1, w - 1 - 2 * edge_margin))
                midpoint = np.array([post[0], int(round(0.5 * post[1]))], dtype=float)
                start = midpoint - 0.5 * length * direction
                tip = midpoint + 0.5 * length * direction
                left = min(start[0], tip[0])
                right = max(start[0], tip[0])
                if left < edge_margin:
                    start[0] += edge_margin - left
                    tip[0] += edge_margin - left
                elif right > w - 1 - edge_margin:
                    start[0] -= right - (w - 1 - edge_margin)
                    tip[0] -= right - (w - 1 - edge_margin)
                start[1] = np.clip(start[1], 8, h - 9)
                tip[1] = np.clip(tip[1], 8, h - 9)
                start_px = tuple(np.rint(start).astype(int))
                tip_px = tuple(np.rint(tip).astype(int))
                cv2.arrowedLine(
                    annotated,
                    start_px,
                    tip_px,
                    (0, 0, 0),
                    line_thickness + 4,
                    cv2.LINE_AA,
                    tipLength=0.32,
                )
                cv2.arrowedLine(
                    annotated,
                    start_px,
                    tip_px,
                    (255, 0, 0),
                    line_thickness,
                    cv2.LINE_AA,
                    tipLength=0.32,
                )

        if label:
            FrontierImageVLM._draw_frontier_label(annotated, label, label_x)
        return annotated

    @staticmethod
    def _draw_frontier_label(image, label: str, anchor_x: Optional[float]) -> None:
        h, w = image.shape[:2]
        text = str(label)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(0.8, w / 700.0))
        thickness = max(1, int(round(min(h, w) / 240.0)))
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        if text_w > w - 16:
            scale *= max(0.35, (w - 16) / max(1, text_w))
            (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

        if anchor_x is None:
            x0 = 8
        else:
            x0 = int(round(float(anchor_x) - text_w / 2.0 - 8))
            x0 = max(6, min(w - text_w - 18, x0))
        y0 = 8
        x1 = min(w - 1, x0 + text_w + 16)
        y1 = min(h - 1, y0 + text_h + baseline + 12)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.rectangle(image, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (255, 0, 0), -1)
        cv2.putText(
            image,
            text,
            (x0 + 8, y1 - baseline - 6),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _target_name(self) -> str:
        with self.shared_state.lock:
            goal_category = str(getattr(self.shared_state.task, "goal_category", "") or "").strip()
        return goal_category or "the target object"

    @staticmethod
    def _build_prompt(target: str, entries: Sequence[FrontierImageEntry]) -> str:
        return f"""Target: {target} (not found yet).

Each transparent red goal post marks one frontier.
Its arrow points from explored space toward unexplored space.
Which Picture's frontier is traversable and most likely to lead to the target's usual room?

Answer only 1-{len(entries)}.
"""

    def _publish_entries(self) -> None:
        published = []
        for entry in self._entries:
            if entry.rgb is None:
                continue
            published.append(
                {
                    "id": int(entry.entry_id),
                    "frontier_idx": int(entry.frontier_idx),
                    "centroid_xy": tuple(entry.centroid_xy),
                    "rgb": entry.rgb,
                    "camera_name": entry.camera_name,
                    "camera_angle_deg": (
                        float(math.degrees(entry.camera_angle_rad))
                        if math.isfinite(entry.camera_angle_rad)
                        else None
                    ),
                    "image_timestamp": float(entry.image_timestamp),
                    "image_camera_xy": entry.image_camera_xy,
                    "image_distance_m": self._image_distance_m(entry),
                    "tier": self._entry_tier(entry),
                }
            )
        with self.shared_state.lock:
            self.shared_state.frontier.frontier_images = published

    def _publish_query_text(self, text: str) -> None:
        with self.shared_state.lock:
            self.shared_state.frontier.frontier_vlm_query_text = str(text or "")
            self.shared_state.frontier.frontier_vlm_query_time = time.time() if text else 0.0

    @staticmethod
    def _build_contact_sheet(
        entries: Sequence[FrontierImageEntry],
        selected_prompt_idx: Optional[int] = None,
        elapsed: Optional[float] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        thumbs = []
        thumb_h = 180
        pad = 6
        for prompt_idx, entry in enumerate(entries):
            rgb = np.asarray(entry.rgb)
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            h, w = rgb.shape[:2]
            scale = float(thumb_h) / max(1.0, float(h))
            thumb_w = max(1, int(round(float(w) * scale)))
            thumb = cv2.resize(rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            # Panel labels match the direction numbers in the VLM prompt.
            label = labels[prompt_idx] if labels is not None else str(prompt_idx + 1)
            FrontierImageVLM._draw_frontier_label(thumb, label, thumb_w / 2.0)
            canvas = thumb

            selected = selected_prompt_idx is not None and prompt_idx == selected_prompt_idx
            color = (255, 0, 0) if selected else (180, 180, 180)
            thickness = 8 if selected else 2
            cv2.rectangle(
                canvas,
                (pad // 2, pad // 2),
                (
                    canvas.shape[1] - pad // 2 - 1,
                    canvas.shape[0] - pad // 2 - 1,
                ),
                color,
                thickness,
                cv2.LINE_AA,
            )
            thumbs.append(canvas)

        if not thumbs:
            return np.empty((0, 0, 3), dtype=np.uint8)

        gap = 8
        cell_h = max(img.shape[0] for img in thumbs)
        cell_w = max(img.shape[1] for img in thumbs)

        # Choose the grid whose rendered pixel dimensions are closest to square.
        # The camera panels are landscape, so simply using ceil(sqrt(n)) columns
        # still produces a very wide sheet (for example, 3x2 for six panels).
        # Accounting for the panel aspect ratio instead gives 2x3 in that case.
        grid_options = []
        for cols in range(1, len(thumbs) + 1):
            rows = int(math.ceil(len(thumbs) / cols))
            grid_w = cols * cell_w + gap * (cols - 1)
            grid_h = rows * cell_h + gap * (rows - 1)
            aspect_error = abs(math.log(grid_w / max(1, grid_h)))
            empty_cells = rows * cols - len(thumbs)
            grid_options.append((aspect_error, empty_cells, rows, cols, grid_h, grid_w))
        _, _, rows, cols, grid_h, grid_w = min(grid_options)

        grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
        for idx, img in enumerate(thumbs):
            row_idx, col_idx = divmod(idx, cols)
            x = col_idx * (cell_w + gap) + (cell_w - img.shape[1]) // 2
            y = row_idx * (cell_h + gap) + (cell_h - img.shape[0]) // 2
            grid[y : y + img.shape[0], x : x + img.shape[1]] = img

        if elapsed is None:
            return grid

        header_h = 30
        sheet = np.full((grid_h + header_h, grid_w, 3), 255, dtype=np.uint8)
        sheet[header_h:, :] = grid
        cv2.putText(
            sheet,
            f"VLM inference time: {elapsed:.2f}s",
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return sheet
