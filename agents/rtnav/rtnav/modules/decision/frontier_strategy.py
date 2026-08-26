"""Choose a sticky exploration frontier from images, then geometry."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rtnav.modules.decision.frontier_image_vlm import FrontierImageVLM


@dataclass
class FailedFrontier:
    xy: Tuple[float, float]
    strikes: int = 1


class FrontierStrategy:
    """Select one frontier and keep it until it is reached or fails.

    Multiple aligned images are ranked by the VLM. One image is selected
    directly. Geometry handles missing images and semantic non-answers.
    """

    MAX_FAILURE_STRIKES = 2
    HEADING_WEIGHT_M = 1.5

    def __init__(self, shared_state, frontier_cfg, decision_cfg):
        self.shared_state = shared_state
        self.image_vlm = FrontierImageVLM(shared_state)

        self._sticky_match_m = float(decision_cfg.frontier_position_threshold)
        self._failure_radius_m = float(frontier_cfg.strategy_failure_radius_m)
        self._fresh_wait_max_s = float(frontier_cfg.strategy_fresh_wait_max_s)
        self._cycle_grid_m = float(frontier_cfg.strategy_cycle_grid_m)

        self._last_frontier_xy: Optional[np.ndarray] = None
        self._last_value = float("-inf")
        self._last_selection_time = 0.0
        self._need_fresh_after_ts = 0.0
        self._last_fresh_wait_log = 0.0
        self._last_wait_log = 0.0

        self._failures: list[FailedFrontier] = []
        self._acyclic_history: set = set()
        self._last_state_action = None

    def select_goal(
        self,
        context: Dict[str, Any],
        allow_image_vlm: bool = True,
        allow_geometric_fallback: bool = True,
    ):
        now = time.time()
        if self._last_frontier_xy is None and now - self._last_selection_time < 0.1:
            return None, "rapid re-selection guard"

        frontier_output = context.get("frontier_output")
        robot_pose = context.get("robot_pose")
        obstacle_map = context.get("obstacle_map")
        if frontier_output is None:
            self._log_wait("No frontier output yet")
            return None, "no frontier_output"
        if robot_pose is None or obstacle_map is None:
            self._log_wait("No robot pose or obstacle map")
            return None, "no robot_pose or obs_map"

        clusters = list(frontier_output.frontier_clusters)
        centroids = list(frontier_output.frontier_centroids)
        if not clusters:
            self._log_wait("No frontier clusters yet")
            return None, "no frontier clusters"

        centroids_xy = list(obstacle_map.px_to_xy(np.asarray(centroids)))
        robot_xy = np.asarray(robot_pose[:2], dtype=np.float32)

        keep = [i for i, xy in enumerate(centroids_xy) if not self._is_banned(xy)]
        if not keep:
            print("[Frontier] all frontiers banned — ignoring failures to keep exploring")
            keep = list(range(len(clusters)))
        clusters = [clusters[i] for i in keep]
        centroids = [centroids[i] for i in keep]
        centroids_xy = [centroids_xy[i] for i in keep]

        sticky = self._try_sticky(clusters, centroids, centroids_xy, obstacle_map, robot_xy)
        if sticky is not None:
            return self._finish_selection(sticky)
        self._clear_sticky_state()

        if not self._frontiers_are_fresh(frontier_output, now):
            self._last_selection_time = now
            return None, "waiting for fresh frontier detection"

        self._sync_images(frontier_output, obstacle_map, centroids_xy)
        candidate_count = self.image_vlm.candidate_count() if allow_image_vlm else 0

        if allow_image_vlm:
            idx = self.image_vlm.single_candidate_frontier_idx()
            result = self._select_index(
                idx, clusters, centroids, centroids_xy, obstacle_map, robot_xy
            )
            if result is not None:
                self.image_vlm.stash_single_choice_sheet()
                scope = self.image_vlm.selection_scope()
                if scope == "none":
                    scope = "live"
                print(f"[FrontierImageVLM] selected the only {scope} frontier")
                return self._finish_selection(result)

        if candidate_count > 1:
            ready, reason = self.image_vlm.ready_to_query()
            idx = None
            if ready:
                gate = self.shared_state.inference
                gate.pause_for_vllm()
                try:
                    idx, reason = self.image_vlm.choose()
                except Exception as exc:
                    reason = f"image VLM error: {exc}"
                finally:
                    gate.resume_after_vllm()
            if idx is not None:
                result = self._select_index(
                    idx, clusters, centroids, centroids_xy, obstacle_map, robot_xy
                )
                if result is not None:
                    return self._finish_selection(result)
                reason = f"VLM-selected frontier {idx} is not navigable"
            print(f"[FrontierImageVLM] no selection ({reason or 'unknown reason'})")
            if not allow_geometric_fallback:
                self._last_selection_time = time.time()
                return None, reason or "image VLM did not select a frontier"

        if not allow_geometric_fallback:
            self._last_selection_time = time.time()
            return None, "geometric fallback disallowed"

        result = self._select_geometric(
            clusters,
            centroids,
            centroids_xy,
            robot_xy,
            obstacle_map,
            float(robot_pose[2]),
        )
        if result is None:
            return None, "no navigable frontiers"
        result = self._avoid_cycle(
            result, robot_xy, clusters, centroids, centroids_xy, obstacle_map
        )
        return self._finish_selection(result)

    def _finish_selection(self, result):
        self._last_selection_time = time.time()
        with self.shared_state.lock:
            self.shared_state.frontier.chosen_frontier_xy = self._visual_xy(result[0])
        return result

    def _select_index(self, idx, clusters, centroids, centroids_xy, obstacle_map, robot_xy):
        if idx is None or not 0 <= int(idx) < len(clusters):
            return None
        idx = int(idx)
        nav = obstacle_map.find_navigable_frontier_goal(centroids[idx])
        if nav is None:
            return None
        nav_xy, nav_px = nav
        if self._is_banned(nav_xy):
            return None
        self._remember_choice(idx, nav_xy, centroids_xy, robot_xy)
        print(f"[FrontierImageVLM] F{idx} -> ({nav_xy[0]:.1f}, {nav_xy[1]:.1f})")
        return nav_xy, nav_px, clusters[idx]

    def _select_geometric(
        self, clusters, centroids, centroids_xy, robot_xy, obstacle_map, robot_yaw
    ):
        heading = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
        choices = []
        for idx, (_, centroid_px, centroid_xy) in enumerate(
            zip(clusters, centroids, centroids_xy)
        ):
            nav = obstacle_map.find_navigable_frontier_goal(centroid_px)
            if nav is None or self._is_banned(nav[0]):
                continue
            delta = np.asarray(centroid_xy, dtype=float) - robot_xy
            distance = float(np.linalg.norm(delta))
            alignment = 0.0 if distance < 1e-6 else float(delta @ heading / distance)
            score = distance - self.HEADING_WEIGHT_M * alignment
            choices.append((score, distance, idx, nav))
        if not choices:
            return None
        score, distance, idx, (nav_xy, nav_px) = min(choices)
        self._remember_choice(idx, nav_xy, centroids_xy, robot_xy)
        print(
            f"[Frontier] geometric -> ({nav_xy[0]:.1f}, {nav_xy[1]:.1f}) "
            f"distance={distance:.1f}m score={score:.2f}"
        )
        return nav_xy, nav_px, clusters[idx]

    def _remember_choice(self, idx, nav_xy, centroids_xy, robot_xy):
        centroid = np.asarray(centroids_xy[idx], dtype=np.float32)
        self._last_frontier_xy = centroid.copy()
        self._last_value = -float(np.linalg.norm(centroid - robot_xy))

    def _try_sticky(self, clusters, centroids, centroids_xy, obstacle_map, robot_xy):
        if self._last_frontier_xy is None or not centroids_xy:
            return None
        points = np.asarray(centroids_xy, dtype=np.float32)
        distances = np.linalg.norm(points - self._last_frontier_xy, axis=1)
        idx = int(np.argmin(distances))
        if distances[idx] > self._sticky_match_m:
            return None
        current_value = -float(np.linalg.norm(points[idx] - robot_xy))
        if current_value + 0.01 <= self._last_value:
            return None
        nav = obstacle_map.find_navigable_frontier_goal(centroids[idx])
        if nav is None or self._is_banned(nav[0]):
            return None
        self._last_frontier_xy = points[idx].copy()
        self._last_value = current_value
        return nav[0], nav[1], clusters[idx]

    def refresh_goal(self, context: Dict[str, Any]):
        """Follow the selected frontier as its live centroid moves."""
        if self._last_frontier_xy is None:
            return None, "no selected frontier"

        frontier_output = context.get("frontier_output")
        obstacle_map = context.get("obstacle_map")
        if frontier_output is None or obstacle_map is None:
            return None, "frontier state unavailable"

        clusters = list(frontier_output.frontier_clusters)
        centroids = list(frontier_output.frontier_centroids)
        if not clusters or not centroids:
            return None, "no live frontiers"
        count = min(len(clusters), len(centroids))
        clusters, centroids = clusters[:count], centroids[:count]

        centroids_xy = np.asarray(obstacle_map.px_to_xy(np.asarray(centroids)), dtype=float)
        distances = np.linalg.norm(centroids_xy - self._last_frontier_xy, axis=1)
        idx = int(np.argmin(distances))
        if distances[idx] > self._sticky_match_m:
            return None, f"nearest live frontier is {distances[idx]:.2f}m away"

        nav = obstacle_map.find_navigable_frontier_goal(centroids[idx])
        if nav is None or self._is_banned(nav[0]):
            return None, "frontier has no navigable approach"

        self._last_frontier_xy = centroids_xy[idx].astype(np.float32)
        robot_pose = context.get("robot_pose")
        if robot_pose is not None:
            self._last_value = -float(
                np.linalg.norm(self._last_frontier_xy - np.asarray(robot_pose[:2], dtype=float))
            )
        with self.shared_state.lock:
            self.shared_state.frontier.chosen_frontier_xy = tuple(
                float(value) for value in self._last_frontier_xy
            )
        return (nav[0], nav[1], clusters[idx]), ""

    def _avoid_cycle(self, result, robot_xy, clusters, centroids, centroids_xy, obstacle_map):
        def cell(xy):
            return tuple(round(float(v) / self._cycle_grid_m) for v in xy[:2])

        state_action = cell(robot_xy), cell(result[0])
        if state_action == self._last_state_action:
            return result
        if state_action in self._acyclic_history and len(centroids_xy) > 1:
            candidates = sorted(
                range(len(centroids_xy)),
                key=lambda i: np.linalg.norm(np.asarray(centroids_xy[i]) - robot_xy),
                reverse=True,
            )
            for idx in candidates:
                candidate_state = cell(robot_xy), cell(centroids_xy[idx])
                if candidate_state in self._acyclic_history:
                    continue
                nav = obstacle_map.find_navigable_frontier_goal(centroids[idx])
                if nav is None or self._is_banned(nav[0]):
                    continue
                result = nav[0], nav[1], clusters[idx]
                self._last_frontier_xy = np.asarray(centroids_xy[idx]).copy()
                state_action = cell(robot_xy), cell(nav[0])
                print(f"[Frontier] broke a selection cycle with farthest frontier F{idx}")
                break
        self._acyclic_history.add(state_action)
        self._last_state_action = state_action
        return result

    def on_goal_reached(self, goal_xy):
        print(f"[Frontier] reached ({goal_xy[0]:.1f}, {goal_xy[1]:.1f})")
        self.image_vlm.drop_near(goal_xy)
        self._remove_failures_near(goal_xy, radius_m=1.5)
        self._need_fresh_after_ts = time.time()
        self._last_state_action = None
        self._clear_sticky_state()
        self._clear_viz_state()

    def on_goal_failed(self, goal_xy, reason: str = ""):
        print(f"[Frontier] failed ({goal_xy[0]:.1f}, {goal_xy[1]:.1f}) reason={reason!r}")
        self.image_vlm.drop_near(goal_xy)
        self._clear_sticky_state()
        self._clear_viz_state()
        if reason == "frontier_vanished":
            return
        self._need_fresh_after_ts = time.time()
        if reason not in ("path_unreachable", "no_progress"):
            return
        failure = self._nearest_failure(goal_xy)
        if failure is None:
            failure = FailedFrontier((float(goal_xy[0]), float(goal_xy[1])))
            self._failures.append(failure)
        else:
            failure.strikes += 1
            failure.xy = (float(goal_xy[0]), float(goal_xy[1]))
        print(
            f"[Frontier] failure strike {failure.strikes}/{self.MAX_FAILURE_STRIKES} "
            f"near ({failure.xy[0]:.1f}, {failure.xy[1]:.1f})"
        )

    def _nearest_failure(self, xy):
        radius2 = self._failure_radius_m**2
        matches = [
            f for f in self._failures if (f.xy[0] - xy[0]) ** 2 + (f.xy[1] - xy[1]) ** 2 <= radius2
        ]
        if not matches:
            return None
        return min(matches, key=lambda f: (f.xy[0] - xy[0]) ** 2 + (f.xy[1] - xy[1]) ** 2)

    def _is_banned(self, xy) -> bool:
        failure = self._nearest_failure(xy)
        return failure is not None and failure.strikes >= self.MAX_FAILURE_STRIKES

    def _remove_failures_near(self, xy, radius_m):
        radius2 = radius_m**2
        self._failures = [
            f for f in self._failures if (f.xy[0] - xy[0]) ** 2 + (f.xy[1] - xy[1]) ** 2 > radius2
        ]

    def _frontiers_are_fresh(self, frontier_output, now):
        if self._need_fresh_after_ts <= 0.0:
            return True
        timestamp = float(getattr(frontier_output, "timestamp", 0.0) or 0.0)
        waited = now - self._need_fresh_after_ts
        if timestamp > self._need_fresh_after_ts or waited >= self._fresh_wait_max_s:
            self._need_fresh_after_ts = 0.0
            return True
        if now - self._last_fresh_wait_log >= 0.5:
            self._last_fresh_wait_log = now
            print(f"[Frontier] waiting for a post-result frontier update ({waited:.1f}s)")
        return False

    def sync_frontier_images(self, frontier_output, obstacle_map):
        if frontier_output is None or obstacle_map is None:
            self.image_vlm.sync([])
            return
        timestamp = float(getattr(frontier_output, "timestamp", 0.0) or 0.0)
        if self._need_fresh_after_ts > 0.0 and timestamp <= self._need_fresh_after_ts:
            return
        centroids = getattr(frontier_output, "frontier_centroids", None)
        if centroids is None or len(centroids) == 0:
            self.image_vlm.sync([])
            return
        try:
            centroids_xy = obstacle_map.px_to_xy(np.asarray(centroids))
        except Exception as exc:
            print(f"[FrontierImageVLM] image sync failed: {exc}")
            return
        self._sync_images(frontier_output, obstacle_map, centroids_xy)

    def _sync_images(self, frontier_output, obstacle_map, centroids_xy):
        cameras, recent_cameras = self._camera_sources()
        self.image_vlm.sync(
            centroids_xy,
            unexplored_directions_xy=getattr(
                frontier_output, "frontier_unexplored_directions", None
            ),
            cameras=cameras,
            recent_cameras=recent_cameras,
        )

    def _camera_sources(self):
        with self.shared_state.lock:
            snapshots = list(self.shared_state.sensor.camera_history)
            snapshots += list(self.shared_state.frontier.frontier_output_camera_history)
            latest = dict(self.shared_state.frontier.frontier_output_cameras)
            current = dict(self.shared_state.sensor.latest_cameras)
        snapshots += ([latest] if latest else []) + ([current] if current else [])
        cameras = {}
        for snapshot_idx, snapshot in enumerate(snapshots):
            for name, camera in (snapshot or {}).items():
                timestamp = float(getattr(camera, "timestamp", 0.0) or 0.0)
                cameras[f"{name}@{snapshot_idx}:{timestamp:.6f}"] = camera
        recent_cameras = {}
        for name, camera in current.items():
            timestamp = float(getattr(camera, "timestamp", 0.0) or 0.0)
            recent_cameras[f"{name}@current:{timestamp:.6f}"] = camera
        return cameras, recent_cameras

    def _clear_sticky_state(self):
        self._last_frontier_xy = None
        self._last_value = float("-inf")

    def _visual_xy(self, fallback):
        if self._last_frontier_xy is None:
            return tuple(float(v) for v in fallback[:2])
        return tuple(float(v) for v in self._last_frontier_xy[:2])

    def _clear_viz_state(self):
        with self.shared_state.lock:
            self.shared_state.frontier.chosen_frontier_xy = None

    def _log_wait(self, message):
        now = time.time()
        if now - self._last_wait_log >= 1.5:
            print(f"[Frontier] {message}")
            self._last_wait_log = now

    def reset(self):
        self._clear_sticky_state()
        self._last_selection_time = 0.0
        self._need_fresh_after_ts = 0.0
        self._failures = []
        self._acyclic_history = set()
        self._last_state_action = None
        self.image_vlm.reset()
        self._clear_viz_state()
        print("[Frontier] reset")

    def shutdown(self):
        self.image_vlm.shutdown()

    def consume_choice_sheet(self):
        return self.image_vlm.consume_choice_sheet()

    def get_state(self):
        return {
            "frontier_xy": self._visual_xy(self._last_frontier_xy)
            if self._last_frontier_xy is not None
            else None,
            "failed_frontiers": len(self._failures),
        }
