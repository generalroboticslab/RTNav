"""Choose and pursue verified object targets."""

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rtnav.modules.decision.target_utils import (
    TARGET_OBSERVE_MIN_TARGET_MASS,
    TARGET_OBSERVE_PROMOTE_MASS,
    normalize_target_names,
    target_confirmation_passes,
    target_confirmation_summary,
    target_matches,
    target_observe_abort_reason,
    target_observe_summary,
)


class TargetStrategy:
    """Navigate to a target and commit when the robot is close enough."""

    TARGET_COMMIT_DIST_M = 0.25
    TARGET_STALLED_COMMIT_DIST_M = 0.7
    TARGET_STALLED_TIMEOUT_S = 5.0
    TARGET_PROGRESS_DELTA_M = 0.15
    TEMPORARY_TARGET_REACHED_DIST_M = 0.2

    def __init__(self, shared_state, decision_cfg, camera_cfg):
        self.shared_state = shared_state
        dcfg = decision_cfg
        self._camera_cfg = camera_cfg
        self.SG_CENTROID_REFRESH_M = float(dcfg.find_sg_centroid_refresh_m)
        self._target_names = []
        self._current_target = None
        self._current_goal = None
        self._complete = False
        self._best_target_dist = None
        self._best_target_dist_t = None

    def _node_ready_for_target(self, node, target_set=None) -> bool:
        return target_confirmation_passes(
            node, target_set if target_set is not None else self._target_label_set()
        )

    def select_goal(
        self,
        param: Any,
        robot_xy: np.ndarray,
        obstacle_map,
    ) -> Optional[Tuple]:
        self._target_names = normalize_target_names(param)

        def _rank(t):
            dist = self._target_surface_dist(t, robot_xy)
            return (
                bool(t.get("temporary_target")),
                -float(t.get("vlm_yes_norm", t.get("target_mass", 0.0))),
                dist if dist is not None else float("inf"),
                -float(t.get("confidence", 0.0)),
            )

        # Already navigating; reuse goal (or rebuild if on_reached cleared it)
        if self._current_target is not None and not self._complete:
            if self._current_target.get("temporary_target"):
                if self._current_goal is None:
                    self._current_goal = self._make_goal(
                        self._current_target, obstacle_map, robot_xy
                    )
                return self._current_goal
            if self._current_target.get("vlm_confirmed"):
                self._refresh_target_from_sg()
            if self._current_goal is None:
                self._current_goal = self._make_goal(self._current_target, obstacle_map, robot_xy)
            return self._current_goal

        candidates = self._collect_vlm_targets()
        if not candidates:
            return None

        # Like VLFM's object map, centered evidence supersedes provisional edge evidence.
        target = min(candidates, key=_rank) if candidates else None
        if target is not None and len(candidates) > 1:
            picked_dist = (
                (float(target["x"]) - float(robot_xy[0])) ** 2
                + (float(target["y"]) - float(robot_xy[1])) ** 2
            ) ** 0.5
            print(
                f"[Target] picked '{target.get('label')}' at "
                f"({target['x']:+.1f},{target['y']:+.1f}) "
                f"center_dist={picked_dist:.1f}m yes={target.get('vlm_yes_norm', 0.0):.2f} "
                f"v={target.get('view_count', 0)} c={target.get('confidence', 0.0):.2f} "
                f"over {len(candidates) - 1} other candidate(s)"
            )

        if target is None:
            return None

        self._start_target(
            target,
            robot_xy=robot_xy,
            observe_mode=bool(target.get("observe_mode")),
        )
        self._current_goal = self._make_goal(target, obstacle_map, robot_xy)
        return self._current_goal

    def _start_target(
        self,
        target: Dict,
        robot_xy: np.ndarray = None,
        observe_mode: Optional[bool] = None,
    ):
        self._current_target = target
        self._current_target["target_set"] = sorted(self._target_label_set())
        if observe_mode is not None:
            self._current_target["observe_mode"] = bool(observe_mode)
        if self._current_target.get("vlm_confirmed"):
            self._current_target["observe_mode"] = False
            self._current_target["observe_reason"] = None
        elif self._current_target.get("observe_mode"):
            self._current_target.setdefault("observe_reason", "low_confidence")
        if not self._current_target.get("vlm_confirmed"):
            self._update_observe_mode_state(self._current_target, robot_xy, log=True)
        self._publish_target_node_id(target.get("node_id"))
        self._complete = False
        self._reset_target_progress()

    def _target_label_set(self) -> set:
        names_lc = {n.lower().strip() for n in self._target_names if str(n).strip()}
        with self.shared_state.lock:
            valid_synonyms = set(getattr(self.shared_state.task, "synonym_to_canonical", {}).keys())
            goal_category = (
                str(getattr(self.shared_state.task, "goal_category", "") or "").lower().strip()
            )
        target_set = names_lc | {goal_category} | valid_synonyms
        target_set.discard("")
        return target_set

    def _target_full_lock_range_m(self) -> float:
        max_depth = getattr(self._camera_cfg, "max_depth", None)
        if max_depth is None:
            raise ValueError("[Target] camera config does not define max_depth")
        return float(max_depth) * 0.99

    def _target_surface_dist(self, target: Dict, robot_xy: np.ndarray = None) -> Optional[float]:
        if target is None or robot_xy is None:
            return None
        sx, sy = self._target_surface_xy(target)
        return float(np.hypot(float(robot_xy[0]) - sx, float(robot_xy[1]) - sy))

    def _update_observe_mode_state(
        self, target: Dict, robot_xy: np.ndarray = None, log: bool = False
    ) -> bool:
        dist = self._target_surface_dist(target, robot_xy)
        if dist is None:
            return bool(target.get("observe_mode"))
        limit = self._target_full_lock_range_m()
        was = bool(target.get("observe_mode"))
        old_reason = target.get("observe_reason")
        if dist >= limit:
            target["observe_mode"] = True
            target["observe_reason"] = "depth_limited"
        elif target.get("observe_reason") == "depth_limited":
            target["observe_mode"] = False
            target["observe_reason"] = None
        target["observe_distance"] = dist
        now = bool(target.get("observe_mode"))
        if log and (now != was or target.get("observe_reason") != old_reason):
            label = target.get("label", target.get("canonical", "?"))
            if now:
                print(
                    f"[Target] observing '{label}' at {dist:.2f}m "
                    f"({target.get('observe_reason')}); target geometry/evidence will refresh"
                )
            else:
                print(f"[Target] '{label}' promoted to full target lock at {dist:.2f}m")
        return now

    def _refresh_target_from_sg(self) -> bool:
        """Pull the current SG node for the active target.

        Fully locked targets keep their original surface goal. Observe-mode
        targets refresh geometry/evidence until promoted or aborted.
        """
        if self._current_target is None:
            return False
        if self._current_target.get("temporary_target"):
            return False
        node_id = self._current_target.get("node_id")
        if node_id is None and not self._current_target.get("vlm_confirmed"):
            return False
        with self.shared_state.lock:
            sg = getattr(self.shared_state.scenegraph, "scene_graph", None)
        if sg is None or not getattr(sg, "nodes", None):
            return False
        live_node = None
        if node_id is not None:
            for nd in sg.nodes:
                if int(getattr(nd, "node_id", 0)) == int(node_id):
                    live_node = nd
                    break
        # Fallback when node_id changed (SG merge): match by label + proximity
        if live_node is None:
            old_x = float(self._current_target.get("x", 0.0))
            old_y = float(self._current_target.get("y", 0.0))
            best = None
            best_d2 = (2.0) ** 2  # 2m max for an evidence-based rematch
            target_set = self._target_label_set()
            for nd in sg.nodes:
                if not self._current_target.get(
                    "vlm_confirmed"
                ) and not self._node_ready_for_target(nd, target_set):
                    continue
                c = getattr(nd, "centroid", None)
                if c is None:
                    continue
                d2 = (float(c[0]) - old_x) ** 2 + (float(c[1]) - old_y) ** 2
                if d2 < best_d2:
                    best = nd
                    best_d2 = d2
            if best is not None:
                live_node = best
                # Rebind id so next refresh is fast AND viz keeps tracking
                new_id = int(getattr(best, "node_id", 0))
                self._current_target["node_id"] = new_id
                self._publish_target_node_id(new_id)
        if live_node is None:
            return False
        if self._current_target.get("observe_mode"):
            target_set = set(self._current_target.get("target_set") or [])
            if not target_set:
                target_set = self._target_label_set()
            reason = target_observe_abort_reason(live_node, target_set, 0)
            if reason:
                self.abandon(reason=reason)
                return False
            promoted_mass = None
            promote = target_observe_summary(
                live_node,
                target_set,
                threshold=TARGET_OBSERVE_PROMOTE_MASS,
            )
            if promote["passes"]:
                promoted_mass = float(promote["target_mass"])
                self._current_target["observe_mode"] = False
                self._current_target["observe_reason"] = None
                with self.shared_state.lock:
                    robot_odom = getattr(self.shared_state.sensor, "latest_odom", None)
                robot_xy_for_lock = (
                    np.array([float(robot_odom[0]), float(robot_odom[1])])
                    if robot_odom is not None
                    else None
                )
                self._update_observe_mode_state(self._current_target, robot_xy_for_lock, log=False)
                if not self._current_target.get("observe_mode"):
                    label = self._get_label()
                    print(
                        f"[Target] observe target '{label}' promoted: "
                        f"target evidence {promoted_mass:.2f}"
                    )
        elif not self._current_target.get("vlm_confirmed") and self._should_abort_for_relabel(
            live_node
        ):
            return False
        was_observing = bool(self._current_target.get("observe_mode"))
        if not was_observing:
            return False
        c = getattr(live_node, "centroid", None)
        if c is None:
            return False
        new_x = float(c[0])
        new_y = float(c[1])
        with self.shared_state.lock:
            robot_odom = getattr(self.shared_state.sensor, "latest_odom", None)
        robot_xy = (
            np.array([float(robot_odom[0]), float(robot_odom[1])])
            if robot_odom is not None
            else None
        )
        surface_x, surface_y = self._surface_xy_from_node(live_node, robot_xy=robot_xy)
        old_x = float(self._current_target.get("x", new_x))
        old_y = float(self._current_target.get("y", new_y))
        old_surface_x = float(self._current_target.get("surface_x", old_x))
        old_surface_y = float(self._current_target.get("surface_y", old_y))
        centroid_drift = float(np.hypot(new_x - old_x, new_y - old_y))
        surface_drift = float(np.hypot(surface_x - old_surface_x, surface_y - old_surface_y))
        self._current_target["x"] = new_x
        self._current_target["y"] = new_y
        self._current_target["surface_x"] = surface_x
        self._current_target["surface_y"] = surface_y
        self._update_observe_mode_state(self._current_target, robot_xy, log=True)
        if not self._current_target.get("observe_mode"):
            return False
        if max(centroid_drift, surface_drift) >= self.SG_CENTROID_REFRESH_M or (
            was_observing and max(centroid_drift, surface_drift) > 0.0
        ):
            print(
                f"[Target] SG target drift for node #{node_id}: "
                f"center={centroid_drift:.2f}m surface={surface_drift:.2f}m — "
                f"refreshing nav goal"
            )
            # Clear current_goal so select_goal re-issues a fresh nav goal
            # at the updated target surface next tick.
            self._current_goal = None
            return True
        # Sub-threshold drift: still update the stored xy so the next
        # commit-check uses the freshest value, but don't tear down the nav goal.
        return False

    def _should_abort_for_relabel(self, live_node) -> bool:
        if self._current_target is None:
            return False
        target_set = set(self._current_target.get("target_set") or [])
        if not target_set:
            target_set = self._target_label_set()

        summary = target_confirmation_summary(live_node, target_set)
        target_mass = float(summary["target_mass"])
        non_label = str(summary["max_non_target_label"])
        non_mass = float(summary["max_non_target_mass"])
        reason = None
        if target_mass < TARGET_OBSERVE_MIN_TARGET_MASS:
            reason = f"target evidence {target_mass:.2f} < {TARGET_OBSERVE_MIN_TARGET_MASS:.2f}"
        elif non_label and non_mass > target_mass:
            reason = (
                f"non-target '{non_label}' mass {non_mass:.2f} > target evidence {target_mass:.2f}"
            )
        if reason is not None:
            self.abandon(reason=reason)
            return True
        return False

    def on_reached(self) -> bool:
        """Nav reached the target surface goal."""
        if self._current_target is None:
            self._complete = True
            return True
        self._refresh_target_from_sg()
        if self._current_target is None:
            return False
        if self._current_target.get("observe_mode"):
            label = self._get_label()
            target_xy = self._target_surface_xy(self._current_target)
            with self.shared_state.lock:
                robot_odom = getattr(self.shared_state.sensor, "latest_odom", None)
            dist = (
                float(
                    np.hypot(
                        float(robot_odom[0]) - target_xy[0],
                        float(robot_odom[1]) - target_xy[1],
                    )
                )
                if robot_odom is not None
                else None
            )
            if dist is not None and dist <= self.TARGET_COMMIT_DIST_M:
                return self._commit(label, target_xy[0], target_xy[1], dist)
            dist_txt = "unknown" if dist is None else f"{float(dist):.2f}m"
            print(
                f"[Target] reached observe target '{label}' shell at {dist_txt}; "
                f"refreshing instead of committing"
            )
            self._current_goal = None
            return False
        label = self._get_label()
        target_xy = self._target_surface_xy(self._current_target)
        with self.shared_state.lock:
            robot_odom = getattr(self.shared_state.sensor, "latest_odom", None)
        dist = (
            float(
                np.hypot(
                    float(robot_odom[0]) - target_xy[0],
                    float(robot_odom[1]) - target_xy[1],
                )
            )
            if robot_odom is not None
            else None
        )
        return self._commit(label, target_xy[0], target_xy[1], dist)

    def _commit(self, label: str, tx: float, ty: float, dist: Optional[float]) -> bool:
        """Mark the episode target complete so /agent_done fires next tick."""
        dist_txt = "unknown" if dist is None else f"{dist:.2f}m"
        print(
            f"[Target] ── commit: '{label}' at nav_goal "
            f"({dist_txt} from target surface). Marking complete for /agent_done."
        )
        self._complete = True
        with self.shared_state.lock:
            self.shared_state.task.complete = True
        print(
            f"[Target] ✓ COMMITTING '{label}' at ({tx:.2f},{ty:.2f}) — "
            f"target navigation terminated, /agent_done imminent."
        )
        self._current_target = None
        self._publish_target_node_id(None)
        self._current_goal = None
        return True

    def commit_if_close(self, robot_xy) -> bool:
        """Commit a confirmed target as soon as the robot reaches its surface."""
        if self._current_target is None or self._complete:
            return False
        if self._current_target.get("temporary_target"):
            return False
        if self._current_target.get("vlm_confirmed"):
            self._refresh_target_from_sg()
            if self._current_target is None:
                return False

        target_xy = self._target_surface_xy(self._current_target)
        dist = float(
            np.hypot(
                float(robot_xy[0]) - target_xy[0],
                float(robot_xy[1]) - target_xy[1],
            )
        )
        if dist <= self.TARGET_COMMIT_DIST_M:
            label = self._get_label()
            print(
                f"[Target] confirmed target '{label}' within commit radius "
                f"({dist:.2f}m <= {self.TARGET_COMMIT_DIST_M:.2f}m) — committing target"
            )
            return self._commit(label, target_xy[0], target_xy[1], dist)

        if dist > self.TARGET_STALLED_COMMIT_DIST_M:
            self._reset_target_progress()
            return False

        now = time.monotonic()
        if self._best_target_dist is None:
            self._best_target_dist = dist
            self._best_target_dist_t = now
            return False
        if dist < self._best_target_dist - self.TARGET_PROGRESS_DELTA_M:
            self._best_target_dist = dist
            self._best_target_dist_t = now
            return False
        if now - self._best_target_dist_t < self.TARGET_STALLED_TIMEOUT_S:
            return False

        label = self._get_label()
        print(
            f"[Target] confirmed target '{label}' stalled within "
            f"{self.TARGET_STALLED_COMMIT_DIST_M:.2f}m for "
            f"{self.TARGET_STALLED_TIMEOUT_S:.1f}s — committing target"
        )
        return self._commit(label, target_xy[0], target_xy[1], dist)

    def finish_temporary_if_close(self, robot_xy) -> bool:
        target = self._current_target
        if target is None or not target.get("temporary_target"):
            return False
        dist = self._target_surface_dist(target, robot_xy)
        if dist is None or dist > self.TEMPORARY_TARGET_REACHED_DIST_M:
            return False
        print(
            f"[Target] temporary target reached "
            f"({dist:.2f}m <= {self.TEMPORARY_TARGET_REACHED_DIST_M:.2f}m)"
        )
        self.finish_temporary_target(reached=True)
        return True

    def _reset_target_progress(self) -> None:
        self._best_target_dist = None
        self._best_target_dist_t = None

    def abort_if_reclassified(self) -> bool:
        """Return True only when the active target fails the post-confirmation
        relabel vote rule and was aborted."""
        if self._current_target is None or self._complete:
            return False
        if self._current_target.get("temporary_target"):
            return False
        if self._current_target.get("vlm_confirmed"):
            return False
        before = self._current_target
        self._refresh_target_from_sg()
        return before is not None and self._current_target is None

    def abandon(self, reason: str = None):
        """Drop an active target whose updated evidence no longer matches."""
        if self._current_target is None:
            return
        tx = float(self._current_target.get("x", 0.0))
        ty = float(self._current_target.get("y", 0.0))
        detail = reason or "target aborted"
        print(
            f"[Target] target '{self._get_label()}' at ({tx:.2f},{ty:.2f}) "
            f"aborted: {detail} — resuming frontier"
        )
        self._current_target = None
        self._current_goal = None
        self._complete = False
        self._publish_target_node_id(None)

    def finish_temporary_target(self, reached: bool, reason: str = "") -> None:
        if self._current_target is None:
            return
        target = self._current_target
        if not target.get("temporary_target"):
            return
        label = target.get("label", target.get("canonical", "?"))
        tx = float(target.get("x", 0.0))
        ty = float(target.get("y", 0.0))
        request_id = target.get("sg_request_id")
        with self.shared_state.lock:
            goals = list(getattr(self.shared_state.target, "target_goals", []) or [])
            kept = []
            for item in goals:
                same_request = request_id is not None and item.get("sg_request_id") == request_id
                same_xy = (
                    item.get("temporary_target")
                    and abs(float(item.get("x", 0.0)) - tx) < 1e-3
                    and abs(float(item.get("y", 0.0)) - ty) < 1e-3
                    and item.get("label", item.get("canonical")) == label
                )
                if same_request or same_xy:
                    continue
                kept.append(item)
            self.shared_state.target.target_goals = kept
        if reached:
            outcome = "reached"
        elif reason == "replaced_by_confirmed_target":
            outcome = "replaced by confirmed target"
        else:
            outcome = f"failed ({reason or 'unknown'})"
        print(
            f"[Target] temporary target '{label}' {outcome} at ({tx:.2f},{ty:.2f}); "
            "re-enabling VLM confirmation"
        )
        self._current_target = None
        self._current_goal = None
        self._complete = False
        self._publish_target_node_id(None)

    def _publish_target_node_id(self, node_id: Optional[int]) -> None:
        """Publish chosen SG node id so viz/probes can resolve live centroid."""
        try:
            with self.shared_state.lock:
                self.shared_state.target.current_target_node_id = (
                    int(node_id) if node_id is not None else None
                )
        except Exception as exc:
            print(f"[Target] publish_target_node_id failed (non-fatal): {exc}")

    @property
    def current_target(self):
        """Node we're driving to, or None."""
        return self._current_target

    def _collect_vlm_targets(self, verbose: bool = True) -> list:
        if not self._target_names:
            return []
        names = [n.lower().strip() for n in self._target_names if str(n).strip()]
        with self.shared_state.lock:
            targets = list(getattr(self.shared_state.target, "target_goals", []) or [])
            blacklisted_nodes = set(
                getattr(self.shared_state.target, "target_node_blacklist_ids", set()) or set()
            )
        out = []
        for target in targets:
            if not target.get("vlm_confirmed"):
                continue
            node_id = target.get("node_id")
            if node_id is not None and int(node_id) in blacklisted_nodes:
                continue
            if not target_matches(target, names):
                continue
            candidate = dict(target)
            candidate["x"] = float(target["x"])
            candidate["y"] = float(target["y"])
            out.append(candidate)
        if out and verbose:
            summary = [
                f"({t['x']:+.1f},{t['y']:+.1f},{t['label']},yes={t.get('vlm_yes_norm', 0.0):.2f})"
                for t in out
            ]
            print(f"[Target] sourced {len(out)} VLM-confirmed target(s): {summary}")
        return out

    def on_failed(self, reason: str):
        """Nav failed.
        policy_stop/no_progress → commit the confirmed target
        aborts are handled only by post-confirmation relabel votes
        """
        if self._current_target is None:
            self._complete = True
            return

        label = self._get_label()

        if reason in {"policy_stop", "no_progress"}:
            self._refresh_target_from_sg()
            if self._current_target is None:
                self._complete = False
                return
            target_xy = self._target_surface_xy(self._current_target)
            with self.shared_state.lock:
                robot_odom = getattr(self.shared_state.sensor, "latest_odom", None)
            dist = (
                float(
                    np.hypot(
                        float(robot_odom[0]) - target_xy[0],
                        float(robot_odom[1]) - target_xy[1],
                    )
                )
                if robot_odom is not None
                else None
            )
            if dist is None or dist > self.TARGET_COMMIT_DIST_M:
                dist_txt = "unknown" if dist is None else f"{dist:.2f}m"
                mode = (
                    "observe target"
                    if self._current_target.get("observe_mode")
                    else "confirmed target"
                )
                print(
                    f"[Target] {reason} for {mode} '{label}' but target is "
                    f"{dist_txt} away (> {self.TARGET_COMMIT_DIST_M:.2f}m) — "
                    f"reissuing target navigation"
                )
                self._current_goal = None
                return
            print(f"[Target] {reason} for confirmed '{label}' — committing target")
            self._commit(label, target_xy[0], target_xy[1], dist)
            return

        raise RuntimeError(f"[Target] unexpected target nav failure reason: {reason}")

    def reset(self):
        self._target_names = []
        self._current_target = None
        self._publish_target_node_id(None)
        self._current_goal = None
        self._complete = False
        self._reset_target_progress()

    def get_state(self) -> Dict:
        return {
            "mode": "find",
            "target": self._get_label(),
            "complete": self._complete,
        }

    def _make_goal(
        self, target: Dict, obstacle_map, robot_xy: np.ndarray = None
    ) -> Optional[Tuple]:
        label = target.get("label", "?")
        goal_xy = self._target_surface_xy(target)
        goal_px = obstacle_map.xy_to_px(np.array([goal_xy]))[0].astype(int)
        goal_px = (int(goal_px[0]), int(goal_px[1]))
        center_xy = (float(target["x"]), float(target["y"]))
        if target.get("temporary_target"):
            print(
                f"[Target] temporary '{label}' approach goal -> "
                f"({goal_xy[0]:.1f}, {goal_xy[1]:.1f})"
            )
        else:
            print(
                f"[Target] '{label}' surface -> "
                f"({goal_xy[0]:.1f}, {goal_xy[1]:.1f}); "
                f"center=({center_xy[0]:.1f}, {center_xy[1]:.1f})"
            )
        return (goal_xy, goal_px, {"action": "find", "target": target})

    def _target_surface_xy(self, target: Dict) -> Tuple[float, float]:
        sx = target.get("surface_x", target.get("x"))
        sy = target.get("surface_y", target.get("y"))
        return (float(sx), float(sy))

    def _surface_xy_from_node(self, node, robot_xy: np.ndarray = None) -> Tuple[float, float]:
        pc_xy = self._surface_xy_from_point_cloud(getattr(node, "point_cloud", None), robot_xy)
        if pc_xy is not None:
            return (float(pc_xy[0]), float(pc_xy[1]))
        centroid = getattr(node, "centroid", None)
        if centroid is None:
            raise ValueError("[Target] target node has no point cloud or centroid")
        return (float(centroid[0]), float(centroid[1]))

    def _surface_xy_from_point_cloud(self, point_cloud, robot_xy: np.ndarray = None):
        if point_cloud is None:
            return None
        pts = np.asarray(point_cloud, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
            return None
        xy = pts[:, :2]
        valid = np.isfinite(xy).all(axis=1)
        xy = xy[valid]
        if len(xy) == 0:
            return None
        if robot_xy is None:
            return np.median(xy, axis=0)
        rxy = np.asarray(robot_xy[:2], dtype=float)
        dists = np.linalg.norm(xy - rxy[None, :], axis=1)
        valid_d = np.isfinite(dists)
        xy = xy[valid_d]
        dists = dists[valid_d]
        if len(xy) == 0:
            return None
        nearest_count = min(len(xy), max(5, int(np.ceil(0.05 * len(xy)))))
        nearest_idx = np.argpartition(dists, nearest_count - 1)[:nearest_count]
        return np.median(xy[nearest_idx], axis=0)

    def _get_label(self) -> str:
        if self._current_target:
            return self._current_target.get("label", self._current_target.get("canonical", "?"))
        return "?"
