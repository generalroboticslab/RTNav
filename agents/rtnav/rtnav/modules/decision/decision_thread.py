"""Turn target evidence and exploration frontiers into navigation goals."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rtnav.modules.decision.frontier_strategy import FrontierStrategy
from rtnav.modules.decision.target_strategy import TargetStrategy
from rtnav.modules.decision.target_utils import (
    label_matches_any,
    normalize_target_names,
)
from rtnav.modules.decision.target_verifier import TEMPORARY_TARGET_DISTANCE_M, TargetVerifier
from rtnav.utils.timing import timing


@dataclass
class TickSnapshot:
    """Decision-relevant state read under one lock."""

    task_complete: bool = False
    reset_counter: int = 0
    goal_category: str = ""

    nav_status: str = "idle"
    nav_failure_reason: Optional[str] = None
    nav_id: int = 0

    robot_odom: Optional[Tuple[float, float, float]] = None
    obstacle_map: Any = None
    frontier_output: Any = None
    robot_xy: Optional[np.ndarray] = None

    @property
    def has_robot_state(self) -> bool:
        return self.robot_odom is not None and self.obstacle_map is not None

    def context(self) -> Dict[str, Any]:
        yaw = float(self.robot_odom[2]) if self.robot_odom is not None else 0.0
        return {
            "robot_pose": (float(self.robot_xy[0]), float(self.robot_xy[1]), yaw),
            "obstacle_map": self.obstacle_map,
            "frontier_output": self.frontier_output,
        }


@dataclass
class NavUpdate:
    """Navigation writes buffered until the end of a decision."""

    goal_xy: Optional[Tuple[float, float]] = None
    goal_px: Optional[Tuple[int, int]] = None
    goal_cluster: Any = None
    goal_source: Optional[str] = None
    status: Optional[str] = None
    failure_reason: Optional[str] = None
    nav_id: Optional[int] = None
    current_target_xy: Optional[Tuple[float, float]] = None
    current_target_label: Optional[str] = None
    dirty: bool = False


class DecisionThread:
    """Pursue a verified target when one exists; otherwise explore."""

    FRONTIER_RESTART_DELAY_S = 5.0
    FRONTIER_RESTART_COOLDOWN_S = 10.0
    FRONTIER_GOAL_REFRESH_M = 0.25

    def __init__(self, shared_state, shutdown_event, cfg):
        self.shared_state = shared_state
        self.shutdown_event = shutdown_event

        self.target = TargetStrategy(shared_state, cfg.decision, cfg.camera)
        self.target_verifier = TargetVerifier(shared_state)
        self.frontier = FrontierStrategy(
            shared_state,
            frontier_cfg=cfg.frontier,
            decision_cfg=cfg.decision,
        )

        self._pending_goal: Optional[Dict[str, Any]] = None
        self._target_strategy_active = False
        self._target_vlm_running = False
        self._temporary_close_vlm_signatures: set = set()

        self._last_nav_id = 0
        self._processing_nav_result = False
        self._no_goal_start: Optional[float] = None
        self._last_frontier_restart = 0.0
        self._done_sent = False
        self._episode_key = None
        self._spin_wait_start: Optional[float] = None

        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        if self._running and self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Decision] Started (tid={self._thread.ident})")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def shutdown(self):
        self.stop()
        self.frontier.shutdown()

    def reset_episode(self) -> None:
        """Reset private state. SharedState owns the shared episode reset."""
        self._pending_goal = None
        self._target_strategy_active = False
        self._target_vlm_running = False
        self._temporary_close_vlm_signatures.clear()
        self._no_goal_start = None
        self._last_frontier_restart = 0.0
        self._done_sent = False
        self._episode_key = None
        self._spin_wait_start = None
        self.target.reset()
        self.target_verifier.reset()
        self.frontier.reset()
        print("[Decision] Episode reset")

    def _loop(self):
        from rtnav.utils.task_gate import wait_for_task_ready

        while self._running:
            if self.shutdown_event is not None and self.shutdown_event.is_set():
                break
            if not wait_for_task_ready(self.shared_state, "Decision", self.shutdown_event):
                break
            try:
                with timing("decision_tick"):
                    self._update()
            except Exception as exc:
                print(f"[Decision] Error: {exc}")
                import traceback

                traceback.print_exc()
            time.sleep(0.05)

    def _snapshot(self) -> TickSnapshot:
        with self.shared_state.lock:
            odom = self.shared_state.sensor.latest_odom
            snap = TickSnapshot(
                task_complete=bool(self.shared_state.task.complete),
                reset_counter=int(self.shared_state.system.episode_index),
                goal_category=str(self.shared_state.task.goal_category or "").lower().strip(),
                nav_status=str(self.shared_state.nav.status),
                nav_failure_reason=self.shared_state.nav.failure_reason,
                nav_id=int(self.shared_state.nav.nav_id),
                robot_odom=odom,
                obstacle_map=self.shared_state.mapping.obstacle_map,
                frontier_output=self.shared_state.frontier.frontier_output,
            )
        if odom is not None:
            snap.robot_xy = np.asarray(odom[:2], dtype=float)
        return snap

    def _is_stale(self, snap: TickSnapshot, action: str) -> bool:
        with self.shared_state.lock:
            reset_counter = int(self.shared_state.system.episode_index)
        task_ready = self.shared_state.task_ready.is_set()
        if task_ready and reset_counter == snap.reset_counter:
            return False
        why = "task changed" if reset_counter != snap.reset_counter else "task not ready"
        print(f"[Decision] Dropping stale {action}: {why}")
        return True

    def _flush_nav(self, update: NavUpdate):
        if not update.dirty:
            return
        with self.shared_state.lock:
            nav = self.shared_state.nav
            nav.goal_xy = update.goal_xy
            nav.goal_px = update.goal_px
            nav.goal_cluster = update.goal_cluster
            nav.goal_source = update.goal_source
            nav.failure_reason = update.failure_reason
            if update.status is not None:
                nav.status = update.status
            if update.nav_id is not None:
                nav.nav_id = update.nav_id

            target = self.shared_state.target
            target.current_target_xy = update.current_target_xy
            target.current_target_label = update.current_target_label

    def _update(self):
        snap = self._snapshot()
        nav = NavUpdate()

        self.frontier.sync_frontier_images(snap.frontier_output, snap.obstacle_map)
        self._reset_for_new_episode_if_needed(snap)

        # Verification pauses only the detector. PointNav continues executing its
        # current command while the VLM inspects candidates.
        if not snap.task_complete and not self._target_strategy_active:
            self._maybe_verify_target_detections(snap, snap.goal_category)
            if self._is_stale(snap, "target verification result"):
                return
            snap = self._snapshot()

        has_target = self.target.current_target is not None or self._has_pending_target(
            snap.goal_category
        )

        if not self._bootstrap_ready(has_target):
            return

        if snap.task_complete:
            self._finish_episode(snap, nav)
            return
        if self._done_sent:
            return

        if self._pending_goal and self._nav_finished(snap):
            self._handle_nav_result(snap, nav)
            self._flush_nav(nav)
            return

        if not snap.has_robot_state:
            return

        has_target = self.target.current_target is not None or self._has_pending_target(
            snap.goal_category
        )

        if self._pending_goal and snap.nav_status == "navigating":
            source = self._pending_goal["source"]
            if source == "target":
                if self._drive_active_target(snap, snap.goal_category, nav):
                    self._flush_nav(nav)
                return
            if source == "frontier":
                if has_target:
                    print("[Decision] Interrupting frontier for verified target")
                else:
                    if self._refresh_active_frontier(snap, nav):
                        self._flush_nav(nav)
                    return

        if has_target:
            goal = self.target.select_goal(
                snap.goal_category,
                snap.robot_xy,
                snap.obstacle_map,
            )
            if goal and goal[0] is not None:
                self._target_strategy_active = True
                self._send_goal(goal, "target", nav)
                self._flush_nav(nav)
                return

        self._target_strategy_active = False
        self._select_frontier(snap, nav)

    def _reset_for_new_episode_if_needed(self, snap: TickSnapshot):
        episode_key = snap.reset_counter, snap.goal_category
        if episode_key == self._episode_key:
            return
        self._episode_key = episode_key
        self._done_sent = False
        self._spin_wait_start = None
        self._pending_goal = None
        self._target_strategy_active = False
        self._temporary_close_vlm_signatures.clear()
        self.target.reset()
        self.target_verifier.reset()
        self.frontier.reset()
        with self.shared_state.lock:
            self.shared_state.target.target_goals = []
            self.shared_state.target.target_node_blacklist_ids.clear()
            self.shared_state.scenegraph.pending_verified_detections = []

    def _bootstrap_ready(self, has_target: bool) -> bool:
        with self.shared_state.lock:
            complete = bool(self.shared_state.system.bootstrap_spin_complete)
        if complete:
            if self._spin_wait_start is not None:
                print(
                    f"[Decision] Bootstrap spin complete after "
                    f"{time.time() - self._spin_wait_start:.1f}s"
                )
                self._spin_wait_start = None
            return True
        if has_target:
            print("[Decision] Verified target bypasses bootstrap-spin gate")
            return True
        if self._spin_wait_start is None:
            self._spin_wait_start = time.time()
            print("[Decision] Waiting for bootstrap spin")
        return False

    def _finish_episode(self, snap: TickSnapshot, nav: NavUpdate):
        if self._done_sent:
            return
        if self._is_stale(snap, "episode completion"):
            return
        self._done_sent = True
        self._clear_nav(nav)
        self._flush_nav(nav)
        print("[Decision] Target complete -> sending /agent_done")
        robot_api = self.shared_state.system.robot_api
        if robot_api is None or not hasattr(robot_api, "send_done"):
            print("[Decision] No RobotAPI available for /agent_done")
            return
        try:
            robot_api.send_done()
        except Exception as exc:
            print(f"[Decision] robot_api.send_done failed: {exc}")

    def _nav_finished(self, snap: TickSnapshot) -> bool:
        if self._pending_goal is None:
            return False
        if snap.nav_id != self._pending_goal["nav_id"]:
            return False
        return snap.nav_status != "navigating"

    def _drive_active_target(self, snap: TickSnapshot, target_name: str, nav: NavUpdate) -> bool:
        if self._maybe_verify_close_target(snap, target_name):
            if self._is_stale(snap, "close-target result"):
                return False
            if self.target.current_target is not None:
                goal = self.target.select_goal(
                    target_name,
                    snap.robot_xy,
                    snap.obstacle_map,
                )
                if goal and goal[0] is not None:
                    self._send_goal(goal, "target", nav)
                    return True

        if self.target.abort_if_reclassified():
            self._target_strategy_active = False
            self._clear_nav(nav)
            return True

        if self.target.finish_temporary_if_close(snap.robot_xy):
            self._target_strategy_active = False
            self._clear_nav(nav)
            return True

        if self._is_stale(snap, "target commit"):
            self._clear_nav(nav)
            return True

        if self.target.commit_if_close(snap.robot_xy):
            self._target_strategy_active = False
            self._clear_nav(nav)
            return True

        goal = self.target.select_goal(
            target_name,
            snap.robot_xy,
            snap.obstacle_map,
        )
        if not goal or goal[0] is None:
            return False
        old_xy = self._pending_goal["xy"]
        new_xy = goal[0]
        if np.linalg.norm(np.asarray(new_xy) - np.asarray(old_xy)) <= 1e-6:
            return False
        self._send_goal(goal, "target", nav)
        return True

    def _handle_nav_result(self, snap: TickSnapshot, nav: NavUpdate):
        if self._processing_nav_result or self._pending_goal is None:
            return
        self._processing_nav_result = True
        try:
            if self._is_stale(snap, "navigation result"):
                self._pending_goal = None
                return
            goal = self._pending_goal
            self._pending_goal = None
            source = goal["source"]
            xy = goal["xy"]
            ok = snap.nav_failure_reason is None
            outcome = "Reached" if ok else f"Failed ({snap.nav_failure_reason})"
            print(f"[Decision] {outcome}: ({xy[0]:.1f}, {xy[1]:.1f}) [{source}]")

            if source == "frontier":
                if ok:
                    self.frontier.on_goal_reached(xy)
                else:
                    self.frontier.on_goal_failed(xy, snap.nav_failure_reason or "")
            else:
                target_info = (goal.get("info") or {}).get("target", {})
                if target_info.get("temporary_target"):
                    self.target.finish_temporary_target(
                        reached=ok, reason=snap.nav_failure_reason or ""
                    )
                    self._target_strategy_active = False
                elif ok:
                    committed = self.target.on_reached()
                    if committed:
                        self._target_strategy_active = False
                else:
                    self.target.on_failed(snap.nav_failure_reason or "")

            self._clear_nav(nav)
        finally:
            self._processing_nav_result = False

    def _has_pending_target(self, target_name: str) -> bool:
        names = normalize_target_names(target_name)
        with self.shared_state.lock:
            targets = list(self.shared_state.target.target_goals)
            blacklisted_nodes = set(self.shared_state.target.target_node_blacklist_ids)
        for target in targets:
            if not target.get("vlm_confirmed"):
                continue
            node_id = target.get("node_id")
            if node_id is not None and int(node_id) in blacklisted_nodes:
                continue
            label = str(target.get("label", target.get("canonical", "")))
            canonical = str(target.get("canonical", label))
            if not (label_matches_any(label, names) or label_matches_any(canonical, names)):
                continue
            return True
        return False

    def _maybe_verify_target_detections(self, snap: TickSnapshot, target_name: str) -> bool:
        if self._target_vlm_running or self._has_pending_target(target_name):
            return False
        candidates = self.target_verifier.collect_candidates(target_name)
        if not candidates:
            return False

        target_name = self.target_verifier.target_name(target_name)
        print(
            f"[TargetVLM] Verifying {len(candidates)} candidate(s) for "
            f"'{target_name}' while navigation continues"
        )
        gate = self.shared_state.inference
        self._target_vlm_running = True
        gate.pause_for_vllm()
        try:
            results = self.target_verifier.verify_batch(candidates, target_name)
            if self._is_stale(snap, "target VLM batch"):
                return True
            accepted = self.target_verifier.apply_results(results)
        finally:
            gate.resume_after_vllm()
            self._target_vlm_running = False

        self._log_verification(results)
        if accepted:
            best = max(accepted, key=lambda x: float(x.get("vlm_yes_norm", 0.0)))
            print(
                f"[TargetVLM] Accepted {len(accepted)} target(s); best "
                f"yes={best.get('vlm_yes_norm', 0.0):.3f} at "
                f"({best['x']:.2f}, {best['y']:.2f})"
            )
        return True

    def _maybe_verify_close_target(self, snap: TickSnapshot, target_name: str) -> bool:
        target = self.target.current_target
        if self._target_vlm_running or target is None or not target.get("temporary_target"):
            return False

        candidates = self.target_verifier.collect_candidates(target_name, include_queried=True)
        close = [
            candidate
            for candidate in candidates
            if not candidate.temporary_target
            and candidate.has_geometry
            and candidate.signature not in self._temporary_close_vlm_signatures
            and np.isfinite(candidate.distance)
            and candidate.distance <= TEMPORARY_TARGET_DISTANCE_M
        ]
        if not close:
            return False
        close.sort(key=lambda candidate: candidate.distance)

        target_name = self.target_verifier.target_name(target_name)
        print(
            f"[TargetVLM] Verifying {len(close)} close candidate(s) during "
            "temporary-target navigation"
        )
        gate = self.shared_state.inference
        self._target_vlm_running = True
        gate.pause_for_vllm()
        try:
            results = self.target_verifier.verify_batch(close, target_name)
            if self._is_stale(snap, "close-target VLM batch"):
                return True
            accepted = self.target_verifier.apply_results(results)
        finally:
            gate.resume_after_vllm()
            self._target_vlm_running = False

        for result in results:
            self._temporary_close_vlm_signatures.add(result.candidate.signature)
        self._log_verification(results, prefix="temporary-nav ")
        # Another offset mask is still provisional; keep approaching until a centered
        # confirmation replaces it.
        if not any(not target.get("temporary_target") for target in accepted):
            return False

        self.target.finish_temporary_target(reached=False, reason="replaced_by_confirmed_target")
        self._target_strategy_active = True
        return True

    @staticmethod
    def _log_verification(results, prefix: str = ""):
        for result in results:
            candidate = result.candidate
            verdict = "yes" if result.accepted else "no"
            node = "none" if candidate.node_id is None else candidate.node_id
            xy = (
                f"({candidate.x:.2f},{candidate.y:.2f})"
                if candidate.x is not None and candidate.y is not None
                else "unprojected"
            )
            print(
                f"[TargetVLM] {prefix}{verdict} p_yes={result.yes_norm:.3f} "
                f"t={result.inference_time_s:.2f}s det='{candidate.label}' "
                f"node={node} xy={xy}"
            )

    def _refresh_active_frontier(self, snap: TickSnapshot, nav: NavUpdate) -> bool:
        old_xy = self._pending_goal["xy"]
        try:
            goal, reason = self.frontier.refresh_goal(snap.context())
        except Exception as exc:
            goal, reason = None, f"refresh failed: {exc}"
        if not goal or goal[0] is None:
            print(f"[Decision] Active frontier vanished ({reason}); cancelling it")
            self.frontier.on_goal_failed(old_xy, reason="frontier_vanished")
            self._clear_nav(nav)
            return True

        new_xy = goal[0]
        drift = float(np.linalg.norm(np.asarray(new_xy) - np.asarray(old_xy)))
        if drift < self.FRONTIER_GOAL_REFRESH_M:
            return False
        print(
            f"[Decision] Updating drifting frontier ({old_xy[0]:.1f}, "
            f"{old_xy[1]:.1f}) -> ({new_xy[0]:.1f}, {new_xy[1]:.1f}), "
            f"drift={drift:.2f}m"
        )
        self._send_goal(self._frontier_goal(goal), "frontier", nav)
        return True

    @staticmethod
    def _frontier_goal(goal):
        info = goal[2] if len(goal) > 2 else None
        return goal[0], goal[1], info if isinstance(info, dict) else {"cluster": info}

    def _select_frontier(self, snap: TickSnapshot, nav: NavUpdate):
        goal = self.frontier.select_goal(snap.context())
        if goal and goal[0] is not None:
            self._send_goal(self._frontier_goal(goal), "frontier", nav)
            self._flush_nav(nav)
            return
        reason = goal[1] if isinstance(goal, tuple) and len(goal) > 1 else ""
        self._handle_no_frontier(str(reason or ""))

    def _handle_no_frontier(self, reason: str):
        now = time.time()
        if self._no_goal_start is None:
            self._no_goal_start = now
            print(f"[Decision] No frontier goal ({reason or 'waiting for map'})")
        if now - self._no_goal_start < self.FRONTIER_RESTART_DELAY_S:
            return
        if now - self._last_frontier_restart < self.FRONTIER_RESTART_COOLDOWN_S:
            return

        self._last_frontier_restart = now
        self._no_goal_start = None
        self._spin_wait_start = None
        self._pending_goal = None
        self.frontier.reset()
        with self.shared_state.lock:
            self.shared_state.frontier.reset()
            self.shared_state.system.exploration_wipe_counter += 1
            self.shared_state.system.bootstrap_spin_complete = False
        print("[Decision] Frontiers exhausted; clearing exploration and re-spinning")

    def _send_goal(self, goal: Tuple, source: str, nav: NavUpdate):
        goal_xy, goal_px, info = goal
        info = info if isinstance(info, dict) else {}
        self._last_nav_id += 1
        self._pending_goal = {
            "xy": tuple(float(v) for v in goal_xy[:2]),
            "px": goal_px,
            "source": source,
            "info": info,
            "nav_id": self._last_nav_id,
        }

        nav.goal_xy = self._pending_goal["xy"]
        nav.goal_px = goal_px
        nav.goal_cluster = info.get("cluster") if source == "frontier" else None
        nav.goal_source = source
        nav.status = "navigating"
        nav.failure_reason = None
        nav.nav_id = self._last_nav_id
        if source == "target" and "target" in info:
            target = info["target"]
            nav.current_target_xy = (
                float(target.get("x", goal_xy[0])),
                float(target.get("y", goal_xy[1])),
            )
            nav.current_target_label = str(target.get("label", "?"))
        nav.dirty = True
        self._no_goal_start = None
        print(f"[Decision] Goal: ({goal_xy[0]:.1f}, {goal_xy[1]:.1f}) [{source}]")
        self._record_decision(source, info, goal_xy)

    def _record_decision(self, source: str, info: Dict, goal_xy):
        recorder = self.shared_state.system.decision_recorder
        if recorder is None:
            return
        try:
            if source == "target":
                label = info.get("target", {}).get("label", "?")
                recorder.record(source, goal_xy, f"verified target '{label}'")
                return
            sheet = self.frontier.consume_choice_sheet()
            caption = "image-VLM frontier selection"
            if sheet is not None:
                recorder.record_image(source, sheet, caption, goal_xy=goal_xy)
            else:
                recorder.record(source, goal_xy, caption)
        except Exception:
            pass

    def _clear_nav(self, nav: NavUpdate):
        self._pending_goal = None
        nav.goal_xy = None
        nav.goal_px = None
        nav.goal_cluster = None
        nav.goal_source = None
        nav.status = "idle"
        nav.failure_reason = None
        nav.current_target_xy = None
        nav.current_target_label = None
        nav.dirty = True

    def get_state(self) -> Dict[str, Any]:
        with self.shared_state.lock:
            return {
                "goal_category": self.shared_state.task.goal_category,
                "pending_goal": self._pending_goal,
                "nav_status": self.shared_state.nav.status,
                "target": self.target.get_state(),
                "frontier": self.frontier.get_state(),
                "task_complete": self.shared_state.task.complete,
            }
