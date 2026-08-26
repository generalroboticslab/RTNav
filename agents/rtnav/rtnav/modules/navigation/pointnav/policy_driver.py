"""PointNav policy driver thread.

Polls ``shared_state.nav.goal_xy`` and drives the robot toward it with the VLFM
PointNav policy: each tick the policy maps (depth, goal range+bearing) to a
discrete action (STOP/FORWARD/LEFT/RIGHT), handed to ``robot_api.execute_action``
which blocks until the 0.25 m / 30 deg primitive completes.

``shared_state.nav`` contract:
  - ``goal_xy`` : (wx, wy) world XY, or None when idle
  - ``status``  : "navigating" means drive; anything else means stop
  - ``nav_id``  : bumped on every new goal so a repeated XY still reads as fresh

On arrival (rho < SUCCESS_DIST) it sets ``status="arrived"`` and clears the goal;
on policy STOP or a watchdog give-up it sets ``status="failed"``.

Checkpoint loading lives in ``pointnav_loader``.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Tuple

import numpy as np
import torch

from rtnav.modules.navigation.pointnav.pointnav_loader import (
    SUCCESS_DIST,
    load_policy,
    prepare_depth,
    resolve_ckpt,
)


class PointNavDriverThread(threading.Thread):
    """See module docstring."""

    # RobotAPI action ints — match utils/robot_api.STOP/FORWARD/LEFT/RIGHT.
    _ACT_STOP, _ACT_FORWARD, _ACT_LEFT, _ACT_RIGHT = 0, 1, 2, 3

    _CPU_THREADS = 4  # Torch intra-op threads for CPU inference

    # Oscillation watchdog: give up if the policy thrashes L↔R.
    _OSC_WINDOW = 8
    _OSC_FLIP_THRESHOLD = 6  # ≥6 of last _OSC_WINDOW actions are L↔R flips
    _OSC_GIVEUP_SECS = 5.0

    # No-progress watchdog for navigable-but-awkward goals.
    _NO_PROGRESS_DELTA_M = 0.15
    _NO_PROGRESS_TIMEOUT_S = 20.0
    # Find owns target completion. A near stalled target returns quickly so Find
    # can apply its 0.25 m surface rule; a far target keeps trying.
    _TARGET_NO_PROGRESS_STOP_RHO = 0.3
    _TARGET_NO_PROGRESS_TIMEOUT_S = 2.0

    # Bootstrap in-place spin at episode start for broad initial coverage.
    _BOOTSTRAP_SPIN_MAX_ACTIONS = 11
    _BOOTSTRAP_SPIN_MAX_WALL_S = 30.0

    def __init__(
        self,
        shared_state,
        shutdown_event: threading.Event,
        *,
        robot_api,  # utils.robot_api.RobotAPI
        ckpt_path: Optional[str] = None,  # None → resolve_ckpt() searches
        device: Optional[torch.device] = None,  # None → CPU
        verbose: bool = False,
    ):
        super().__init__(name="PointNavDriverThread", daemon=True)
        self._shared_state = shared_state
        self._shutdown = shutdown_event
        self._verbose = verbose
        self._device = device if device is not None else torch.device("cpu")
        # Cap Torch intra-op threads + MKL/OMP for CPU inference.
        if self._device.type == "cpu":
            n_threads = self._CPU_THREADS
            try:
                torch.set_num_threads(n_threads)
            except Exception:
                pass
            os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
        if robot_api is None:
            raise RuntimeError(
                "PointNavDriverThread requires a RobotAPI instance — "
                "construct one with utils.robot_api.RobotAPI(node) and pass "
                "via robot_api=…"
            )
        self._robot_api = robot_api

        resolved_ckpt = resolve_ckpt(ckpt_path)
        print(f"[pointnav] loading checkpoint {resolved_ckpt} → {self._device}")
        self._policy = load_policy(resolved_ckpt, self._device)
        try:
            self._policy.policy.eval()
        except AttributeError:
            pass

        discrete = not hasattr(self._policy.policy.action_distribution, "mu_maybe_std")
        if not discrete:
            raise RuntimeError(
                "PointNavDriverThread requires a discrete-action policy, "
                "but the loaded policy is continuous."
            )

        # Per-active-goal state.
        self._active_goal: Optional[Tuple[float, float]] = None
        self._active_goal_id: Optional[int] = None
        self._reset_mask()

        # Oscillation watchdog: give up on a goal if the policy thrashes L↔R.
        self._action_history: list = []
        self._osc_start_t: Optional[float] = None

        # No-progress watchdog for goals that are navigable but physically awkward.
        self._best_rho = float("inf")
        self._best_rho_t = 0.0
        # Bootstrap mapping: in-place spin at episode start for broad initial coverage.
        self._bootstrap_spin_done = False
        with self._shared_state.lock:
            self._bootstrap_episode_index = self._shared_state.system.episode_index
            # Tell DecisionThread to defer reasoning until the spin finishes.
            self._shared_state.system.bootstrap_spin_complete = False

    # ── helpers ──────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        """No-op: this thread resets itself in-loop via episode_index."""

    def _reset_mask(self) -> None:
        self._mask = torch.zeros(1, 1, dtype=torch.bool, device=self._device)

    def _clear_active_goal(self) -> None:
        self._active_goal = None
        self._active_goal_id = None

    def _read_goal(self) -> Tuple[Optional[Tuple[float, float]], Optional[int], str]:
        """Snapshot the goal + a monotonically-increasing id so we can
        detect *new* requests (same XY twice in a row still counts as
        new because ``nav_id`` bumps on every producer-side update)."""
        with self._shared_state.lock:
            nav = self._shared_state.nav
            goal_xy = nav.goal_xy
            nav_id = int(nav.nav_id)
            status = str(nav.status)
        if goal_xy is None:
            return None, nav_id, status
        return (float(goal_xy[0]), float(goal_xy[1])), nav_id, status

    def _read_sensors(self) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float]]]:
        with self._shared_state.lock:
            habitat_obs = self._shared_state.sensor.habitat_obs
            odom = self._shared_state.sensor.latest_odom
        depth = None
        if habitat_obs is not None:
            depth = habitat_obs.depth
        return depth, odom

    def _begin_new_goal(self, goal_xy: Tuple[float, float], nav_id: int) -> None:
        self._active_goal = goal_xy
        self._active_goal_id = nav_id
        self._policy.reset()
        self._reset_mask()
        self._best_rho = float("inf")
        self._best_rho_t = time.time()
        if self._verbose:
            print(f"[pointnav] new goal ({goal_xy[0]:+.2f}, {goal_xy[1]:+.2f})")

    def _finish_goal(self, reason: str) -> None:
        ready_event = self._shared_state.task_ready
        if ready_event is not None and not ready_event.is_set():
            self._clear_active_goal()
            return
        # Zero velocity directly; NOT execute_action(STOP), which would
        # publish AgentDoneMsg and signal episode completion.
        self._robot_api._publish_vel(0.0, 0.0, 0.0)
        with self._shared_state.lock:
            nav = self._shared_state.nav
            # Only stamp if the goal we're finishing is still active.
            if int(nav.nav_id) == self._active_goal_id:
                nav.status = "arrived" if reason == "success" else "failed"
                nav.goal_xy = None
                nav.failure_reason = None if reason == "success" else reason
        if self._verbose or reason != "success":
            print(f"[pointnav] {reason} (goal_id={self._active_goal_id})")
        self._clear_active_goal()
        self._action_history.clear()
        self._osc_start_t = None

    def _decide_action(self, depth: np.ndarray, rho: float, theta: float) -> int:
        """Run one policy forward pass; return the discrete action int
        (0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT)."""
        depth_224 = prepare_depth(depth)
        pn_obs = {
            "depth": depth_224[None, ..., None].astype(np.float32),
            "pointgoal_with_gps_compass": np.array([[rho, theta]], dtype=np.float32),
        }
        with torch.inference_mode():
            action = self._policy.act(pn_obs, self._mask, deterministic=True)
        self._mask[:] = True
        a = int(action[0, 0].item())
        self._action_history.append(a)
        if len(self._action_history) > self._OSC_WINDOW:
            self._action_history.pop(0)
        self._update_oscillation_state()
        return a

    def _update_oscillation_state(self) -> None:
        """Flag sustained L↔R thrashing for the main loop's give-up watchdog."""
        if len(self._action_history) < self._OSC_WINDOW:
            self._osc_start_t = None
            return
        flips = sum(
            1
            for prev, curr in zip(self._action_history[:-1], self._action_history[1:])
            if {prev, curr} == {2, 3}
        )
        if flips < self._OSC_FLIP_THRESHOLD:
            self._osc_start_t = None
        elif self._osc_start_t is None:
            self._osc_start_t = time.time()

    # ── bootstrap spin ───────────────────────────────────────────────

    def _refresh_bootstrap_on_episode_reset(self) -> None:
        with self._shared_state.lock:
            counter = self._shared_state.system.episode_index
        if counter != self._bootstrap_episode_index:
            self._bootstrap_episode_index = counter
            self._bootstrap_spin_done = False
            with self._shared_state.lock:
                self._shared_state.system.bootstrap_spin_complete = False
            # Reset the policy RNN state so stale context from the previous
            # task doesn't drive the new episode's first actions.
            try:
                self._policy.reset()
            except Exception as exc:
                print(f"[pointnav] policy.reset() on episode boundary failed: {exc}")
            self._reset_mask()
            self._clear_active_goal()
            self._best_rho = float("inf")
            self._best_rho_t = time.time()
            print("[pointnav] bootstrap + policy state reset for new episode")

    def _refresh_bootstrap_restart_request(self) -> None:
        """Decision cleared bootstrap_spin_complete mid-episode (frontier-
        exhaustion restart) — re-run the bootstrap spin with fresh policy state."""
        with self._shared_state.lock:
            requested = not self._shared_state.system.bootstrap_spin_complete
        if not requested or not self._bootstrap_spin_done:
            return
        try:
            self._policy.reset()
        except Exception as exc:
            print(f"[pointnav] policy.reset() before bootstrap restart failed: {exc}")
        self._reset_mask()
        self._clear_active_goal()
        self._best_rho = float("inf")
        self._best_rho_t = time.time()
        self._bootstrap_spin_done = False
        print("[pointnav] bootstrap spin restart requested")

    # ── main loop ────────────────────────────────────────────────────

    def _target_goal_pending(self) -> bool:
        with self._shared_state.lock:
            nav = self._shared_state.nav
            return bool(
                nav.goal_source == "target"
                and nav.goal_xy is not None
                and nav.status == "navigating"
            )

    def _active_goal_is_target(self) -> bool:
        with self._shared_state.lock:
            nav = self._shared_state.nav
            return bool(nav.goal_source == "target" and int(nav.nav_id) == self._active_goal_id)

    def _no_progress_timeout_s(self, rho: float) -> float:
        if self._active_goal_is_target() and (
            rho < self._TARGET_NO_PROGRESS_STOP_RHO
            or self._best_rho < self._TARGET_NO_PROGRESS_STOP_RHO
        ):
            return self._TARGET_NO_PROGRESS_TIMEOUT_S
        return self._NO_PROGRESS_TIMEOUT_S

    def _target_no_progress_should_stop(self, rho: float) -> bool:
        """Target goals give up only near the surface; far targets keep trying."""
        return (
            (not self._active_goal_is_target())
            or rho < self._TARGET_NO_PROGRESS_STOP_RHO
            or self._best_rho < self._TARGET_NO_PROGRESS_STOP_RHO
        )

    def _bootstrap_spin(self) -> None:
        """One-shot startup scan: up to ``_BOOTSTRAP_SPIN_MAX_ACTIONS`` LEFT
        turns for a broad initial panorama. Runs to completion before any
        reasoning starts — Decision blocks on
        ``shared_state.system.bootstrap_spin_complete`` so the scene graph +
        obstacle map cover the surroundings before the first decision.
        """
        if self._bootstrap_spin_done:
            return
        start_t = time.time()
        for _ in range(self._BOOTSTRAP_SPIN_MAX_ACTIONS):
            if self._shutdown.is_set() or self._target_goal_pending():
                break
            if self._robot_api.episode_done:
                break
            if time.time() - start_t > self._BOOTSTRAP_SPIN_MAX_WALL_S:
                break
            self._robot_api.execute_action(self._ACT_LEFT)
        # Always mark complete so Decision can proceed — a partial panorama
        # beats deadlocking on a spin that won't finish.
        self._bootstrap_spin_done = True
        with self._shared_state.lock:
            # Never let Decision consume a frontier snapshot created before
            # the spin finished. FrontierDetectionThread immediately forces a
            # replacement from the caught-up VLFM map.
            self._shared_state.frontier.frontier_output = None
            self._shared_state.system.bootstrap_spin_complete = True

    def _run_loop(self) -> None:
        """Main loop: each policy decision drives a blocking
        robot_api.execute_action (0.25 m forward / 30° turn)."""
        from rtnav.utils.task_gate import wait_for_task_ready

        if not wait_for_task_ready(self._shared_state, "pointnav", self._shutdown):
            return
        last_log_t = 0.0

        while not self._shutdown.is_set():
            # Per-iteration task_ready gate: don't act during episode reset.
            if not wait_for_task_ready(self._shared_state, "pointnav", self._shutdown):
                break
            self._refresh_bootstrap_on_episode_reset()
            self._refresh_bootstrap_restart_request()

            # Bootstrap scan first for broad initial coverage.
            if not self._bootstrap_spin_done:
                self._bootstrap_spin()
                continue

            goal, nav_id, status = self._read_goal()
            if goal is None or status != "navigating":
                if self._active_goal is not None:
                    self._clear_active_goal()
                time.sleep(0.1)
                continue

            if self._active_goal is None or nav_id != self._active_goal_id:
                self._begin_new_goal(goal, nav_id)

            depth, odom = self._read_sensors()
            if depth is None or odom is None:
                time.sleep(0.05)
                continue

            cur_xy = (float(odom[0]), float(odom[1]))
            cur_yaw = float(odom[2])

            dx = goal[0] - cur_xy[0]
            dy = goal[1] - cur_xy[1]
            rho = float(np.hypot(dx, dy))

            if rho < SUCCESS_DIST and not self._active_goal_is_target():
                self._finish_goal("success")
                continue

            # No-progress watchdog.
            now_t = time.time()
            if rho < self._best_rho - self._NO_PROGRESS_DELTA_M:
                self._best_rho = rho
                self._best_rho_t = now_t
            elif now_t - self._best_rho_t > self._no_progress_timeout_s(rho):
                if not self._target_no_progress_should_stop(rho):
                    print(
                        f"[pointnav] target no progress for "
                        f"{now_t - self._best_rho_t:.1f}s "
                        f"(best ρ={self._best_rho:.2f}m, current ρ={rho:.2f}m) "
                        f"but ρ>={self._TARGET_NO_PROGRESS_STOP_RHO:.2f}m — continuing"
                    )
                    self._best_rho = rho
                    self._best_rho_t = now_t
                else:
                    print(
                        f"[pointnav] no progress for {now_t - self._best_rho_t:.1f}s "
                        f"(best ρ={self._best_rho:.2f}m, current ρ={rho:.2f}m) — giving up"
                    )
                    self._finish_goal("no_progress")
                    continue

            # Sustained-oscillation watchdog.
            if self._osc_start_t is not None and now_t - self._osc_start_t > self._OSC_GIVEUP_SECS:
                if not self._target_no_progress_should_stop(rho):
                    print(
                        f"[pointnav] target oscillation for "
                        f"{now_t - self._osc_start_t:.1f}s (ρ={rho:.2f}m) "
                        f"but ρ>={self._TARGET_NO_PROGRESS_STOP_RHO:.2f}m — continuing"
                    )
                    self._osc_start_t = None
                else:
                    print(
                        f"[pointnav] sustained oscillation for "
                        f"{now_t - self._osc_start_t:.1f}s (ρ={rho:.2f}m) — "
                        f"giving up on goal"
                    )
                    self._finish_goal("no_progress")
                    continue

            bearing_world = float(np.arctan2(dy, dx))
            theta = float((bearing_world - cur_yaw + np.pi) % (2 * np.pi) - np.pi)

            action_int = self._decide_action(depth, rho, theta)

            if self._verbose:
                now = time.time()
                if now - last_log_t > 0.5:
                    last_log_t = now
                    act_name = {0: "STOP", 1: "FWD", 2: "L", 3: "R"}.get(action_int, "?")
                    print(
                        f"[pointnav] rho={rho:5.2f}m  "
                        f"theta={np.rad2deg(theta):+6.1f}°  action={act_name}"
                    )

            if action_int == self._ACT_STOP:
                self._finish_goal("policy_stop")
                continue

            # Blocking — drives 0.25 m forward or 30° turn.
            self._robot_api.execute_action(action_int)

    def run(self):
        self._run_loop()
