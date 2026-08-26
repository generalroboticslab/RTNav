"""Discrete action -> velocity command translator for async mode.

The agent computes a discrete action (FORWARD/LEFT/RIGHT/STOP/LOOK_UP/LOOK_DOWN).
RobotAPI converts it into velocity commands published over /cmd_vel, monitors
the /obs stream to detect completion, then returns control to the agent.
"""

import math
import threading
import time
from typing import Optional

import numpy as np
from rclpy.node import Node

from ros_msgs.msg import (
    Observation as ObservationMsg,
    VelocityCommand as VelocityCommandMsg,
    AgentDone as AgentDoneMsg,
)

STOP, FORWARD, LEFT, RIGHT, LOOK_UP, LOOK_DOWN = 0, 1, 2, 3, 4, 5

class RobotAPI:
    """Translates discrete actions to ROS2 velocity commands for async mode."""

    def __init__(
        self,
        node: Node,
        forward_step: float = 0.25,
        turn_angle_deg: float = 30.0,
        look_angle_deg: float = 30.0,
        linear_speed: float = 0.25, # m/s
        angular_speed: float = math.radians(30.0), # rad/s
        pitch_speed: float = math.radians(30.0), # rad/s
        cmd_rate_hz: float = 30.0,
    ):
        self._node = node
        self._forward_step = forward_step
        self._turn_angle_rad = math.radians(turn_angle_deg)
        self._look_angle_rad = math.radians(look_angle_deg)
        self._linear_speed = linear_speed
        self._angular_speed = angular_speed
        self._pitch_speed = pitch_speed
        self._cmd_period = 1.0 / cmd_rate_hz
        self._stall_timeout = 1.0
        self._dist_eps = 1e-4 # mark action as complete if it's close enough to target
        self._angle_eps = math.radians(0.05)

        self._pub_vel = node.create_publisher(VelocityCommandMsg, "/cmd_vel", 1)
        self._pub_done = node.create_publisher(AgentDoneMsg, "/agent_done", 10)

        self._latest_obs: Optional[ObservationMsg] = None
        self._obs_lock = threading.Lock()
        self._episode_hash: Optional[int] = None
        self._step_id = 0
        self._episode_done_event = threading.Event()

    # ── Episode lifecycle ────────────────────────────────────────────────

    def reset_episode(self, episode_hash: Optional[int] = None, preserve_step_id: bool = False):
        """preserve_step_id keeps _step_id monotonic across episodes so stale
        in-flight commands can't outrank new ones at the env's step-id gate."""
        self._episode_done_event.clear()
        with self._obs_lock:
            self._latest_obs = None
            self._episode_hash = episode_hash
        if not preserve_step_id:
            self._step_id = 0

    def set_episode_done(self, episode_hash: Optional[int] = None):
        if episode_hash is not None:
            with self._obs_lock:
                if self._episode_hash is not None and int(episode_hash) != self._episode_hash:
                    return
        self._episode_done_event.set()

    @property
    def episode_done(self) -> bool:
        return self._episode_done_event.is_set()

    def update_observation(self, obs: ObservationMsg):
        with self._obs_lock:
            ep_hash = int(getattr(obs, "episode_id_hash", 0))
            if self._episode_hash is None:
                self._episode_hash = ep_hash
            elif ep_hash != self._episode_hash:
                return
            if self._latest_obs is not None and obs.step_id < self._latest_obs.step_id:
                return
            self._latest_obs = obs

    def get_latest_observation(self) -> Optional[ObservationMsg]:
        with self._obs_lock:
            return self._latest_obs

    def _observation_matches_current_action(self, obs: Optional[ObservationMsg]) -> bool:
        return (
            obs is not None
            and int(getattr(obs, "applied_agent_step_id", 0)) == self._step_id
        )

    # ── Main entry point ─────────────────────────────────────────────────

    def execute_action(self, action: int) -> Optional[ObservationMsg]:
        """Execute one discrete action, blocking until complete.
        Returns latest observation, or None for STOP."""
        self._step_id += 1
        if action == STOP:
            self._send_stop()
            return None
        if action == FORWARD:
            return self._run_forward()
        if action in (LEFT, RIGHT):
            return self._run_turn(1 if action == LEFT else -1)
        if action in (LOOK_UP, LOOK_DOWN):
            return self._run_look(1 if action == LOOK_UP else -1)
        self._publish_vel(0, 0, 0)
        return self.get_latest_observation()

    # ── Execution helpers ────────────────────────────────────────────────

    def _run_forward(self) -> Optional[ObservationMsg]:
        """Drive forward until forward_step metres covered."""
        start = self._wait_for_obs()
        if start is None:
            return None
        start_xy = np.array(start.gps, dtype=np.float64)
        dist = 0.0
        prev_step, prev_dist = start.step_id, 0.0
        stall_since = None
        latest = None

        for _ in range(int(10.0 / self._cmd_period)):
            if self.episode_done:
                break
            self._publish_vel(self._linear_speed, 0, 0)
            time.sleep(self._cmd_period)
            cur = self.get_latest_observation()
            if cur is None:
                continue
            if cur.step_id == prev_step:
                continue
            prev_step = cur.step_id
            if not self._observation_matches_current_action(cur):
                continue
            latest = cur
            if stall_since is None:
                stall_since = time.monotonic()
            dist = np.linalg.norm(np.array(cur.gps, dtype=np.float64) - start_xy)
            if dist + self._dist_eps >= self._forward_step:
                break
            if abs(dist - prev_dist) < 0.001:
                if time.monotonic() - stall_since >= self._stall_timeout:
                    break  # collision stall
            else:
                stall_since = time.monotonic()
            prev_dist = dist

        return self._stop_and_wait(latest)

    def _run_turn(self, direction: int) -> Optional[ObservationMsg]:
        """Rotate until turn_angle radians covered. direction: +1=left, -1=right."""
        start = self._wait_for_obs()
        if start is None:
            return None
        prev_compass = float(start.compass)
        progress = 0.0
        prev_step, prev_progress = start.step_id, 0.0
        stall_since = None
        latest = None

        for _ in range(int(10.0 / self._cmd_period)):
            if self.episode_done:
                break
            self._publish_vel(0, self._angular_speed * direction, 0)
            time.sleep(self._cmd_period)
            cur = self.get_latest_observation()
            if cur is None:
                continue
            if cur.step_id == prev_step:
                continue
            prev_step = cur.step_id
            if not self._observation_matches_current_action(cur):
                continue

            latest = cur
            if stall_since is None:
                stall_since = time.monotonic()
            cc = float(cur.compass)
            progress += direction * self._angle_diff(cc, prev_compass)
            prev_compass = cc
            if progress + self._angle_eps >= self._turn_angle_rad:
                break
            if abs(progress - prev_progress) < 0.001:
                if time.monotonic() - stall_since >= self._stall_timeout:
                    break
            else:
                stall_since = time.monotonic()
            prev_progress = progress

        return self._stop_and_wait(latest)

    def _run_look(self, direction: int) -> Optional[ObservationMsg]:
        """Tilt camera until look_angle radians covered. direction: +1=up, -1=down."""
        start = self._wait_for_obs()
        if start is None:
            return None
        start_pitch = float(getattr(start, "camera_pitch", 0.0))
        prev_step, prev_progress = start.step_id, 0.0
        stall_since = None
        latest = None

        for _ in range(int(10.0 / self._cmd_period)):
            if self.episode_done:
                break
            self._publish_vel(0, 0, self._pitch_speed * direction)
            time.sleep(self._cmd_period)
            cur = self.get_latest_observation()
            if cur is None or cur.step_id == prev_step:
                continue
            prev_step = cur.step_id
            if not self._observation_matches_current_action(cur):
                continue
            latest = cur
            if stall_since is None:
                stall_since = time.monotonic()
            cur_pitch = float(getattr(cur, "camera_pitch", start_pitch))
            progress = direction * (cur_pitch - start_pitch)
            if progress + self._angle_eps >= self._look_angle_rad:
                break
            if abs(progress - prev_progress) < 0.001:
                if time.monotonic() - stall_since >= self._stall_timeout:
                    break
            else:
                stall_since = time.monotonic()
            prev_progress = progress

        return self._stop_and_wait(latest)

    # ── Low-level ────────────────────────────────────────────────────────

    def _send_stop(self):
        self._publish_vel(0, 0, 0)
        self.send_done()

    def send_done(self) -> None:
        """Publish /agent_done with the current step_id.

        Public entry point so other components (decision_thread, agent eval
        loop) can signal episode completion via the same publisher
        ``execute_action(STOP)`` uses — keeping a single source of /agent_done
        per process and matching the VLFM baseline's pattern.
        """
        msg = AgentDoneMsg()
        msg.stop = True
        msg.agent_step_id = self._step_id
        self._pub_done.publish(msg)

    def _stop_and_wait(
        self, latest: Optional[ObservationMsg]
    ) -> Optional[ObservationMsg]:
        previous_step = latest.step_id if latest is not None else -1
        settled = self._wait_for_fresh_obs(
            previous_step,
            timeout=1.0,
            keep_stopped=True,
            expected_agent_step_id=self._step_id,
        )
        return settled if settled is not None else latest

    def _publish_vel(self, linear: float, angular: float, pitch: float):
        msg = VelocityCommandMsg()
        msg.linear_velocity = float(linear)
        msg.angular_velocity = float(angular)
        msg.camera_pitch_velocity = float(pitch)
        msg.agent_step_id = self._step_id
        self._pub_vel.publish(msg)

    def _wait_for_obs(self, timeout: float = 5.0) -> Optional[ObservationMsg]:
        t0 = time.time()
        while time.time() - t0 < timeout:
            obs = self.get_latest_observation()
            if obs is not None:
                return obs
            time.sleep(0.01)
        self._node.get_logger().error("Timed out waiting for observation")
        return None

    def _wait_for_fresh_obs(
        self,
        prev_step: int,
        timeout: float = 1.0,
        keep_stopped: bool = False,
        expected_agent_step_id: Optional[int] = None,
    ) -> Optional[ObservationMsg]:
        t0 = time.time()
        sleep_period = self._cmd_period if keep_stopped else 0.005
        while time.time() - t0 < timeout:
            if self.episode_done:
                return None
            if keep_stopped:
                self._publish_vel(0, 0, 0)
            obs = self.get_latest_observation()
            expected_matches = (
                expected_agent_step_id is None
                or (
                    obs is not None
                    and int(getattr(obs, "applied_agent_step_id", 0))
                    == expected_agent_step_id
                )
            )
            if obs is not None and obs.step_id != prev_step and expected_matches:
                return obs
            time.sleep(sleep_period)
        return None

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d
