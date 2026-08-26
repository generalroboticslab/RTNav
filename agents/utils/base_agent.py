"""Base ROS2 agent node with common eval loop for sync/async modes.

Subclass and implement:
  _load_policy()                         — load model weights
  _reset_policy_state()                  — reset hidden state for new episode
  _compute_action(obs_dict, step) -> (action_int, policy_info_dict)
"""

import json
import os
import random
import threading
import time
from typing import Optional, Tuple

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from ros_msgs.msg import Observation as ObservationMsg
from ros_msgs.srv import ResetEpisode, SyncStep

from ros_helpers import (
    DONE_SENTINEL,
    step_result_to_obs_dict,
    obs_msg_to_obs_dict,
    step_result_to_obs_msg,
)
from robot_api import RobotAPI
from video_recorder import AgentObservationVideoRecorder


class BaseAgentNode(Node):
    """Base class for ObjectNav agent ROS2 nodes."""

    def __init__(self, name: str, mode: str = "sync"):
        super().__init__(name)
        self._mode = mode
        self._cb_group = ReentrantCallbackGroup()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"{name} starting in {mode} mode on {self._device}")

        # Service clients
        self._cli_reset = self.create_client(ResetEpisode, "/reset_episode", callback_group=self._cb_group)
        self._cli_step = None
        if mode == "sync":
            self._cli_step = self.create_client(SyncStep, "/sync_step", callback_group=self._cb_group)

        # Async mode
        self._robot_api: Optional[RobotAPI] = None
        if mode == "async":
            self._obs_control_cb_group = MutuallyExclusiveCallbackGroup()
            self._robot_api = RobotAPI(self)
            self.create_subscription(
                ObservationMsg, "/obs", self._on_obs_control, 1,
                callback_group=self._obs_control_cb_group,
            )

        # Agent-side video: none/display/save. In async mode frames come from /obs.
        video_mode = os.environ.get("AGENT_VIDEO_MODE", "").strip().lower()
        if not video_mode:
            video_mode = "save" if os.environ.get("VIDEO_DIR", "") else "none"
        if video_mode not in ("none", "display", "save"):
            raise ValueError(f"AGENT_VIDEO_MODE must be none, display, or save; got {video_mode!r}")
        self._record_async_observations = mode == "async" and video_mode in ("display", "save")
        self._latest_policy_info = {}
        self._video_recorder = AgentObservationVideoRecorder(
            os.environ.get("VIDEO_DIR", ""), video_mode
        )
        if self._record_async_observations:
            self._obs_video_cb_group = MutuallyExclusiveCallbackGroup()
            video_qos_depth = max(
                1, int(os.environ.get("AGENT_VIDEO_QOS_DEPTH", "300"))
            )
            video_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=video_qos_depth,
            )
            self.create_subscription(
                ObservationMsg, "/obs", self._on_obs_video, video_qos,
                callback_group=self._obs_video_cb_group,
            )
        self._eval_done = False

    # ── Override these ───────────────────────────────────────────────────

    def _load_policy(self):
        raise NotImplementedError

    def _reset_policy_state(self):
        """Reset hidden state / caches for a new episode."""
        pass

    def _on_episode_start(self, reset_resp, initial_obs: dict):
        """Optional hook for subclasses that need episode metadata."""
        pass

    def _compute_action(self, obs_dict: dict, step: int) -> Tuple[int, dict]:
        """Compute action from observation. Returns (discrete_action_id, policy_info_dict_for_video)."""
        raise NotImplementedError

    # ── Eval loop ────────────────────────────────────────────────────────

    def run_eval(self, num_episodes: int = -1):
        self._load_policy()
        service_wait_timeout = float(os.environ.get("AGENT_SERVICE_WAIT_TIMEOUT", "300"))
        self.get_logger().info("Waiting for env services …")
        if not self._cli_reset.wait_for_service(timeout_sec=service_wait_timeout):
            raise RuntimeError(
                f"Timed out waiting {service_wait_timeout:.0f}s for /reset_episode. "
                "The paired env container may have exited before starting ROS services."
            )
        if self._cli_step and not self._cli_step.wait_for_service(timeout_sec=service_wait_timeout):
            raise RuntimeError(
                f"Timed out waiting {service_wait_timeout:.0f}s for /sync_step. "
                "The paired env container may have exited before starting ROS services."
            )

        ep_idx = 0
        remaining = True

        while remaining:
            if 0 < num_episodes <= ep_idx:
                break

            # Reset episode
            resp = self._call_reset()
            if resp is None:
                break
            remaining = resp.episodes_remaining
            ep_idx += 1
            self.get_logger().info(
                f"Episode {ep_idx}: scene={os.path.basename(resp.scene_id)} "
                f"ep_id={resp.episode_id}"
            )

            self._video_recorder.new_episode(resp.scene_id, resp.episode_id)
            self._latest_policy_info = {}
            obs = step_result_to_obs_dict(resp.observation)
            if self._record_async_observations:
                self._video_recorder.record_step(obs, self._latest_policy_info, 0)

            if self._robot_api:
                self._robot_api.reset_episode(resp.episode_id_hash)
                self._robot_api.update_observation(step_result_to_obs_msg(resp.observation, resp.episode_id_hash))

            self._on_episode_start(resp, obs)
            self._reset_policy_state()

            # Step loop
            done, step = False, 0
            while not done:
                action, pi = self._compute_action(obs, step)
                self._latest_policy_info = pi or {}
                if not self._record_async_observations:
                    self._video_recorder.record_step(obs, self._latest_policy_info, step)
                if self._mode == "sync":
                    obs, _, done, _ = self._sync_step(action, step)
                else:
                    obs, done = self._async_step(action)
                step += 1

            self.get_logger().info(f"Episode {ep_idx} done after {step} steps.")
            self._video_recorder.save_episode()

        self.get_logger().info("Evaluation complete.")
        self._eval_done = True

    # ── Service helpers ──────────────────────────────────────────────────

    def _call_reset(self):
        future = self._cli_reset.call_async(ResetEpisode.Request())
        return self._wait_future(future, 120.0)

    def _sync_step(self, action: int, step: int):
        req = SyncStep.Request()
        req.action = action
        req.agent_step_id = step
        resp = self._wait_future(self._cli_step.call_async(req), 60.0)
        if resp is None:
            msg = (
                "[BaseAgent] FATAL: sync step service timed out; "
                f"action={action} agent_step_id={step} timeout_s=60.0"
            )
            self.get_logger().error(msg)
            raise TimeoutError(msg)
        r = resp.result
        info = json.loads(r.info_json) if r.info_json else {}
        return step_result_to_obs_dict(r), r.reward, r.done, info

    def _async_step(self, action: int):
        obs_msg = self._robot_api.execute_action(action)
        if obs_msg is None or self._robot_api.episode_done:
            return (obs_msg_to_obs_dict(obs_msg) if obs_msg else {}), True
        return obs_msg_to_obs_dict(obs_msg), False

    def _wait_future(self, future, timeout: float):
        deadline = time.time() + timeout
        while not future.done():
            if time.time() > deadline:
                self.get_logger().error(f"Service timed out after {timeout}s")
                return None
            time.sleep(0.005)
        return future.result()

    def _on_obs_control(self, msg: ObservationMsg):
        if not self._robot_api:
            return
        if msg.step_id == DONE_SENTINEL:
            self._robot_api.set_episode_done(getattr(msg, "episode_id_hash", None))
            return
        self._robot_api.update_observation(msg)

    def _on_obs_video(self, msg: ObservationMsg):
        if msg.step_id == DONE_SENTINEL:
            return
        self._video_recorder.record_step(
            obs_msg_to_obs_dict(msg), self._latest_policy_info, msg.step_id
        )


# ── Standard entry point ─────────────────────────────────────────────────


def run_agent(node_class, env_prefix: str = ""):
    """Standard main() for agent nodes.

    Reads MODE and NUM_EPISODES from environment, optionally prefixed
    (e.g. VLFM_MODE, DAGRL_NUM_EPISODES) for backward compatibility.
    """
    seed = int(os.environ.get("HABITAT_SEED", "1"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    def _env(key, default):
        """Read env var with optional prefix fallback."""
        return os.environ.get(f"{env_prefix}{key}", os.environ.get(key, default))

    mode = _env("MODE", "sync")
    num_eps = int(_env("NUM_EPISODES", "-1"))

    rclpy.init()
    node = node_class(mode=mode)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run_eval(num_eps)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
