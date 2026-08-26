"""ROS2 node for ObjectNav evaluation.

Runs the Habitat simulator, manages episodes, and supports:
  - Sync mode:  /sync_step service (agent sends action, gets result)
  - Async mode: publishes /obs at sim_hz, subscribes to /cmd_vel
  - Episode slicing for parallel evaluation

Configure via environment variables:
  MODE: sync (default), async
  BENCHMARK: ovon (default), hm3d_v1, hm3d_v2
  HYDRA_CONFIG_PATH: absolute path to shared env config dir
    HYDRA_CONFIG_NAME: hydra config name override (otherwise BENCHMARK maps to hm3d or ovon)
  EPISODE_OFFSET / EPISODE_COUNT: for parallel episode slicing
  ENV_STATS_PATH: output path for per-episode metrics JSON
"""

import hashlib
import importlib
import json
import os
import random
import sys
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import habitat_sim

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String

import hydra
from omegaconf import DictConfig, OmegaConf
from habitat import make_dataset
from habitat.config import read_write


def patch_config(cfg: DictConfig) -> DictConfig:
    sim_config = cfg.habitat.simulator
    if len(sim_config.agents) == 1:
        with read_write(sim_config):
            sim_config.agents_order = list(sim_config.agents.keys())

    assert len(sim_config.agents_order) == len(sim_config.agents) and set(
        sim_config.agents_order
    ) == set(sim_config.agents.keys()), (
        "habitat.simulator.agents_order should be set explicitly "
        "and match the agents' keys in habitat.simulator.agents.\n"
        f"habitat.simulator.agents_order: {sim_config.agents_order}\n"
        f"habitat.simulator.agents: {list(sim_config.agents.keys())}"
    )

    OmegaConf.set_readonly(cfg, True)
    return cfg


from habitat.config.default_structured_configs import register_hydra_plugin

# Core OVON registration (needed for all baselines)
from ovon.config import HabitatConfigPlugin as OVONConfigPlugin
register_hydra_plugin(OVONConfigPlugin)
importlib.import_module("ovon.task.sensors")
importlib.import_module("ovon.measurements")
importlib.import_module("ovon.dataset")
importlib.import_module("ovon.task.simulator")

# Widely accepted HM3D names. Also used by VLFM.
# See https://arxiv.org/pdf/2007.00643z, https://aihabitat.org/challenge/2023/#footnote-3/

_HM3D_GOAL_NAMES = {
    "plant": "potted plant",
    "tv_monitor": "tv",
    "sofa": "couch",
}

# VLFM-specific registration (only present in VLFM containers)
try:
    importlib.import_module("frontier_exploration")
    importlib.import_module("vlfm.measurements.traveled_stairs")
    importlib.import_module("vlfm.obs_transformers.resize")
    importlib.import_module("vlfm.policy.habitat_policies")
except ImportError:
    pass

# ── ROS2 message imports ────────────────────────────────────────────────
from ros_msgs.msg import (
    Observation as ObservationMsg,
    EpisodeInfo as EpisodeInfoMsg,
    VelocityCommand as VelocityCommandMsg,
    AgentDone as AgentDoneMsg,
)
from ros_msgs.srv import ResetEpisode, SyncStep

from ros_helpers import (
    DONE_SENTINEL, episode_hash, pack_obs_fields,
    build_step_result_msg, build_obs_msg,
)
from habitat_env import HabitatEnvWrapper, AsyncHabitatEnv
from video_recorder import EnvVideoRecorder

from async_ui import (
    VizState, show_live_viz, start_viz_web_server,
    make_kbd_state, poll_keyboard_cmd,
)


def _array_summary(arr: Any) -> dict:
    a = np.asarray(arr)
    out = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
    }
    if a.size > 0:
        out["min"] = float(np.min(a))
        out["max"] = float(np.max(a))
        out["sum"] = float(np.sum(a, dtype=np.float64))
        out["nnz"] = int(np.count_nonzero(a))
    else:
        out["min"] = None
        out["max"] = None
        out["sum"] = None
        out["nnz"] = 0
    out["hash"] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    return out


def _obs_summary(obs: Dict[str, Any]) -> dict:
    summary = {}
    for key in ("rgb", "depth", "gps", "gps_3d", "compass", "heading", "sensor_pose", "objectgoal"):
        if key in obs:
            summary[key] = _array_summary(obs[key])
    return summary


_BENCHMARK_ENV_CONFIGS = {
    "ovon": {
        "dataset_split": "val_unseen",
        "success_distance": 0.25,
        "max_episode_steps": 500,
        "agent_height": 1.41,
        "camera_height": 1.31,
        "agent_radius": 0.17,
        "rgb_width": 360,
        "rgb_height": 640,
        "hfov": 42,
        "min_depth": 0.5,
        "max_depth": 5.0,
        "allow_sliding": False,
        "forward_step_size": 0.25,
        "turn_angle": 30,
        "tilt_angle": 30,
    },
    "hm3d": {
        "dataset_split": "val",
        "success_distance": 0.1,
        "max_episode_steps": 500,
        "agent_height": 0.88,
        "camera_height": 0.88,
        "agent_radius": 0.18,
        "rgb_width": 640,
        "rgb_height": 480,
        "hfov": 79,
        "min_depth": 0.5,
        "max_depth": 5.0,
        "allow_sliding": False,
        "forward_step_size": 0.25,
        "turn_angle": 30,
        "tilt_angle": 30,
    },
}


def _benchmark_env_config(benchmark: str) -> dict:
    if benchmark in ("hm3d_v1", "hm3d_v2"):
        benchmark = "hm3d"
    try:
        return _BENCHMARK_ENV_CONFIGS[benchmark]
    except KeyError as exc:
        raise ValueError(f"Unsupported BENCHMARK={benchmark!r}") from exc


def _apply_benchmark_overrides(config: DictConfig, benchmark: str) -> dict:
    overrides = _benchmark_env_config(benchmark)
    sim_cfg = config.habitat.simulator
    agent_cfg = sim_cfg.agents.main_agent
    sensors_cfg = agent_cfg.sim_sensors

    config.habitat.environment.max_episode_steps = overrides["max_episode_steps"]
    sim_cfg.forward_step_size = overrides["forward_step_size"]
    sim_cfg.turn_angle = overrides["turn_angle"]
    sim_cfg.tilt_angle = overrides["tilt_angle"]
    sim_cfg.habitat_sim_v0.allow_sliding = overrides["allow_sliding"]

    agent_cfg.height = overrides["agent_height"]
    agent_cfg.radius = overrides["agent_radius"]

    for sensor_name in ("rgb_sensor", "depth_sensor", "semantic_sensor"):
        sensor_cfg = getattr(sensors_cfg, sensor_name, None)
        if sensor_cfg is None:
            continue
        sensor_cfg.width = overrides["rgb_width"]
        sensor_cfg.height = overrides["rgb_height"]
        sensor_cfg.hfov = overrides["hfov"]
        sensor_cfg.position = [0.0, overrides["camera_height"], 0.0]
        if sensor_name == "depth_sensor":
            sensor_cfg.min_depth = overrides["min_depth"]
            sensor_cfg.max_depth = overrides["max_depth"]
            sensor_cfg.normalize_depth = True

    measurements_cfg = getattr(config.habitat.task, "measurements", None)
    if measurements_cfg is not None:
        success_cfg = getattr(measurements_cfg, "success", None)
        if success_cfg is not None:
            success_cfg.success_distance = overrides["success_distance"]
        failure_modes_cfg = getattr(measurements_cfg, "failure_modes", None)
        if failure_modes_cfg is not None and hasattr(failure_modes_cfg, "success_distance"):
            failure_modes_cfg.success_distance = overrides["success_distance"]

    base_explorer_cfg = getattr(getattr(config.habitat.task, "lab_sensors", None), "base_explorer", None)
    if base_explorer_cfg is not None:
        base_explorer_cfg.forward_step_size = overrides["forward_step_size"]
        base_explorer_cfg.turn_angle = overrides["turn_angle"]
        base_explorer_cfg.success_distance = overrides["success_distance"]
        base_explorer_cfg.fov = overrides["hfov"]

    return overrides


_OVON_FAILURE_MODE_KEYS = (
    "stop_too_far",
    "stop_failure",
    "recognition_failure",
    "misidentification",
    "exploration",
)


def _ovon_failure_mode_from_info(
    info: dict,
    success: float,
    failure_reason: str,
    min_d2g: float,
    success_distance: float,
) -> str:
    if success:
        return "success"
    failure_modes = info.get("failure_modes")
    if not isinstance(failure_modes, dict):
        return ""
    if failure_reason == "timeout":
        area_seen = float(failure_modes.get("area_seen", 0.0) or 0.0)
        if area_seen < 0.01:
            return "exploration"
        if min_d2g < success_distance:
            return "stop_failure"
        return "recognition_failure"
    for mode in _OVON_FAILURE_MODE_KEYS:
        if failure_modes.get(mode):
            return mode
    return ""


class EnvNode(Node):
    """Unified ROS2 node hosting the Habitat simulator."""

    def __init__(self, config: DictConfig, benchmark: str = "ovon", video_dir: str = ""):
        super().__init__("env_node")
        self._benchmark = benchmark
        self._cb = MutuallyExclusiveCallbackGroup()
        mode_label = "async" if self.__class__.__name__ == "AsyncEnvNode" else "sync"
        self.get_logger().info(f"Env node starting in {mode_label} mode")

        with read_write(config):
            overrides = _apply_benchmark_overrides(config, benchmark)
            if benchmark == "ovon":
                config.habitat.task.lab_sensors.pop("objectgoal_sensor", None)
            config.habitat_baselines.num_environments = 1

        self.get_logger().info(
            "Forced %s env config: split=%s success=%.2f max_steps=%d camera=%dx%d hfov=%s depth=[%.1f, %.1f]"
            % (
                benchmark,
                overrides["dataset_split"],
                overrides["success_distance"],
                overrides["max_episode_steps"],
                overrides["rgb_width"],
                overrides["rgb_height"],
                overrides["hfov"],
                overrides["min_depth"],
                overrides["max_depth"],
            )
        )

        self._env = self._make_env(config)
        self._total_episodes = self._env.number_of_episodes
        self.get_logger().info(f"Loaded {self._total_episodes} episodes")

        self._step_id = 0
        self._episode_count = 0
        self._ep_hash = 0
        self._scene_id = ""
        self._episode_id = ""
        self._target_name = ""
        self._ep_reward = 0.0
        self._stats: Dict[str, dict] = {}
        self._successes = 0
        self._total = 0
        self._ep_start_time = 0.0
        self._geodesic = 0.0
        self._sct_values: list = []
        self._linear_speed = 0.25
        self._episode_path_length = 0.0
        self._last_gps = None
        self._trace_path = os.environ.get("ENV_TRACE_PATH", "").strip()
        self._trace_episodes = []
        self._episode_trace = None
        self._max_episode_steps = int(overrides["max_episode_steps"])
        self._success_distance = float(overrides["success_distance"])
        self._failure_reason = ""
        self._ep_min_d2g = float("inf")
        self._traj = []  # per-episode [x, z, yaw, t, step]; dumped at episode end
        self._traj_min_step_m2 = 0.0025  # 0.05 m² distance subsample

        # Video
        self._video = EnvVideoRecorder(video_dir)

        # ROS2 interfaces
        self._pub_obs = self.create_publisher(ObservationMsg, "/obs", 10)
        self._pub_ep = self.create_publisher(EpisodeInfoMsg, "/episode_info", 10)
        self._pub_ep_json = self.create_publisher(String, "/episode_info_json", 10)
        self._srv_reset = self.create_service(ResetEpisode, "/reset_episode", self._handle_reset, callback_group=self._cb)
        self._srv_step = self.create_service(SyncStep, "/sync_step", self._handle_sync_step, callback_group=self._cb)

        self.get_logger().info("Env node ready.")

    def _make_env(self, config: DictConfig):
        return HabitatEnvWrapper(config)

    # ── Reset service ────────────────────────────────────────────────────
    def _handle_reset(self, req, resp):
        obs = self._env.reset()
        ep = self._env.current_episode
        self._scene_id = ep.scene_id
        self._episode_id = str(ep.episode_id)
        self._traj = []
        self._target_name = self._get_target_name(ep)
        self._ep_hash = episode_hash(self._scene_id, self._episode_id)
        self._step_id = 0
        self._ep_reward = 0.0
        self._ep_start_time = time.time()
        self._geodesic = (float(ep.info.get("geodesic_distance", 0)) if hasattr(ep, "info") and isinstance(ep.info, dict) else 0.0)
        self._failure_reason = ""
        self._ep_min_d2g = float("inf")
        self._episode_path_length = 0.0
        self._last_gps = self._gps_from_obs(obs)
        self._episode_count += 1
        self._video.new_episode(self._scene_id, self._episode_id)

        self._episode_trace = None
        if self._trace_path:
            self._episode_trace = {
                "scene_id": self._scene_id,
                "episode_id": self._episode_id,
                "goal_name": self._get_target_name(ep),
                "start_position": list(getattr(ep, "start_position", []) or []),
                "start_rotation": list(getattr(ep, "start_rotation", []) or []),
                "first_observation": _obs_summary(obs),
                "actions": [],
                "final": {},
            }
            self._write_trace()

        self.get_logger().info(
            f"Reset → ep {self._episode_count}/{self._total_episodes} "
            f"scene={os.path.basename(self._scene_id)} id={self._episode_id}"
        )

        episode_info = self._episode_info_payload(ep)
        f = pack_obs_fields(obs, self._step_id, self._ep_hash)
        resp.observation = build_step_result_msg(f, 0.0, False, episode_info)
        resp.scene_id = self._scene_id
        resp.episode_id = self._episode_id
        resp.episode_id_hash = self._ep_hash
        resp.episodes_remaining = self._episode_count < self._total_episodes

        self._publish_episode_info(ep, episode_info)
        return resp

    def _get_target_name(self, episode) -> str:
        if hasattr(episode, "object_category"):
            return self._format_target_name(str(episode.object_category))
        if hasattr(episode, "goals") and episode.goals:
            goal = episode.goals[0]
            if hasattr(goal, "object_category"):
                return self._format_target_name(str(goal.object_category))
            if hasattr(goal, "object_name"):
                return self._format_target_name(str(goal.object_name))
        if hasattr(episode, "info") and isinstance(episode.info, dict):
            return self._format_target_name(str(episode.info.get("object_category", "unknown object")))
        return "unknown object"

    def _format_target_name(self, name: str) -> str:
        if self._benchmark in ("hm3d", "hm3d_v1", "hm3d_v2"):
            return _HM3D_GOAL_NAMES.get(name, name)
        return name

    def _episode_info_payload(self, episode):
        payload = {
            "scene_id": self._scene_id,
            "episode_id": self._episode_id,
            "episode_id_hash": int(self._ep_hash),
            "target": self._get_target_name(episode),
            "geodesic_distance": self._geodesic,
            "success_distance": self._success_distance,
            "start_position": list(getattr(episode, "start_position", []) or []),
            "start_rotation": list(getattr(episode, "start_rotation", []) or []),
        }
        return payload

    def _publish_episode_info(self, episode, payload=None):
        ep_msg = EpisodeInfoMsg()
        ep_msg.scene_id = self._scene_id
        ep_msg.episode_id = self._episode_id
        ep_msg.episode_id_hash = self._ep_hash
        self._pub_ep.publish(ep_msg)

        if payload is None:
            payload = self._episode_info_payload(episode)

        ep_json = String()
        ep_json.data = json.dumps(payload)
        self._pub_ep_json.publish(ep_json)

    def _map_action(self, action: int) -> int:
        return int(action)

    # ── Sync step service ────────────────────────────────────────────────
    def _handle_sync_step(self, req, resp):
        self._step_id += 1
        obs, reward, done, info = self._env.step(self._map_action(req.action))
        self._update_path_length_from_obs(obs)
        self._ep_reward += reward
        self._update_min_d2g(info)
        self._video.record_step(obs, info, self._step_id)
        f = pack_obs_fields(obs, self._step_id, self._ep_hash)
        resp.result = build_step_result_msg(f, reward, done, info)
        if self._episode_trace is not None:
            action_record = {
                "step": int(self._step_id),
                "action": int(req.action),
                "done": bool(done),
            }
            if "distance_to_goal" in info:
                action_record["distance_to_goal"] = self._metric_float(
                    info.get("distance_to_goal", -1), "distance_to_target"
                )
            self._episode_trace["actions"].append(action_record)
        if done:
            if int(req.action) == HabitatEnvWrapper.STOP:
                self._failure_reason = "agent_stop"
            elif self._step_id >= self._max_episode_steps:
                self._failure_reason = "timeout"
            else:
                self._failure_reason = "episode_over"
            self._log_episode(info)
        return resp

    # ── Metrics ──────────────────────────────────────────────────────────

    def _episode_step_count(self) -> int:
        return int(self._step_id)

    def _log_episode(self, info: dict):
        s = self._metric_float(info.get("success", 0), "composite_success", "task_success")
        spl = self._metric_float(info.get("spl", 0), "composite_spl")
        soft_spl = self._metric_float(info.get("soft_spl", 0), "composite_softspl")
        distance_to_goal = self._metric_float(info.get("distance_to_goal", -1), "distance_to_target")
        if s:
            self._successes += 1
        self._total += 1

        ct = time.time() - self._ep_start_time
        ot = self._geodesic / self._linear_speed if self._geodesic > 0 else 0
        sct = float(s) * ot / max(ct, ot) if ot > 0 else 0
        self._sct_values.append(sct)

        sr = self._successes / self._total * 100
        self.get_logger().info(
            f"Episode done — succ={bool(s)} spl={spl:.3f} sct={sct:.3f} | "
            f"SR: {sr:.1f}% ({self._successes}/{self._total})"
        )

        key = f"{self._scene_id}|{self._episode_id}"
        stats = {
            "target": getattr(self, "_target_name", "") or "",
            "reward": self._ep_reward,
            "success": float(s),
            "spl": float(spl),
            "distance_to_goal": distance_to_goal,
            "soft_spl": soft_spl,
            "sct": sct,
            "completion_time": ct,
            "optimal_time": ot,
            "geodesic_distance": self._geodesic,
            "step_count": self._episode_step_count(),
            "path_length": float(self._episode_path_length),
        }
        ovon_failure_mode = _ovon_failure_mode_from_info(
            info,
            s,
            self._failure_reason,
            self._ep_min_d2g,
            self._success_distance,
        )
        if ovon_failure_mode:
            stats["ovon_failure_mode"] = ovon_failure_mode
        self._failure_reason = ""
        if isinstance(info.get("success"), dict):
            success_info = info["success"]
            for metric_name in (
                "partial_success",
                "object_success",
                "image_success",
                "description_success",
            ):
                stats[metric_name] = float(success_info.get(metric_name, 0.0))
        self._stats[key] = stats
        self._video.save_episode(self._stats[key])
        self._save_trajectory()
        if self._episode_trace is not None:
            self._episode_trace["final"] = {
                "success": float(s),
                "spl": float(spl),
                "distance_to_goal": float(distance_to_goal),
                "step_count": self._episode_step_count(),
                "path_length": float(self._episode_path_length),
            }
            self._trace_episodes.append(self._episode_trace)
            self._episode_trace = None
            self._write_trace()
        self._write_stats()

    def _record_traj_point(self) -> None:
        """Append the agent's GT world (x, z, yaw, t, step) for this tick,
        distance-subsampled. Cheap (no measure) — safe at async substep rate."""
        try:
            st = self._env.habitat_env.sim.get_agent_state(0)
            p = st.position
            x, z = float(p[0]), float(p[2])
            if self._traj:
                lx, lz = self._traj[-1][0], self._traj[-1][1]
                if (x - lx) ** 2 + (z - lz) ** 2 < self._traj_min_step_m2:
                    return
            r = st.rotation
            yaw = float(2.0 * np.arctan2(float(r.y), float(r.w)))
            self._traj.append([x, z, yaw, time.time(), int(self._step_id)])
        except Exception:
            pass

    def _save_trajectory(self) -> None:
        """Dump the episode trajectory (world coords) next to the stats, for the
        offline render to overlay. Best-effort."""
        if not self._traj:
            return
        try:
            stats_path = os.environ.get("ENV_STATS_PATH", "logs/env_stats.json")
            traj_dir = os.path.join(os.path.dirname(stats_path) or ".", "trajectories")
            os.makedirs(traj_dir, exist_ok=True)
            scene = os.path.basename(self._scene_id).split(".")[0]
            with open(os.path.join(traj_dir, f"{scene}_{self._episode_id}.json"), "w") as f:
                json.dump(
                    {"scene": self._scene_id, "episode_id": self._episode_id,
                     "max_steps": int(self._max_episode_steps), "traj": self._traj}, f
                )
        except Exception as e:
            self.get_logger().warning(f"trajectory save failed (non-fatal): {e}")

    @staticmethod
    def _gps_from_obs(obs: Dict[str, Any]):
        gps = obs.get("gps")
        if gps is None:
            return None
        arr = np.asarray(gps, dtype=np.float32).reshape(-1)
        if arr.size < 2 or not np.isfinite(arr[:2]).all():
            return None
        return arr[:2]

    def _update_path_length_from_obs(self, obs: Dict[str, Any]):
        gps = self._gps_from_obs(obs)
        if gps is None:
            return
        if self._last_gps is not None:
            self._episode_path_length += float(np.linalg.norm(gps - self._last_gps))
        self._last_gps = gps
        self._record_traj_point()

    @staticmethod
    def _metric_float(metric: Any, *keys: str) -> float:
        if isinstance(metric, dict):
            for key in keys:
                if key in metric:
                    return float(metric[key])
            return 0.0
        return float(metric)

    def _write_stats(self):
        path = os.environ.get("ENV_STATS_PATH", "logs/env_stats.json")
        agg = {}
        if self._stats:
            for k in next(iter(self._stats.values())):
                values = [v[k] for v in self._stats.values() if isinstance(v.get(k), (int, float, np.number))]
                if values:
                    agg[k] = float(np.mean(values))
        agg.update(num_episodes=self._total, num_successes=self._successes)
        ovon_counts: Dict[str, int] = {}
        for v in self._stats.values():
            om = v.get("ovon_failure_mode")
            if om:
                ovon_counts[om] = ovon_counts.get(om, 0) + 1
        if ovon_counts:
            agg["ovon_failure_modes"] = ovon_counts
        # Resilient write: try the configured path, else fall back to /tmp.
        # A logging failure (e.g. read-only/absent /opt/rtnav/logs on a manual
        # run) must never crash the eval loop.
        payload = {"aggregated": agg, "per_episode": self._stats}
        last_err = None
        for p in (path, os.path.join("/tmp", os.path.basename(path) or "env_stats.json")):
            try:
                os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
                with open(p, "w") as f:
                    json.dump(payload, f, indent=2)
                if p != path:
                    self.get_logger().warning(
                        f"stats: {path} not writable — wrote fallback {p}"
                    )
                return
            except OSError as e:
                last_err = e
        self.get_logger().warning(f"stats write failed (non-fatal): {last_err}")

    def _update_min_d2g(self, info: dict) -> None:
        d = info.get("distance_to_goal")
        if d is None:
            return
        try:
            self._ep_min_d2g = min(self._ep_min_d2g, float(d))
        except (TypeError, ValueError):
            pass

    def _write_trace(self):
        if not self._trace_path:
            return
        os.makedirs(os.path.dirname(self._trace_path) or ".", exist_ok=True)
        payload = {
            "source": "ros_env_node",
            "episodes": self._trace_episodes,
            "current_episode": self._episode_trace,
        }
        with open(self._trace_path, "w") as f:
            json.dump(payload, f, indent=2)

    def shutdown(self):
        self._write_stats()
        self._write_trace()
        self._env.close()


class AsyncEnvNode(EnvNode):
    """ROS2 node hosting the Habitat simulator in async mode.

    Drives the sim from /cmd_vel at ``sim_hz`` (default 30 Hz). Inherits
    :class:`EnvNode`; replaces the env wrapper with :class:`AsyncHabitatEnv`,
    adds Bullet physics, /agent_done handling, and an optional debug UI (live
    viz, web viz, keyboard teleop).
    """

    def __init__(self, config: DictConfig, benchmark: str = "ovon", video_dir: str = ""):
        super().__init__(config, benchmark=benchmark, video_dir=video_dir)

        # /sync_step is sync-only; async drives the sim from /cmd_vel.
        self.destroy_service(self._srv_step)
        self._srv_step = None

        self._async_timeout = 500
        self._linear_speed = 0.25
        self._vel_cmd = VelocityCommandMsg()
        self._vel_lock = threading.Lock()
        self._agent_step_id = 0
        self._pose_agent_step_id = 0
        self._agent_step_sensor_pose = np.zeros(3, dtype=np.float64)
        self._async_done = False
        self._ep_min_d2g = float("inf")

        self._viz_state = self._init_viz_state()
        self._kbd_state = make_kbd_state(
            enabled=os.environ.get("ASYNC_KEYBOARD_CONTROL", "0") != "0",
            logger=self.get_logger(),
        )
        self._log_physics_config(config)
        self._dump_sensor_suite()

        self.create_subscription(
            VelocityCommandMsg, "/cmd_vel", self._on_vel, 1, callback_group=self._cb,
        )
        self.create_subscription(
            AgentDoneMsg, "/agent_done", self._on_done, 10, callback_group=self._cb,
        )
        hz = self.declare_parameter("sim_hz", 30.0).value
        self._sim_dt = 1.0 / float(hz)
        self._sim_timer = self.create_timer(
            self._sim_dt, self._async_tick, callback_group=self._cb,
        )
        self.get_logger().info(
            f"Async env node ready (sim_hz={hz}, timeout={self._async_timeout}s)."
        )

    # ── Overrides ────────────────────────────────────────────────────────

    def _make_env(self, config: DictConfig):
        self._maybe_enable_physics(config)
        with read_write(config):
            config.habitat.environment.max_episode_steps = 0
        return AsyncHabitatEnv(config)

    def _handle_reset(self, req, resp):
        self._async_done = False
        self._vel_cmd = VelocityCommandMsg()
        self._agent_step_id = 0
        self._pose_agent_step_id = 0
        self._agent_step_sensor_pose = np.zeros(3, dtype=np.float64)
        self._ep_min_d2g = float("inf")
        self._failure_reason = ""
        resp = super()._handle_reset(req, resp)

        last_obs = getattr(self._env, "_last_obs", None) or {}
        self._show_live_viz(last_obs)
        return resp

    def _log_episode(self, info: dict):
        super()._log_episode(info)
        key = f"{self._scene_id}|{self._episode_id}"
        if key in self._stats:
            self._stats[key]["sim_step_count"] = int(self._step_id)
            self._stats[key]["min_d2g_during_episode"] = float(self._ep_min_d2g)
            self._write_stats()

    def _episode_step_count(self) -> int:
        return int(self._agent_step_id)

    # ── Init helpers ─────────────────────────────────────────────────────

    def _maybe_enable_physics(self, config: DictConfig) -> None:
        mode = os.environ.get("HABITAT_ENABLE_PHYSICS", "auto").strip().lower()
        if mode in ("0", "false", "off", "no"):
            return
        if mode not in ("auto", "1", "true", "on", "yes"):
            raise ValueError(
                f"Invalid HABITAT_ENABLE_PHYSICS={mode!r} (use auto/1/0)."
            )
        if not habitat_sim.built_with_bullet:
            raise RuntimeError(
                "HABITAT_ENABLE_PHYSICS requested but habitat-sim has no Bullet."
            )
        # patch_config() in main() calls OmegaConf.set_readonly on the cfg, so we
        # need a read_write context to mutate the sim subtree here.
        with read_write(config):
            config.habitat.simulator.habitat_sim_v0.enable_physics = True
            cfg_file = os.environ.get("HABITAT_PHYSICS_CONFIG_FILE", "")
            if cfg_file:
                config.habitat.simulator.habitat_sim_v0.physics_config_file = cfg_file

    def _log_physics_config(self, config: DictConfig) -> None:
        try:
            sim_v0 = config.habitat.simulator.habitat_sim_v0
            self.get_logger().info(
                f"Physics: enable={bool(getattr(sim_v0, 'enable_physics', False))} "
                f"cfg={getattr(sim_v0, 'physics_config_file', '')!r} "
                f"built_with_bullet={bool(habitat_sim.built_with_bullet)}"
            )
        except Exception:
            pass

    def _dump_sensor_suite(self) -> None:
        try:
            suite = self._env.habitat_env.sim.sensor_suite
            self.get_logger().info(f"[sim-diag] sensor_suite has {len(suite.sensors)} sensors:")
            for uuid, sensor in suite.sensors.items():
                try:
                    shape = getattr(sensor.observation_space, "shape", "?")
                    dtype = getattr(sensor.observation_space, "dtype", "?")
                except Exception:
                    shape = dtype = "?"
                self.get_logger().info(
                    f"[sim-diag]   {uuid!r}: shape={shape} dtype={dtype} class={type(sensor).__name__}"
                )
        except Exception as e:
            self.get_logger().warn(f"[sim-diag] sensor dump failed: {e}")

    def _init_viz_state(self) -> Optional[VizState]:
        state = VizState(
            live_viz=os.environ.get("SIM_LIVE_VIZ", "0") != "0",
            web_viz=os.environ.get("SIM_VIZ_WEB", "0") != "0",
            web_port=int(os.environ.get("SIM_VIZ_WEB_PORT", "8767")),
        )
        if state.live_viz:
            try:
                import cv2  # noqa: F401
                self.get_logger().info(
                    f"[sim-viz] SIM_LIVE_VIZ=1; DISPLAY={os.environ.get('DISPLAY', '')!r}"
                )
            except Exception as e:
                self.get_logger().warn(f"[sim-viz] cv2 import failed — disabling: {e}")
                state.live_viz = False
        if state.web_viz:
            start_viz_web_server(state, step_id_getter=lambda: self._step_id, logger=self.get_logger())
        return state

    # ── /cmd_vel + /agent_done + sim tick ────────────────────────────────

    def _on_vel(self, msg):
        incoming_step_id = int(msg.agent_step_id)
        with self._vel_lock:
            if incoming_step_id < self._agent_step_id:
                return
            self._vel_cmd = msg
            self._agent_step_id = incoming_step_id

    def _on_done(self, msg):
        if not msg.stop or self._async_done:
            return
        incoming_step_id = int(msg.agent_step_id)
        with self._vel_lock:
            if incoming_step_id < self._agent_step_id:
                return
            self._agent_step_id = incoming_step_id
            self._vel_cmd = VelocityCommandMsg()
            self._vel_cmd.agent_step_id = incoming_step_id
        if self._episode_count == 0:
            self.get_logger().warn("Ignoring /agent_done: no episode has started yet.")
            return
        self.get_logger().info("Agent STOP signal.")
        self._failure_reason = "agent_stop"

        if self._habitat_episode_over():
            self._finalize(*self._fetch_cached_episode_state())
            return

        try:
            obs, reward, done, info = self._env.step(self._map_action(self._env.STOP))
        except AssertionError as exc:
            self.get_logger().warn(f"step(STOP) raised; using cached state ({exc})")
            self._finalize(*self._fetch_cached_episode_state())
            return

        self._ep_reward += reward
        self._update_min_d2g(info)
        self._finalize(info, obs, log=bool(done))

    def _async_tick(self):
        if self._episode_count == 0 or self._async_done:
            return

        now = time.time()

        if now - self._ep_start_time > self._async_timeout:
            self.get_logger().info(f"Wall-clock timeout ({self._async_timeout}s)")
            obs, reward, done, info = self._env.step(self._env.STOP)
            self._ep_reward += reward
            self._update_min_d2g(info)
            if not done:
                info = {"success": 0, "spl": 0, "distance_to_goal": -1, "soft_spl": 0}
            self._failure_reason = "timeout"
            self._finalize(info, obs)
            return

        kbd_cmd = poll_keyboard_cmd(self._kbd_state) if self._kbd_state else None
        if self._kbd_state and self._kbd_state.quit_requested:
            self._handle_keyboard_quit()
            return

        if kbd_cmd is not None:
            cmd_lin, _cmd_lin_y, cmd_ang = kbd_cmd
            cmd_pitch = 0.0
            cmd_agent_step_id = self._agent_step_id
        else:
            with self._vel_lock:
                vel = self._vel_cmd
            cmd_lin = float(vel.linear_velocity)
            cmd_ang = float(vel.angular_velocity)
            cmd_pitch = float(getattr(vel, "camera_pitch_velocity", 0.0))
            cmd_agent_step_id = int(vel.agent_step_id)

        obs, reward, done, info = self._env.velocity_step(
            cmd_lin, cmd_ang, dt=self._sim_dt, camera_pitch=cmd_pitch,
        )
        obs = self._with_agent_step_sensor_pose(obs, cmd_agent_step_id)
        obs["applied_agent_step_id"] = int(cmd_agent_step_id)
        self._step_id += 1
        self._ep_reward += reward
        self._update_path_length_from_obs(obs)
        self._update_min_d2g(info)
        self._video.record_step(obs, info, self._step_id)
        self._show_live_viz(obs)
        self._pub_obs.publish(
            build_obs_msg(pack_obs_fields(obs, self._step_id, self._ep_hash))
        )

        if done:
            self._finalize(info, obs)

    def _with_agent_step_sensor_pose(self, obs: Dict[str, Any], agent_step_id: int) -> Dict[str, Any]:
        if agent_step_id != self._pose_agent_step_id:
            self._pose_agent_step_id = agent_step_id
            self._agent_step_sensor_pose = np.zeros(3, dtype=np.float64)

        tick_pose = np.asarray(obs.get("sensor_pose", [0.0, 0.0, 0.0]), dtype=np.float64)
        self._agent_step_sensor_pose = self._compose_sensor_pose(
            self._agent_step_sensor_pose, tick_pose
        )
        out = dict(obs)
        out["sensor_pose"] = self._agent_step_sensor_pose.astype(np.float32)
        return out

    @staticmethod
    def _compose_sensor_pose(total, delta):
        dx, dy, dtheta = np.asarray(delta, dtype=np.float64).reshape(-1)[:3]
        c, s = np.cos(total[2]), np.sin(total[2])
        return np.array(
            [
                total[0] + c * dx - s * dy,
                total[1] + s * dx + c * dy,
                total[2] + dtheta,
            ],
            dtype=np.float64,
        )

    def _handle_keyboard_quit(self):
        self.get_logger().info("Keyboard STOP signal.")
        obs, reward, done, info = self._env.step(self._env.STOP)
        self._ep_reward += reward
        self._kbd_state.quit_requested = False
        if done:
            self._failure_reason = "keyboard_stop"
        self._finalize(info, obs, log=bool(done))

    def _finalize(self, info, obs, *, log=True):
        self._step_id += 1
        self._video.record_step(obs, info, self._step_id)
        if log:
            self._log_episode(info)
        self._pub_done_sentinel(obs)
        self._async_done = True

    def _pub_done_sentinel(self, obs):
        self._pub_obs.publish(build_obs_msg(pack_obs_fields(obs, DONE_SENTINEL, self._ep_hash)))

    def _habitat_episode_over(self) -> bool:
        try:
            return bool(getattr(self._env.habitat_env, "episode_over", False))
        except Exception:
            return False

    def _fetch_cached_episode_state(self):
        try:
            info = self._env.habitat_env.task.measurements.get_metrics()
        except Exception as exc:
            msg = (
                "[env_node] FATAL: failed to fetch cached episode metrics; "
                f"episode_count={self._episode_count} ep_hash={self._ep_hash} "
                f"env={type(self._env).__name__} error={exc!r}"
            )
            self.get_logger().error(msg)
            raise RuntimeError(msg) from exc
        obs = getattr(self._env, "_last_obs", None) or {}
        return info, obs

    def _show_live_viz(self, obs):
        if not self._viz_state:
            return
        show_live_viz(
            obs,
            step_id=self._step_id,
            episode_count=self._episode_count,
            viz_state=self._viz_state,
            skip_waitkey=bool(self._kbd_state and self._kbd_state.enabled),
            logger=self.get_logger(),
        )

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

class _ConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath):
        search_path.append(provider="habitat", path="config/")

register_hydra_plugin(_ConfigPlugin)

# Determine config from env vars. Environment configs are shared across baselines.
_BENCHMARK = os.environ.get("BENCHMARK")
_MODE = os.environ.get("MODE")
_CONFIGS_BY_BENCHMARK = {
    "hm3d": "hm3d",
    "hm3d_v1": "hm3d_v1",
    "hm3d_v2": "hm3d_v2",
    "ovon": "ovon",
}
_ASYNC_CONFIG_NAMES = {
    "vlfm_ovon": "experiments/vlfm_objectnav_ovon",
    "vlfm_hm3d": "experiments/vlfm_objectnav_hm3d",
    "rtnav_ovon": "experiments/rt_ovn_objectnav_ovon",
    "rtnav_hm3d": "experiments/rt_ovn_objectnav_hm3d",
    "rtnav_hm3d_v1": "experiments/rt_ovn_objectnav_hm3d_v1",
    "rtnav_hm3d_v2": "experiments/rt_ovn_objectnav_hm3d_v2",
}


def _resolve_config_name() -> str:
    explicit_config_name = os.environ.get("HYDRA_CONFIG_NAME", "")
    if explicit_config_name:
        return explicit_config_name

    if _MODE == "async":
        baseline = os.environ.get("BASELINE")
        name = _ASYNC_CONFIG_NAMES.get(
            f"{baseline}_{_BENCHMARK}", _ASYNC_CONFIG_NAMES["vlfm_ovon"],
        )
        if _BENCHMARK.lower().startswith("hm3d") and "ovon" in name.lower():
            name = _ASYNC_CONFIG_NAMES.get(f"{baseline}_hm3d", name)
        return name

    config_name = _CONFIGS_BY_BENCHMARK.get(_BENCHMARK)
    if config_name is None:
        raise ValueError(
            "Missing HYDRA_CONFIG_NAME for unsupported BENCHMARK "
            f"{_BENCHMARK!r}. Expected one of: {sorted(_CONFIGS_BY_BENCHMARK)}."
        )
    return config_name


_config_name = _resolve_config_name()
_config_path = os.environ.get("HYDRA_CONFIG_PATH") or (
    "/opt/vlfm/config" if _MODE == "async" else "/opt/rt_ovn/env_config"
)

@hydra.main(version_base=None, config_path=_config_path, config_name=_config_name)
def main(cfg: DictConfig):
    # Seeding
    random.seed(cfg.habitat.seed)
    np.random.seed(cfg.habitat.seed)
    torch.manual_seed(cfg.habitat.seed)
    if getattr(cfg.habitat_baselines, "force_torch_single_threaded", False):
        if torch.cuda.is_available():
            torch.set_num_threads(1)

    cfg = patch_config(cfg)
    with read_write(cfg):
        cfg.habitat.environment.iterator_options.shuffle = False
        if _MODE == "async":
            eval_split = os.environ.get("EVAL_SPLIT", "")
            if eval_split:
                cfg.habitat_baselines.eval.split = eval_split
            cfg.habitat.dataset.split = cfg.habitat_baselines.eval.split
            if "evaluate" in cfg.habitat_baselines:
                cfg.habitat_baselines.evaluate = True
            if "load_resume_state_config" in cfg.habitat_baselines:
                cfg.habitat_baselines.load_resume_state_config = False
            if os.environ.get("DISABLE_TOP_DOWN_MAP", "0") not in ("0", "", "false", "no"):
                try:
                    measures = cfg.habitat.task.measurements
                    if hasattr(measures, "top_down_map"):
                        del measures["top_down_map"]
                        print("[env_node] top_down_map disabled")
                except Exception as e:
                    print(f"[env_node] could not remove top_down_map: {e}")
        else:
            benchmark_cfg = _benchmark_env_config(_BENCHMARK)
            eval_split = os.environ.get("EVAL_SPLIT") or benchmark_cfg["dataset_split"]
            cfg.habitat_baselines.eval.split = eval_split
            cfg.habitat.dataset.split = cfg.habitat_baselines.eval.split

    # Match the official OpenFMNav/L3MVN eval ordering.
    dataset = make_dataset(cfg.habitat.dataset.type)
    scenes = dataset.get_scenes_to_load(cfg.habitat.dataset)
    with read_write(cfg):
        cfg.habitat.dataset.content_scenes = scenes

    mode, benchmark, video_dir = _MODE, _BENCHMARK, os.environ.get("VIDEO_DIR", "")
    assert mode in ("sync", "async"), f"Invalid MODE: {mode}"
    assert benchmark in _CONFIGS_BY_BENCHMARK, f"Invalid BENCHMARK: {benchmark}"

    print(f"Starting EnvNode with MODE={mode} BENCHMARK={benchmark}")
    rclpy.init()
    node_cls = AsyncEnvNode if mode == "async" else EnvNode
    node = node_cls(cfg, benchmark=benchmark, video_dir=video_dir)

    # Episode slicing for parallel evaluation
    ep_offset = int(os.environ.get("EPISODE_OFFSET", "0"))
    ep_count = int(os.environ.get("EPISODE_COUNT", "0"))
    if ep_count > 0:
        hab_env = node._env.habitat_env
        all_eps = hab_env.episodes
        sliced = all_eps[ep_offset : ep_offset + ep_count]
        hab_env.episodes = sliced
        node._total_episodes = len(sliced)
        node.get_logger().info(
            f"Episode slice: offset={ep_offset} count={ep_count} "
            f"→ {len(sliced)} episodes (of {len(all_eps)} total)"
        )

    # EPISODE_KEYS=<scene_stem>|<ep_id>,... — arbitrary subset filter
    ep_keys = os.environ.get("EPISODE_KEYS", "").strip()
    if ep_keys:
        wanted = {tuple(k.strip().split("|", 1)) for k in ep_keys.split(",") if "|" in k}
        hab_env = node._env.habitat_env
        hab_env.episodes = [
            e for e in hab_env.episodes
            if (os.path.basename(e.scene_id).split(".")[0], str(e.episode_id)) in wanted
        ]
        node._total_episodes = len(hab_env.episodes)
        node.get_logger().info(f"EPISODE_KEYS: kept {len(hab_env.episodes)} episodes")

    try:
        # IMPORTANT: SingleThreadedExecutor — Habitat-sim's OpenGL context
        # is bound to the thread that created it (main thread).
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    if "--viz" in sys.argv:
        sys.argv.remove("--viz")
        os.environ["SIM_LIVE_VIZ"] = "1"
    if "--viz_web" in sys.argv:
        sys.argv.remove("--viz_web")
        os.environ["SIM_VIZ_WEB"] = "1"
    for a in list(sys.argv):
        if a.startswith("--viz_web_port="):
            os.environ["SIM_VIZ_WEB_PORT"] = a.split("=", 1)[1]
            sys.argv.remove(a)
    if "--keyboard" in sys.argv:
        sys.argv.remove("--keyboard")
        os.environ["ASYNC_KEYBOARD_CONTROL"] = "1"

    main()
