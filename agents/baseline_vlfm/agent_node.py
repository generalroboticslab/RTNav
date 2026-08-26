#!/usr/bin/env python3
"""VLFM agent node — loads HabitatITMPolicyV2 and runs eval via ROS2.

VLFM uses either OWLv2 for all text goals or the original YOLOv7 +
GroundingDINO detector stack, plus BLIP2, SAM, and PointNav. VLM servers run
in the same container and are accessed over HTTP.
"""

import os
import json
import threading
import time
from typing import Tuple

import torch
from std_msgs.msg import String

from base_agent import BaseAgentNode, run_agent
from ros_helpers import obs_dict_to_tensor_batch

_SENSOR_CONFIGS = {
    "ovon": {"camera_height": 1.31, "min_depth": 0.5, "max_depth": 5.0, "camera_fov": 42, "image_width": 360, "image_height": 640},
    "hm3d": {"camera_height": 0.88, "min_depth": 0.5, "max_depth": 5.0,  "camera_fov": 79, "image_width": 640, "image_height": 480},
}

class VLFMAgentNode(BaseAgentNode):
    """VLFM agent with ITM policy."""

    def __init__(self, mode: str = "sync"):
        super().__init__("vlfm_agent_node", mode)
        self._policy = None
        self._benchmark = os.environ.get("BENCHMARK", os.environ.get("VLFM_CONFIG", "ovon"))
        self._episode_info = {}
        self._episode_info_lock = threading.Lock()
        self._episode_info_event = threading.Event()
        self._last_policy_target = None
        self._trace_path = os.environ.get("VLFM_AGENT_TRACE_PATH", "").strip()
        self._trace = None
        if self._benchmark == "ovon":
            self._episode_info_sub = self.create_subscription(
                String, "/episode_info_json", self._on_episode_info_json, 10
            )

    def _load_policy(self):
        from vlfm.policy.habitat_policies import HabitatITMPolicyV2

        benchmark = self._benchmark
        sensor_cfg = _SENSOR_CONFIGS.get(benchmark)
        if sensor_cfg is None:
            raise ValueError(f"Unknown BENCHMARK={benchmark!r}")
        self.get_logger().info(f"VLFM sensor config for {benchmark}: {sensor_cfg}")

        self._policy = HabitatITMPolicyV2(
            text_prompt="Seems like there is a target_object ahead.",
            pointnav_policy_path=os.environ.get("POINTNAV_WEIGHTS", "data/pointnav_weights.pth"),
            depth_image_shape=(224, 224),
            pointnav_stop_radius=0.9,
            use_max_confidence=False,
            object_map_erosion_size=5,
            exploration_thresh=0.0,
            obstacle_map_area_threshold=1.5,
            min_obstacle_height=0.61,
            max_obstacle_height=0.88,
            hole_area_thresh=100000,
            use_vqa=False,
            vqa_prompt="Is this ",
            coco_threshold=0.8,
            non_coco_threshold=0.4,
            agent_radius=0.18,
            dataset_type="hm3d",
            visualize=self._video_recorder.enabled,
            **sensor_cfg,
        )
        self.get_logger().info("VLFM policy loaded.")

    def _reset_policy_state(self):
        self._last_policy_target = None

    def _on_episode_info_json(self, msg):
        try:
            info = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._episode_info_lock:
            self._episode_info = info
            self._episode_info_event.set()

    def _on_episode_start(self, reset_resp, initial_obs: dict):
        if self._benchmark != "ovon":
            self._start_trace(reset_resp, {})
            return
        info = self._wait_for_episode_info(reset_resp)
        self._start_trace(reset_resp, info)

    def _wait_for_episode_info(self, reset_resp):
        self._episode_info_event.clear()
        end_time = time.time() + 1.0
        while time.time() < end_time:
            with self._episode_info_lock:
                info = dict(self._episode_info)
            if int(info.get("episode_id_hash", -1)) == int(reset_resp.episode_id_hash):
                return info
            self._episode_info_event.wait(timeout=0.05)
        raise RuntimeError(
            f"Timed out waiting for {self._benchmark} episode metadata "
            f"for episode hash {reset_resp.episode_id_hash}"
        )

    def _start_trace(self, reset_resp, episode_info: dict):
        if not self._trace_path:
            self._trace = None
            return
        os.makedirs(os.path.dirname(self._trace_path) or ".", exist_ok=True)
        self._trace = {
            "source": "ros_vlfm_agent",
            "scene_id": reset_resp.scene_id,
            "episode_id": str(reset_resp.episode_id),
            "goal_name": episode_info.get("target", ""),
            "actions": [],
        }
        self._write_trace()

    def _append_trace(self, step: int, action: int):
        if self._trace is None:
            return
        self._trace["actions"].append(int(action))
        self._trace.setdefault("steps", []).append({"step": int(step), "action": int(action)})
        self._write_trace()

    def _write_trace(self):
        if not self._trace_path or self._trace is None:
            return
        with open(self._trace_path, "w", encoding="utf-8") as f:
            json.dump(self._trace, f, indent=2)

    def _episode_target(self) -> str:
        with self._episode_info_lock:
            info = dict(self._episode_info)
        target = info.get("target") or info.get("target_category")
        target = str(target).strip() if target is not None else ""
        if not target or target == "unknown object":
            raise RuntimeError(f"Missing {self._benchmark} target in episode metadata: {info}")
        return target

    def _prepare_ovon_policy_target(self, obs_dict: dict, step: int):
        self._prepare_text_policy_target(obs_dict, step, self._episode_target())

    def _prepare_text_policy_target(self, obs_dict: dict, step: int, target: str):
        obs_dict.pop("clip_objectgoal", None)
        obs_dict.pop("objectgoal", None)
        if target == self._last_policy_target:
            return
        if step == 0 and hasattr(self._policy, "_reset"):
            self._policy._reset()
            self._policy._did_reset = True
        if hasattr(self._policy, "set_target_object"):
            self._policy.set_target_object(target)
        else:
            self._policy._target_object = target
            self._policy._non_coco_caption = f"{target} ."
        self._last_policy_target = target

    def _compute_action(self, obs_dict: dict, step: int) -> Tuple[int, dict]:
        obs_dict = dict(obs_dict)
        if self._benchmark == "ovon":
            self._prepare_ovon_policy_target(obs_dict, step)
        batch = obs_dict_to_tensor_batch(obs_dict, self._device)

        from habitat_baselines.common.tensor_dict import TensorDict
        td = TensorDict(batch)

        masks = torch.ones(1, 1, dtype=torch.bool, device=self._device) if step > 0 \
            else torch.zeros(1, 1, dtype=torch.bool, device=self._device)
        rnn = torch.zeros(1, 1, device=self._device)
        prev = torch.zeros(1, 1, dtype=torch.long, device=self._device)

        with torch.inference_mode():
            ad = self._policy.act(td, rnn, prev, masks, deterministic=False)

        pi = ad.policy_info[0] if ad.policy_info else {}
        action = ad.actions.cpu().item()
        self._append_trace(step, action)
        return action, pi


if __name__ == "__main__":
    run_agent(VLFMAgentNode, env_prefix="VLFM_")
