#!/usr/bin/env python3
"""BeliefMapNav agent node — loads HabitatITMPolicyV2 and runs eval via ROS2.

BeliefMapNav is a VLFM fork: YOLOv7 + GroundingDINO + MobileSAM run as VLM
servers in this container, and the policy adds an OpenFusion SLAM belief map
scored by GPT-4o. Set BMN_DUMMY_LLM=1 to stub the GPT-4o calls.
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


class BeliefMapNavAgentNode(BaseAgentNode):
    """BeliefMapNav agent with belief-map ITM policy."""

    def __init__(self, mode: str = "sync"):
        super().__init__("beliefmapnav_agent_node", mode)
        self._policy = None
        self._benchmark = os.environ.get("BENCHMARK", "ovon")
        self._episode_info = {}
        self._episode_info_lock = threading.Lock()
        self._episode_info_event = threading.Event()
        self._last_policy_target = None
        self._trace_path = os.environ.get("BMN_AGENT_TRACE_PATH", "").strip()
        self._trace = None
        self._trace_episodes = []
        if self._benchmark == "ovon":
            self._episode_info_sub = self.create_subscription(
                String, "/episode_info_json", self._on_episode_info_json, 10
            )

    def _load_policy(self):
        # Same determinism switch the official run uses (no-op unless
        # BMN_DETERMINISTIC=1). Must run before the policy builds its CUDA state.
        #
        # The seed must match what the official run ends up with, not just be
        # "some fixed seed": vlfm/run.py calls enable_determinism() at import,
        # but habitat_baselines/run.py then re-seeds random/numpy/torch with
        # config.habitat.seed (100). The belief map samples points with
        # np.random.choice / torch.randperm / np.random.permutation
        # (openfusion/state.py:154, 674, 1188), so a different seed produces a
        # different map and diverges from the official trace.
        from vlfm.utils.determinism import enable_determinism

        if enable_determinism(seed=int(os.environ.get("BMN_SEED", "100"))):
            self.get_logger().info("Determinism enabled for the ROS agent (seed=100).")

        from vlfm.policy.habitat_policies import HabitatITMPolicyV2

        benchmark = self._benchmark
        sensor_cfg = _SENSOR_CONFIGS.get(benchmark)
        if sensor_cfg is None:
            raise ValueError(f"Unknown BENCHMARK={benchmark!r}")
        self.get_logger().info(f"BeliefMapNav sensor config for {benchmark}: {sensor_cfg}")

        # base_objectnav_policy.py reads gpu_id/gpu_num with a bare int(), and
        # itm_policy.py joins result_path onto ./outputs; both raise on None.
        os.environ.setdefault("gpu_id", "0")
        os.environ.setdefault("gpu_num", "0")
        os.environ.setdefault("result_path", "ros_eval")

        # These must stay in sync with what HabitatMixin.from_config() resolves in
        # the official run, i.e. BeliefMapNav's VLFMConfig defaults
        # (base_objectnav_policy.py:1408) — NOT VLFM's, which differ on
        # pointnav_stop_radius, obstacle_map_area_threshold, min_obstacle_height
        # and use_vqa. use_vqa routes to the GPT-4o detection_refinement call
        # (stubbed by BMN_DUMMY_LLM), not to a BLIP2 server, so no extra VLM
        # server is needed. Verified against the official probe's config dump.
        self._policy = HabitatITMPolicyV2(
            text_prompt="Seems like there is a target_object ahead.",
            pointnav_policy_path=os.environ.get("POINTNAV_WEIGHTS", "data/pointnav_weights.pth"),
            depth_image_shape=(224, 224),
            pointnav_stop_radius=0.85,
            use_max_confidence=False,
            object_map_erosion_size=5,
            exploration_thresh=0.0,
            obstacle_map_area_threshold=0.8,
            min_obstacle_height=0.5,
            max_obstacle_height=0.88,
            hole_area_thresh=100000,
            use_vqa=True,
            vqa_prompt="Is this ",
            coco_threshold=0.8,
            non_coco_threshold=0.4,
            agent_radius=0.18,
            dataset_type="hm3d",
            visualize=self._video_recorder.enabled,
            **sensor_cfg,
        )
        # Re-assert determinism AFTER the policy is built. enable_determinism()
        # above runs before SAM2/CLIP load, and ultralytics turns TF32 back on
        # during model setup — measured as tf32_mm/tf32_cudnn True on the ROS
        # side vs False in the official run, which changed CLIP's embeddings
        # (emb_sum -176.76311 vs -176.72477) and diverged the belief map from
        # frame one.
        if os.environ.get("BMN_DETERMINISTIC", "").strip() == "1":
            import torch as _t

            _t.backends.cuda.matmul.allow_tf32 = False
            _t.backends.cudnn.allow_tf32 = False
            self.get_logger().info(
                "TF32 re-disabled after policy load (matmul=%s cudnn=%s)"
                % (_t.backends.cuda.matmul.allow_tf32, _t.backends.cudnn.allow_tf32)
            )

        self.get_logger().info("BeliefMapNav policy loaded.")

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
            "source": "ros_beliefmapnav_agent",
            "scene_id": reset_resp.scene_id,
            "episode_id": str(reset_resp.episode_id),
            "goal_name": episode_info.get("target", ""),
            "actions": [],
        }
        # Accumulate every episode in a single {"episodes": [...]} document.
        # Previously each episode overwrote the whole file, so multi-episode
        # probes only kept the last episode and compare_runs.py fell back to a
        # flat comparison of official ep0 vs ROS ep1 — a spurious mismatch.
        self._trace_episodes.append(self._trace)
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
            json.dump({"episodes": self._trace_episodes}, f, indent=2)

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
        obs_dict["gps"] = obs_dict.pop("gps_3d")
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
    run_agent(BeliefMapNavAgentNode, env_prefix="BMN_")
