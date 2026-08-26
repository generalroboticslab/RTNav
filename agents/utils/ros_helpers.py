"""Shared ROS2 <-> NumPy conversion helpers for ObjectNav evaluation."""

import hashlib
import json

import numpy as np
from sensor_msgs.msg import Image

DONE_SENTINEL = (1 << 64) - 1  # max uint64, used as episode-done signal


# ── Image conversion (no cv_bridge dependency) ──────────────────────────


def image_msg_to_numpy(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image -> numpy array."""
    if msg.encoding == "rgb8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    if msg.encoding == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )[:, :, ::-1].copy()
    raise ValueError(f"Unsupported encoding: {msg.encoding}")


def numpy_to_image_msg(arr: np.ndarray, encoding: str) -> Image:
    """Convert numpy array -> sensor_msgs/Image."""
    msg = Image()
    if arr.ndim == 2:
        msg.height, msg.width = arr.shape
        msg.step = msg.width * arr.dtype.itemsize
    else:
        msg.height, msg.width = arr.shape[:2]
        msg.step = msg.width * arr.shape[2] * arr.dtype.itemsize
    msg.encoding = encoding
    msg.is_bigendian = False
    # rclpy on ROS2 Humble (Python 3.11) stores uint8[] as a Python list,
    # so `msg.data = bytes` triggers an O(n) bytes→list conversion (~30ms for
    # a 480×640 frame). Bypass the property setter by writing a flat numpy
    # uint8 array directly to the backing _data slot; rclpy's C serialiser
    # accepts any buffer-protocol object, keeping the publish path unchanged.
    flat = np.ascontiguousarray(arr).view(np.uint8).ravel()
    try:
        msg._data = flat
    except AttributeError:
        msg.data = arr.tobytes()  # fallback for future rclpy versions
    return msg


# ── Episode hashing ─────────────────────────────────────────────────────


def episode_hash(scene_id: str, episode_id: str) -> int:
    """Deterministic uint64 hash for (scene, episode) pair."""
    return int.from_bytes(
        hashlib.sha256(f"{scene_id}|{episode_id}".encode()).digest()[:8], "little"
    )


# ── Observation packing (dict -> ROS2 message fields) ───────────────────


def pack_obs_fields(obs: dict, step_id: int, ep_hash: int) -> dict:
    """Pack observation dict into flat fields for ROS2 messages."""
    rgb = obs["rgb"]
    depth = obs["depth"]
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    fields = {
        "step_id": int(step_id),
        "applied_agent_step_id": int(
            np.asarray(obs.get("applied_agent_step_id", 0)).reshape(-1)[0]
        ),
        "episode_id_hash": int(ep_hash),
        "rgb": numpy_to_image_msg(np.ascontiguousarray(rgb), "rgb8"),
        "depth": numpy_to_image_msg(
            np.ascontiguousarray(depth.astype(np.float32)), "32FC1"
        ),
        "gps": _to_list(obs["gps"]),
        "gps_3d": _to_list(obs["gps_3d"]),
        "compass": _to_scalar(obs["compass"]),
        "heading": _to_scalar(obs["heading"]),
        "sensor_pose": _to_list(obs.get("sensor_pose", [0.0, 0.0, 0.0])),
        "camera_pitch": _to_scalar(obs.get("camera_pitch", 0.0)),
    }

    # Frontier sensor (filter NaN/inf rows)
    if "frontier_sensor" in obs:
        f = np.asarray(obs["frontier_sensor"], dtype=np.float32)
        if f.ndim == 2:
            f = f[np.isfinite(f).all(axis=1)]
        if f.ndim == 2 and f.shape[0] > 0:
            fields["num_frontiers"] = f.shape[0]
            fields["frontier_sensor"] = f.flatten().tolist()
        else:
            fields["num_frontiers"] = 0
            fields["frontier_sensor"] = []
    else:
        fields["num_frontiers"] = 0
        fields["frontier_sensor"] = []

    if "clip_objectgoal" in obs:
        fields["clip_objectgoal"] = (
            np.asarray(obs["clip_objectgoal"], dtype=np.float32).flatten().tolist()
        )
    else:
        fields["clip_objectgoal"] = []

    # Integer goal ID
    fields["objectgoal"] = (
        int(np.asarray(obs["objectgoal"]).flatten()[0])
        if "objectgoal" in obs
        else -1
    )
    return fields


# ── Message builders ─────────────────────────────────────────────────────


def build_step_result_msg(obs_fields: dict, reward: float, done: bool, info: dict):
    """Build a StepResult ROS2 message."""
    from ros_msgs.msg import StepResult as StepResultMsg

    msg = StepResultMsg()
    for k in (
        "step_id", "rgb", "depth", "gps", "gps_3d", "frontier_sensor",
        "num_frontiers", "clip_objectgoal", "objectgoal", "sensor_pose", "camera_pitch",
    ):
        setattr(msg, k, obs_fields[k])
    msg.compass = float(obs_fields["compass"])
    msg.heading = float(obs_fields["heading"])
    msg.reward = float(reward)
    msg.done = done
    msg.info_json = json.dumps(_safe_json(info))
    return msg


def build_obs_msg(obs_fields: dict):
    """Build an Observation ROS2 message."""
    from ros_msgs.msg import Observation as ObservationMsg

    msg = ObservationMsg()
    for k in (
        "step_id", "applied_agent_step_id", "episode_id_hash", "rgb", "depth", "gps", "gps_3d",
        "frontier_sensor", "num_frontiers", "clip_objectgoal", "objectgoal", "sensor_pose",
        "camera_pitch",
    ):
        setattr(msg, k, obs_fields[k])
    msg.compass = float(obs_fields["compass"])
    msg.heading = float(obs_fields["heading"])
    return msg


# ── Message -> observation dict ──────────────────────────────────────────


def step_result_to_obs_dict(result) -> dict:
    """Convert StepResult or Observation message -> observation dict."""
    rgb = image_msg_to_numpy(result.rgb)
    depth = image_msg_to_numpy(result.depth)
    if depth.ndim == 2:
        depth = depth.reshape(depth.shape[0], depth.shape[1], 1)

    obs = {
        "rgb": rgb,
        "depth": depth,
        "gps": np.array(result.gps, dtype=np.float32),
        "gps_3d": np.array(result.gps_3d, dtype=np.float32),
        "compass": np.array([result.compass], dtype=np.float32),
        "heading": np.array([result.heading], dtype=np.float32),
        "sensor_pose": np.array(result.sensor_pose, dtype=np.float32),
        "camera_pitch": np.array([getattr(result, "camera_pitch", 0.0)], dtype=np.float32),
    }
    if result.num_frontiers > 0:
        obs["frontier_sensor"] = np.array(
            result.frontier_sensor, dtype=np.float32
        ).reshape(result.num_frontiers, 2)
    else:
        obs["frontier_sensor"] = np.zeros((0, 2), dtype=np.float32)
    if len(result.clip_objectgoal) > 0:
        obs["clip_objectgoal"] = np.array(result.clip_objectgoal, dtype=np.float32)
    if result.objectgoal >= 0:
        obs["objectgoal"] = np.array([result.objectgoal], dtype=np.int64)
    if hasattr(result, "step_id"):
        obs["step_id"] = np.array([result.step_id], dtype=np.int64)
    if hasattr(result, "applied_agent_step_id"):
        obs["applied_agent_step_id"] = np.array(
            [result.applied_agent_step_id], dtype=np.int64
        )
    return obs


# Observation message uses the same field structure as StepResult
obs_msg_to_obs_dict = step_result_to_obs_dict


def step_result_to_obs_msg(result, ep_hash: int):
    """Convert StepResult -> ObservationMsg (for seeding RobotAPI)."""
    from ros_msgs.msg import Observation as ObservationMsg

    msg = ObservationMsg()
    msg.step_id = result.step_id
    msg.applied_agent_step_id = 0
    msg.episode_id_hash = ep_hash
    for k in (
        "rgb", "depth", "gps", "gps_3d", "compass", "heading",
        "sensor_pose", "frontier_sensor", "num_frontiers", "clip_objectgoal", "objectgoal",
    ):
        setattr(msg, k, getattr(result, k))
    msg.camera_pitch = float(getattr(result, "camera_pitch", 0.0))
    return msg


# ── Tensor conversion ───────────────────────────────────────────────────


def obs_dict_to_tensor_batch(obs: dict, device) -> dict:
    """Convert observation dict -> batched tensor dict (batch=1) for policy."""
    import torch

    batch = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v).unsqueeze(0)
            if k in ("objectgoal", "step_id"):
                t = t.long()
            elif k != "rgb":
                t = t.float()
            batch[k] = t.to(device)
        else:
            batch[k] = v
    return batch


# ── Internal helpers ─────────────────────────────────────────────────────


def _to_scalar(x) -> float:
    return float(x) if np.ndim(x) == 0 else float(np.asarray(x).flat[0])


def _to_list(x) -> list:
    return x.tolist() if hasattr(x, "tolist") else list(x)


def _safe_json(info: dict) -> dict:
    safe = {}
    for k, v in info.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            if isinstance(v, np.ndarray):
                safe[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                safe[k] = v.item()
    return safe
