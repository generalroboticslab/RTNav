"""Camera intrinsics, extrinsics, and depth unprojection — plain geometry utils.

Frames (VLFM habitat-sim chain): world/base = X-forward, Y-left, Z-up,
yaw = +compass; camera = OpenCV optical (X-right, Y-down, Z-forward).
"""

from typing import Optional

import numpy as np


def build_K(hfov_deg: float, width: int, height: int) -> np.ndarray:
    """3x3 pinhole intrinsics for a square-pixel camera (fx=fy)."""
    W, H = int(width), int(height)
    fx = (W / 2.0) / np.tan(np.radians(float(hfov_deg) / 2.0))
    return np.array(
        [[fx, 0.0, W / 2.0], [0.0, fx, H / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


# OpenCV-optical → base axes for a forward-facing camera:
# cam (right, down, forward) → base (-Y, -Z, +X).
_R_BASE_CAM_FORWARD = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def build_T_base_cam(position_base, forward_facing: bool = True) -> np.ndarray:
    """4x4 camera-optical → base transform for a forward-facing camera at
    position_base (forward, right, up) in the base frame."""
    if not forward_facing:
        raise NotImplementedError("Non-forward-facing cameras need an explicit rotation")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _R_BASE_CAM_FORWARD
    T[:3, 3] = np.asarray(position_base, dtype=np.float64)
    return T


def decode_depth_to_meters(
    depth_raw: np.ndarray,
    normalized: bool = False,
    sensor_min_m: float = 0.0,
    sensor_max_m: float = 10.0,
) -> np.ndarray:
    """Decode depth to meters: normalized [0,1] → [sensor_min_m, sensor_max_m]
    (habitat default), else values are already meters."""
    d = depth_raw.astype(np.float32)
    if normalized:
        d = d * (float(sensor_max_m) - float(sensor_min_m)) + float(sensor_min_m)
    return d


def build_T_world_base(gps, compass) -> np.ndarray:
    """HM3D (gps, compass) → 4x4 world-from-base transform (yaw = +compass)."""
    yaw = float(compass)
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4, dtype=np.float64)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    T[0, 3] = float(gps[0])
    T[1, 3] = -float(gps[1])  # habitat GPS makes +right positive; we want +left
    return T


def depth_to_pointcloud_world(
    depth_m: np.ndarray,
    T_world_cam: np.ndarray,
    K: np.ndarray,
    min_depth: float,
    max_depth: float,
    rgb: Optional[np.ndarray] = None,
):
    """Back-project a metric depth image (OpenCV-optical) to world points.

    Valid range (min_depth, max_depth) is exclusive by an epsilon: habitat clips
    out-of-range rays to the endpoints, so exact min/max are phantom sentinels.
    Returns pts_world (N,3) float32, + colors (N,3) in [0,1] if rgb given.
    """
    min_m = float(min_depth)
    max_m = float(max_depth)

    eps = 1e-3
    lower = max(min_m, 0.0) + eps
    upper = max_m - eps
    mask = np.isfinite(depth_m) & (depth_m > lower) & (depth_m < upper)
    vs, us = np.where(mask)
    if us.size == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return (empty, empty) if rgb is not None else empty

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth_m[vs, us].astype(np.float64)
    x_cam = (us - cx) * z / fx
    y_cam = (vs - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)
    pts_world = (T_world_cam @ pts_cam.T).T[:, :3].astype(np.float32)
    if rgb is None:
        return pts_world

    if rgb.shape[:2] != depth_m.shape:
        import cv2

        rgb = cv2.resize(rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)
    colors = rgb[vs, us].astype(np.float32) / 255.0  # RGB 0-1
    return pts_world, colors
