from typing import Tuple

from rtnav.config import configclass


@configclass
class CameraConfig:
    """Sim camera intrinsics/extrinsics/depth; PerceptionThread uses it to build CameraFrames."""

    name: str = "cam"  # key for this camera's frames in the sensor dicts

    # intrinsics (build_K): fx=fy=(W/2)/tan(hfov/2), cx=W/2, cy=H/2
    width: int = 0
    height: int = 0
    hfov_deg: float = 0.0

    # extrinsics: optical origin in base frame (forward, right, up), meters
    position_base: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    forward_facing: bool = True

    # depth
    depth_normalized: bool = False  # if [0,1], rescale via sensor_depth_*_m
    sensor_depth_min_m: float = 0.0
    sensor_depth_max_m: float = 10.0
    min_depth: float = 0.3  # valid range (m)
    max_depth: float = 10.0
    depth_hole_area_thresh_px2: int = 100000  # fill holes < this (px²); 0 off
