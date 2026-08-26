"""OVON Habitat configuration.

Separate from HM3D ObjectNav on purpose: the OVON benchmark in this repo uses
the articulated Fetch/Stretch-style camera stack from the OVON Hydra configs,
not the LoCoBot camera used by Habitat's objectnav_hm3d benchmark.
"""

from dataclasses import field
from typing import Tuple

from rtnav.config import configclass
from rtnav.config.environments.hm3d_config import (
    HM3DMappingConfig,
    HM3DSceneGraphConfig,
)
from rtnav.config.modules.camera_cfg import CameraConfig
from rtnav.config.modules.decision_cfg import DecisionConfig
from rtnav.config.modules.detection_cfg import DetectionConfig
from rtnav.config.modules.frontier_cfg import FrontierConfig


@configclass
class OVONCameraConfig(CameraConfig):
    """OVON camera (from the OVON Hydra output + baseline configs):
    360x640, hfov 42°, height 1.31 m, sim depth normalized 0.5–5.0 m."""

    name: str = "ovon"
    width: int = 360
    height: int = 640
    hfov_deg: float = 42.0
    position_base: Tuple[float, float, float] = (0.0, 0.0, 1.31)
    forward_facing: bool = True
    depth_normalized: bool = True
    sensor_depth_min_m: float = 0.5
    sensor_depth_max_m: float = 5.0
    min_depth: float = 0.5
    max_depth: float = 5.0


@configclass
class OVONConfig:
    """OVON environment configuration."""

    env_name: str = "ovon"
    use_sim_time: bool = True

    mapping: HM3DMappingConfig = field(default_factory=HM3DMappingConfig)
    scenegraph: HM3DSceneGraphConfig = field(default_factory=HM3DSceneGraphConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    camera: OVONCameraConfig = field(default_factory=OVONCameraConfig)
    frontier: FrontierConfig = field(default_factory=FrontierConfig)
