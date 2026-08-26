"""Habitat Sim (HM3D ObjectNav) environment config."""

from dataclasses import field
from typing import Tuple

from rtnav.config import configclass
from rtnav.config.modules.camera_cfg import CameraConfig
from rtnav.config.modules.decision_cfg import DecisionConfig
from rtnav.config.modules.detection_cfg import DetectionConfig
from rtnav.config.modules.frontier_cfg import FrontierConfig
from rtnav.config.modules.mapping_cfg import MappingConfig
from rtnav.config.modules.scenegraph_cfg import SceneGraphConfig


@configclass
class HM3DMappingConfig(MappingConfig):
    obstacle_height_min: float = float("-inf")
    obstacle_height_max: float = 1.8  # cap on world-Z (rel. base) into the height map
    obstacle_map_size: int = 1600  # ~53m @ 30ppm, centered on spawn
    pixels_per_meter: float = 30.0
    use_local_height_range: bool = True  # height filter relative to robot base
    slope_threshold: float = 10.0  # only near-vertical jumps flag as obstacle
    wall_height_threshold_m: float = 0.3  # above robot base z, frame-local
    safety_margin: float = 0.2  # terrain-analysis inflation (the only inflation stage)


@configclass
class HM3DSceneGraphConfig(SceneGraphConfig):
    min_confirmation_views: int = 3  # HM3D is packed; reduce chair/couch FPs


@configclass
class HM3DCameraConfig(CameraConfig):
    # Habitat sim camera (objectnav_hm3d.yaml: 640x480 hfov=79, depth 0.5-5m, sensor z=0.88).
    name: str = "hm3d"
    width: int = 640
    height: int = 480
    hfov_deg: float = 79.0
    position_base: Tuple[float, float, float] = (0.0, 0.0, 0.88)
    forward_facing: bool = True
    depth_normalized: bool = True
    sensor_depth_min_m: float = 0.5
    sensor_depth_max_m: float = 5.0
    min_depth: float = 0.5
    max_depth: float = 5.0


@configclass
class HM3DConfig:
    """Habitat Sim environment configuration."""

    env_name: str = "habitat-sim"
    use_sim_time: bool = True

    mapping: HM3DMappingConfig = field(default_factory=HM3DMappingConfig)
    scenegraph: HM3DSceneGraphConfig = field(default_factory=HM3DSceneGraphConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    camera: HM3DCameraConfig = field(default_factory=HM3DCameraConfig)
    frontier: FrontierConfig = field(default_factory=FrontierConfig)
