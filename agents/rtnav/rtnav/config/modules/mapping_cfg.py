from rtnav.config import configclass


@configclass
class MappingConfig:
    # === Grid settings ===
    obstacle_map_size: int = 40000  # pixels (40000 @ 20ppm = 2km × 2km)
    pixels_per_meter: float = 20.0  # 5cm resolution

    # === Terrain analysis (TerrainAnalyzer) ===
    slope_threshold: float = 0.3
    safety_margin: float = 0.3  # buffer zone around obstacles (meters)
    erode_boundary: bool = False  # erode valid mask edges (outdoor)
    # Force cells within this many pixels of the depth-coverage edge to safe,
    # suppressing the fake slope spikes the gaussian-smoothed fill creates there.
    boundary_safe_band_px: int = 2
    # Thin-wall rescue: cells at height >= robot_z + this are force-marked
    # non-navigable (1-2 px walls don't give the analyzer enough neighbors for
    # slope), applied only within wall_height_radius_m of the robot.
    wall_height_threshold_m: float = 0.3
    wall_height_radius_m: float = 3.0

    # === Height range of integrated points ===
    obstacle_height_min: float = float("-inf")
    obstacle_height_max: float = float("inf")
    # When True, obstacle_height_min/max are relative to robot base; else world Z.
    use_local_height_range: bool = False
