from rtnav.config import configclass


@configclass
class FrontierConfig:
    # === VLFM detector ===
    vlfm_use_local_height_range: bool = (
        False  # interpret obstacle-height band relative to robot base z (stairs)
    )
    vlfm_min_obstacle_height: float = 0.61  # point-cloud height band counted as obstacles (m)
    vlfm_max_obstacle_height: float = 0.88
    vlfm_floor_drop_height: float = -0.5  # block points >50 cm below the starting floor
    vlfm_area_thresh_m2: float = 1.5  # min adjacent unexplored area to keep a waypoint
    vlfm_update_every_n_steps: int = (
        4  # throttle HabitatObstacleMap.update_map to once per N env steps
    )

    min_explored_threshold: int = 5  # explored-cell count before frontier detection starts
    update_every_n_obs: int = 4  # throttle detector.detect() to once per N map outputs; 1 disables

    # === Strategy (decision/frontier_strategy.py) ===
    strategy_fresh_wait_max_s: float = (
        2.0  # max wait for a FrontierOutput newer than the last reach
    )
    strategy_cycle_grid_m: float = 1.0  # grid cell size (m) for AcyclicEnforcer state-action keys
    strategy_failure_radius_m: float = 1.0  # ban every frontier within this radius of a failed goal
