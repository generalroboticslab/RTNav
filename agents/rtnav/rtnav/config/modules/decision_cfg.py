from rtnav.config import configclass


@configclass
class DecisionConfig:
    frontier_position_threshold: float = 0.5
    find_sg_centroid_refresh_m: float = 0.3
