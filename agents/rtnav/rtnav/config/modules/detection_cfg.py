from rtnav.config import configclass
from rtnav.modules.perception.detection_policy import (
    DEFAULT_INFERENCE_RES,
    DEFAULT_THRESHOLD,
)


@configclass
class DetectionConfig:
    """OWLv2 open-vocabulary detection settings."""

    threshold: float = DEFAULT_THRESHOLD
    inference_res: int = DEFAULT_INFERENCE_RES  # larger = more accurate, slower
