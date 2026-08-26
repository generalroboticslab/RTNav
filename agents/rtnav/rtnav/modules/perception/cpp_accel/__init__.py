"""C++ acceleration for perception: detection post-processing and depth hole-filling."""

from typing import Tuple

import numpy as np

from . import perception_accel as _perception_accel


def fill_small_holes(depth: np.ndarray, area_thresh: int) -> np.ndarray:
    """C++ fill of zero-holes smaller than area_thresh (px²) in a depth image."""
    return _perception_accel.fill_small_holes(depth.astype(np.float32), int(area_thresh))


def batch_postprocess_detections(
    bboxes: np.ndarray,
    rot_k: int,
    img_w: int,
    img_h: int,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    T_world_cam: np.ndarray,
    min_depth_m: float = 0.0,
    max_depth_m: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """C++ batch post-processing of detections. Returns
    (adjusted_bboxes (N,4), valid_mask (N,), world_xy (N,2), has_world (N,)).

    Depths of exactly min_depth_m / max_depth_m are habitat clip sentinels, not
    real hits — excluded with an epsilon to avoid phantom targets at those radii.
    """
    return _perception_accel.batch_postprocess_detections(
        bboxes.astype(np.float64),
        int(rot_k),
        int(img_w),
        int(img_h),
        depth.astype(np.float32),
        intrinsics.astype(np.float64),
        T_world_cam.astype(np.float64),
        float(min_depth_m),
        float(max_depth_m),
    )
