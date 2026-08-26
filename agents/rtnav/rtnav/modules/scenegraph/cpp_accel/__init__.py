"""C++ acceleration for scene-graph hot paths.

Imports the compiled ``sg_accel`` extension; raises ImportError if not built.
Run ``agents/rtnav/scripts/verify_cpp_accel.py`` to verify the install.
"""

from typing import List, Tuple

import numpy as np

from . import sg_accel as _sg_accel


def project_detections_to_3d(
    depth: np.ndarray,
    rgb: np.ndarray,
    bboxes: np.ndarray,
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    max_points: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """C++ 2D→3D detection projection. Returns
    (valid_mask, centroids, bbox_mins, bbox_maxs, point_clouds, colors)."""
    return _sg_accel.project_detections_to_3d(
        depth.astype(np.float32),
        rgb.astype(np.uint8),
        bboxes.astype(np.float64),
        intrinsics.astype(np.float64),
        rotation.astype(np.float64),
        translation.astype(np.float64),
        int(max_points),
    )


def score_candidates_cpp(
    det_indices: np.ndarray,
    node_indices: np.ndarray,
    det_bboxes: np.ndarray,
    det_features: np.ndarray,
    det_labels: np.ndarray,
    det_centroids: np.ndarray,
    node_bboxes: np.ndarray,
    node_features: np.ndarray,
    node_labels: np.ndarray,
    node_centroids: np.ndarray,
) -> np.ndarray:
    return _sg_accel.score_candidates(
        det_indices.astype(np.int32),
        node_indices.astype(np.int32),
        det_bboxes.astype(np.float64),
        det_features.astype(np.float64),
        det_labels.astype(np.int32),
        det_centroids.astype(np.float64),
        node_bboxes.astype(np.float64),
        node_features.astype(np.float64),
        node_labels.astype(np.int32),
        node_centroids.astype(np.float64),
    )


def find_merge_candidates_cpp(
    node_centroids: np.ndarray,
    node_labels: np.ndarray,
    max_distance: float = 3.0,
) -> np.ndarray:
    return _sg_accel.find_merge_candidates(
        node_centroids.astype(np.float64),
        node_labels.astype(np.int32),
        float(max_distance),
    )


def batch_bbox_iou_containment_cpp(all_bboxes: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return _sg_accel.batch_bbox_iou_containment(
        all_bboxes.astype(np.float64),
        pairs.astype(np.int32),
    )


def batch_voxel_iou_cpp(all_voxel_keys: List[np.ndarray], pairs: np.ndarray) -> np.ndarray:
    keys_list = [k.astype(np.int32) for k in all_voxel_keys]
    return _sg_accel.batch_voxel_iou(keys_list, pairs.astype(np.int32))
