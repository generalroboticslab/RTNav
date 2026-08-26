"""C++-backed 3D detection projection.

Projects per-camera 2D detections into 3D ``ObjectDetection`` objects via the
compiled perception C++ extension.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from rtnav.modules.scenegraph.cpp_accel import (
    project_detections_to_3d as project_detections_to_3d_cpp,
)
from rtnav.modules.scenegraph.scene_graph import ObjectDetection


def _rescale_intrinsics(
    K: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]
) -> np.ndarray:
    """Scale a 3x3 intrinsics matrix from src to dst resolution."""
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    if src_h == dst_h and src_w == dst_w:
        return K
    K_scaled = K.copy()
    sx = dst_w / src_w
    sy = dst_h / src_h
    K_scaled[0, 0] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[0, 2] *= sx
    K_scaled[1, 2] *= sy
    return K_scaled


def _sanitize_depth(depth: np.ndarray, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    eps = 1e-3
    invalid_depth = (
        ~np.isfinite(depth) | (depth <= max(min_depth_m, 0.0) + eps) | (depth >= max_depth_m - eps)
    )
    if not np.any(invalid_depth):
        return depth
    depth = np.array(depth, copy=True)
    depth[invalid_depth] = 0.0
    return depth


def _downsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, colors
    indices = np.random.choice(len(points), int(max_points), replace=False)
    return points[indices], colors[indices]


def _open3d_dbscan_filtering(
    points: np.ndarray,
    colors: np.ndarray,
    eps: float = 0.2,
    min_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "[det3d] MobileSAM mask projection requires open3d for VLFM-style DBSCAN filtering"
        ) from exc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.array(pcd.cluster_dbscan(eps, min_points))
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    non_noise = unique_labels != -1
    if not np.any(non_noise):
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)

    kept_labels = unique_labels[non_noise]
    kept_counts = label_counts[non_noise]
    largest_label = kept_labels[np.argmax(kept_counts)]
    keep = labels == largest_label
    return points[keep], colors[keep]


def project_mask_detection_to_3d(
    cam: Any,
    det: Any,
    object_mask: np.ndarray,
    erosion_size: int = 5,
    max_points: int = 5000,
    use_dbscan: bool = True,
    failure_context: str = "",
) -> Optional[ObjectDetection]:
    """Project a SAM object mask to a VLFM-style cleaned 3D point cloud."""

    def _fail(reason: str) -> None:
        label = str(getattr(det, "name", "object"))
        bbox = tuple(float(v) for v in getattr(det, "bbox", ()))
        context = f" {failure_context}" if failure_context else ""
        print(
            f"[det3d] mask projection failed{context}: label='{label}' bbox={bbox} reason={reason}"
        )

    rgb = cam.rgb_image
    depth = cam.depth_image
    if rgb is None or depth is None:
        _fail(f"missing {'rgb' if rgb is None else 'depth'} image")
        return None
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if object_mask is None:
        _fail("missing object mask")
        return None
    if object_mask.ndim != 2:
        raise ValueError(f"[det3d] object mask must be 2D, got shape {object_mask.shape}")

    min_depth_m = float(getattr(cam, "min_depth_m", 0.0))
    max_depth_m = float(getattr(cam, "max_depth_m", 10.0))
    depth = _sanitize_depth(depth, min_depth_m, max_depth_m)

    H_rgb, W_rgb = rgb.shape[:2]
    dH, dW = depth.shape[:2]
    mask_h, mask_w = object_mask.shape[:2]
    mask_u8 = (object_mask > 0).astype(np.uint8)
    if (mask_h, mask_w) == (dH, dW):
        mask_depth = mask_u8
    elif (mask_h, mask_w) == (H_rgb, W_rgb):
        mask_depth = cv2.resize(mask_u8, (dW, dH), interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError(
            "[det3d] object mask shape must match RGB or depth shape; "
            f"got mask={object_mask.shape}, rgb={rgb.shape[:2]}, depth={depth.shape[:2]}"
        )

    final_mask = mask_depth * 255
    final_mask = cv2.erode(final_mask, None, iterations=int(erosion_size))  # type: ignore
    mask_depth = final_mask > 0
    if not np.any(mask_depth):
        raw_count = int(np.count_nonzero(mask_u8))
        _fail(
            f"mask empty after erosion; raw_mask_pixels={raw_count} "
            f"erosion_size={int(erosion_size)}"
        )
        return None

    _K = getattr(cam, "intrinsics_depth", None)
    if dH != H_rgb or dW != W_rgb:
        intrinsics = (
            _K if _K is not None else _rescale_intrinsics(cam.intrinsics, (H_rgb, W_rgb), (dH, dW))
        )
        rgb_small = cv2.resize(rgb, (dW, dH), interpolation=cv2.INTER_AREA)
    else:
        intrinsics = _K if _K is not None else cam.intrinsics
        rgb_small = rgb
    intrinsics = np.asarray(intrinsics, dtype=float)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("[det3d] camera intrinsics must be a finite 3x3 matrix")

    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if abs(fx) < 1e-9 or abs(fy) < 1e-9:
        raise ValueError("[det3d] camera intrinsics have zero focal length")

    ys, xs = np.where(mask_depth)
    zs = depth[ys, xs].astype(float)
    valid = np.isfinite(zs) & (zs > 0.0)
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 10:
        _fail(
            f"too few valid depth pixels after erosion; "
            f"valid_depth_pixels={valid_count}/{len(zs)} min_depth_m={min_depth_m:.3f} "
            f"max_depth_m={max_depth_m:.3f}"
        )
        return None
    xs = xs[valid].astype(float)
    ys = ys[valid].astype(float)
    zs = zs[valid]

    x_cam = (xs - cx) / fx * zs
    y_cam = (ys - cy) / fy * zs
    points_cam = np.stack((x_cam, y_cam, zs), axis=1)
    R = np.asarray(cam.T_world_camera[:3, :3], dtype=float)
    t = np.asarray(cam.T_world_camera[:3, 3], dtype=float)
    points_world = points_cam @ R.T + t[None, :]
    colors = rgb_small[ys.astype(int), xs.astype(int)]

    finite = np.isfinite(points_world).all(axis=1)
    points_world = points_world[finite]
    colors = colors[finite]
    if len(points_world) < 10:
        _fail(f"too few finite world points; finite_points={len(points_world)}/{len(finite)}")
        return None

    points_world, colors = _downsample_points(points_world, colors, max_points)
    if use_dbscan:
        pre_dbscan_count = len(points_world)
        points_world, colors = _open3d_dbscan_filtering(points_world, colors)
        if len(points_world) == 0:
            _fail(f"DBSCAN removed all points; input_points={pre_dbscan_count}")
            return None

    centroid = np.mean(points_world, axis=0)
    bbox_mins = np.min(points_world, axis=0)
    bbox_maxs = np.max(points_world, axis=0)

    return ObjectDetection(
        label=det.name,
        bbox_3d=np.array([bbox_mins, bbox_maxs]),
        centroid=centroid,
        point_cloud=points_world,
        colors=colors.astype(np.uint8),
        owl_features=det.embeddings,
        confidence=det.confidence,
        robot_distance=float(np.linalg.norm(centroid - t)),
        observed_from_xy=(float(t[0]), float(t[1])),
        observed_from_yaw=float(np.arctan2(R[1, 2], R[0, 2])),
        top_k_label_probs=getattr(det, "top_k_label_probs", None),
    )


def _process_camera_cpp(
    cam: Any,
    detections: List[Any],
) -> List[ObjectDetection]:
    """Process a single camera using C++ acceleration.

    OPTIMIZATION: Works at depth's native resolution (e.g. 128x78) instead of
    upscaling to RGB resolution.  Bboxes are scaled from RGB to depth coords
    before passing to C++.  Colors come from a small RGB downsampled to depth
    resolution.
    """
    rgb = cam.rgb_image
    depth = cam.depth_image
    min_depth_m = float(getattr(cam, "min_depth_m", 0.0))
    max_depth_m = float(getattr(cam, "max_depth_m", 10.0))
    if depth is not None:
        # Reject non-finite and both clip endpoints (max = no-return sentinel,
        # e.g. windows/glass behind curtains) so objects aren't placed at 5m.
        eps = 1e-3
        invalid_depth = (
            ~np.isfinite(depth)
            | (depth <= max(min_depth_m, 0.0) + eps)
            | (depth >= max_depth_m - eps)
        )
        if np.any(invalid_depth):
            depth = np.array(depth, copy=True)
            depth[invalid_depth] = 0.0
    H_rgb, W_rgb = rgb.shape[:2]
    dH, dW = depth.shape[:2]

    _K = getattr(cam, "intrinsics_depth", None)

    if dH != H_rgb or dW != W_rgb:
        if _K is not None:
            intrinsics = _K
        else:
            intrinsics = _rescale_intrinsics(cam.intrinsics, (H_rgb, W_rgb), (dH, dW))
        sx = dW / W_rgb
        sy = dH / H_rgb
        rgb_small = cv2.resize(rgb, (dW, dH), interpolation=cv2.INTER_AREA)
    else:
        intrinsics = _K if _K is not None else cam.intrinsics
        sx, sy = 1.0, 1.0
        rgb_small = rgb

    # Scale bboxes from RGB coords to depth coords.
    raw_bboxes = np.array([det.bbox for det in detections], dtype=np.float64)
    scaled_bboxes = raw_bboxes.copy()
    scaled_bboxes[:, 0] *= sx
    scaled_bboxes[:, 2] *= sx
    scaled_bboxes[:, 1] *= sy
    scaled_bboxes[:, 3] *= sy

    R = cam.T_world_camera[:3, :3]
    t = cam.T_world_camera[:3, 3]

    valid_mask, centroids, bbox_mins, bbox_maxs, point_clouds, colors_list = (
        project_detections_to_3d_cpp(
            depth=depth,
            rgb=rgb_small,
            bboxes=scaled_bboxes,
            intrinsics=intrinsics,
            rotation=R,
            translation=t,
            max_points=500,
        )
    )

    results = []
    for i, det in enumerate(detections):
        if not valid_mask[i]:
            continue

        results.append(
            ObjectDetection(
                label=det.name,
                bbox_3d=np.array([bbox_mins[i], bbox_maxs[i]]),
                centroid=centroids[i],
                point_cloud=point_clouds[i],
                colors=colors_list[i],
                owl_features=det.embeddings,
                confidence=det.confidence,
                robot_distance=float(np.linalg.norm(centroids[i] - t)),
                observed_from_xy=(float(t[0]), float(t[1])),
                # Viewing heading = world yaw of the camera's forward (+Z optical) axis.
                observed_from_yaw=float(np.arctan2(R[1, 2], R[0, 2])),
                top_k_label_probs=getattr(det, "top_k_label_probs", None),
            )
        )

    return results


def process_all_cameras_parallel(camera_results: Dict[str, Any]) -> List[ObjectDetection]:
    """Project all cameras' 2D detections to 3D via the C++ backend."""
    import time as _time

    t0 = _time.perf_counter()
    all_results = []
    total_dets = 0

    for cam_name, cam in camera_results.items():
        detections = cam.detections
        if not detections:
            continue

        total_dets += len(detections)

        # Cap detections per camera by confidence to bound processing time.
        if len(detections) > 30:
            detections = sorted(detections, key=lambda d: d.confidence, reverse=True)[:30]

        results = _process_camera_cpp(cam, detections)

        all_results.extend(results)

    # Filter out detections with NaN/inf centroids.
    valid_results = []
    for det in all_results:
        if det.centroid is not None and np.isfinite(det.centroid).all():
            valid_results.append(det)

    t1 = _time.perf_counter()
    det3d_ms = (t1 - t0) * 1000

    if len(valid_results) < len(all_results):
        print(
            f"[det3d] Filtered {len(all_results) - len(valid_results)} detections with invalid centroids"
        )

    if det3d_ms > 50:
        print(
            f"[det3d] {det3d_ms:.0f}ms for {len(valid_results)} valid / {total_dets} raw across {len(camera_results)} cameras"
        )

    return valid_results
