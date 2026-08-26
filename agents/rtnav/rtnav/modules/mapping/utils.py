import cv2
import numpy as np
from numba import njit


def fog_of_war_local_inplace(
    output_mask: np.ndarray,
    agent_px: tuple,
    agent_yaw: float,
    fov_deg: float,
    max_range_px: float,
    occupancy_map: np.ndarray,
    per_ray_max_px: np.ndarray,
    padding: int = 10,
) -> None:
    """Reveal the FOV wedge into output_mask in-place over a local ROI (avoids full-size
    temporaries on large maps). occupancy_map should be the persistent _obstacle_mask, not
    the per-frame occupancy, so raycasting never flickers through walls."""
    h, w = output_mask.shape
    cx, cy = int(agent_px[0]), int(agent_px[1])

    roi_radius = int(max_range_px + padding)
    x_min = max(0, cx - roi_radius)
    x_max = min(w, cx + roi_radius)
    y_min = max(0, cy - roi_radius)
    y_max = min(h, cy + roi_radius)

    if x_max <= x_min or y_max <= y_min:
        return

    local_cx = cx - x_min
    local_cy = cy - y_min

    local_occupancy = np.ascontiguousarray(occupancy_map[y_min:y_max, x_min:x_max])
    local_h, local_w = local_occupancy.shape

    # Un-dilated occupancy for the explored mask; the dilation below is only for
    # raycasting so the step can't skip through a 1-cell-thick wall.
    occupancy_for_mask = local_occupancy.copy()

    if local_occupancy.any():
        inflate_kernel = np.ones((3, 3), dtype=np.uint8)
        obstacles_for_rays = cv2.dilate(local_occupancy, inflate_kernel)
    else:
        obstacles_for_rays = local_occupancy

    fov_rad = np.radians(fov_deg)
    start_angle = agent_yaw - fov_rad / 2
    end_angle = agent_yaw + fov_rad / 2

    num_rays = int(len(per_ray_max_px))
    max_ranges = np.asarray(per_ray_max_px, dtype=np.float32)
    angles = np.linspace(start_angle, end_angle, num_rays)

    # Cast rays using dilated obstacles so thin walls block correctly.
    visible_polygon = cast_rays_batch(
        local_cx, local_cy, angles, max_ranges, obstacles_for_rays, local_h, local_w
    )

    local_visible = np.zeros((local_h, local_w), dtype=np.uint8)
    cv2.fillPoly(local_visible, [visible_polygon.astype(np.int32)], 1)

    # Exclude actual obstacle cells from explored using un-dilated mask.
    local_visible = cv2.bitwise_and(local_visible, 1 - occupancy_for_mask)

    output_mask[y_min:y_max, x_min:x_max] |= local_visible


@njit
def cast_rays_batch(
    cx: int,
    cy: int,
    angles: np.ndarray,
    max_ranges: np.ndarray,
    obstacles: np.ndarray,
    h: int,
    w: int,
    margin: float = 0.0,
) -> np.ndarray:
    """Cast rays from (cx, cy); each ray's max range is max_ranges[i] — 0 collapses the
    ray to the camera so depth-less bins add nothing to the filled polygon. Returns polygon
    points with the center as the first point."""
    n_rays = len(angles)
    result = np.zeros((n_rays + 1, 2), dtype=np.int32)
    result[0, 0] = cx
    result[0, 1] = cy

    step_size = 1.0

    for i in range(n_rays):
        max_range = max_ranges[i]
        if max_range <= 0.0:
            result[i + 1, 0] = cx
            result[i + 1, 1] = cy
            continue

        angle = angles[i]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        last_valid_x = cx
        last_valid_y = cy

        dist = 0.0
        while dist < max_range:
            x = int(cx + dist * cos_a)
            y = int(cy - dist * sin_a)

            if x < 0 or x >= w or y < 0 or y >= h:
                break

            if obstacles[y, x] > 0:
                safe_dist = max(0.0, dist - margin)
                last_valid_x = int(cx + safe_dist * cos_a)
                last_valid_y = int(cy - safe_dist * sin_a)
                break

            last_valid_x = x
            last_valid_y = y
            dist += step_size

        # Snap endpoint exactly to max_range, not step_size past it.
        # Prevents explored bleeding beyond actual depth boundary.
        if dist >= max_range:
            last_valid_x = int(cx + max_range * cos_a)
            last_valid_y = int(cy - max_range * sin_a)
            last_valid_x = max(0, min(w - 1, last_valid_x))
            last_valid_y = max(0, min(h - 1, last_valid_y))

        result[i + 1, 0] = last_valid_x
        result[i + 1, 1] = last_valid_y

    return result
