"""Robot-centered obstacle-map rendering for optional diagnostics."""

import cv2
import numpy as np


def _vlfm_explored_in_view(snapshot, obstacle_map, bounds, shape):
    """Sample the VLFM explored mask into an RTNav-renderer pixel crop."""
    height, width = shape
    x_min, y_min = bounds[:2]
    full_x = x_min + np.arange(width, dtype=np.float64)
    full_y = y_min + np.arange(height, dtype=np.float64)
    world_x = (full_x - float(obstacle_map.origin_px[0])) / float(obstacle_map.ppm)
    world_y = -(full_y - float(obstacle_map.origin_px[1])) / float(obstacle_map.ppm)

    source = np.asarray(snapshot.explored, dtype=bool)
    source_h, source_w = source.shape[:2]
    source_ppm = float(snapshot.pixels_per_meter)
    source_origin = np.asarray(snapshot.episode_pixel_origin, dtype=np.float64)

    # HabitatObstacleMap._xy_to_px swaps XY, then flips its first pixel axis.
    source_x = source_h - (
        np.rint(world_y * source_ppm).astype(np.int64) + int(source_origin[0])
    )
    source_y = np.rint(world_x * source_ppm).astype(np.int64) + int(source_origin[1])
    valid_x = (source_x >= 0) & (source_x < source_w)
    valid_y = (source_y >= 0) & (source_y < source_h)

    clipped_x = np.clip(source_x, 0, max(0, source_w - 1))
    clipped_y = np.clip(source_y, 0, max(0, source_h - 1))
    sampled = source[clipped_y[None, :], clipped_x[:, None]]
    sampled &= valid_x[:, None] & valid_y[None, :]
    return sampled


def render_region(obstacle_map, view_radius_m, vlfm_map_snapshot=None):
    """Render a local map crop and return it with its full-map pixel bounds."""
    view_size = max(400, min(int(view_radius_m * obstacle_map.ppm * 2), 1200))
    robot_px = obstacle_map.xy_to_px(np.array([obstacle_map._last_robot_pose[:2]]))[0]
    rx, ry = map(int, robot_px)
    half = view_size // 2

    x_min, x_max = max(0, rx - half), min(obstacle_map.size, rx + half)
    y_min, y_max = max(0, ry - half), min(obstacle_map.size, ry + half)
    if x_max - x_min < view_size:
        x_min = 0 if x_min == 0 else max(0, obstacle_map.size - view_size)
        x_max = min(obstacle_map.size, x_min + view_size)
    if y_max - y_min < view_size:
        y_min = 0 if y_min == 0 else max(0, obstacle_map.size - view_size)
        y_max = min(obstacle_map.size, y_min + view_size)

    traversability = obstacle_map.traversability[y_min:y_max, x_min:x_max]
    image = np.full((*traversability.shape, 3), 220, dtype=np.uint8)
    # Gray always means VLFM-explored; leave it blank until that map exists.
    if vlfm_map_snapshot is not None:
        explored = _vlfm_explored_in_view(
            vlfm_map_snapshot,
            obstacle_map,
            (x_min, y_min, x_max, y_max),
            traversability.shape,
        )
        image[explored] = (170, 170, 170)
    image[traversability < 0.1] = (0, 0, 0)

    scale_px = int(5.0 * obstacle_map.ppm)
    bar_y = image.shape[0] - 15
    cv2.line(image, (10, bar_y), (10 + scale_px, bar_y), (0, 0, 0), 2)
    cv2.putText(image, "5m", (12, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return image, (x_min, y_min, x_max, y_max)


def world_to_view_px(obstacle_map, xy, bounds):
    """Convert world coordinates to pixels in a rendered local crop."""
    pixel = obstacle_map.xy_to_px(np.asarray(xy, dtype=float).reshape(1, 2))[0]
    return int(pixel[0] - bounds[0]), int(pixel[1] - bounds[1])
