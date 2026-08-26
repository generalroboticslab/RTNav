"""Navigable goal search for the obstacle map."""

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from rtnav.modules.mapping.obstacle_map.obstacle_map import ObstacleMap


def find_navigable_goal(
    obs_map: "ObstacleMap",
    goal_xy: Tuple[float, float],
    search_radius_m: float = 2.0,
    navigable: Optional[np.ndarray] = None,
) -> Optional[Tuple[Tuple[float, float], np.ndarray]]:
    """Closest navigable cell within ``search_radius_m``, or ``None``."""
    goal_px = obs_map.xy_to_px(np.array([goal_xy]))[0]
    gx, gy = int(goal_px[0]), int(goal_px[1])
    navigable = (
        np.asarray(obs_map.navigable, dtype=bool)
        if navigable is None else np.asarray(navigable, dtype=bool)
    )

    height, width = navigable.shape
    if 0 <= gx < width and 0 <= gy < height and navigable[gy, gx]:
        return tuple(goal_xy), goal_px

    radius = max(0, int(search_radius_m * obs_map.ppm))
    x0, x1 = max(0, gx - radius), min(width, gx + radius + 1)
    y0, y1 = max(0, gy - radius), min(height, gy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    ys, xs = np.nonzero(navigable[y0:y1, x0:x1])
    if len(xs) == 0:
        return None

    xs += x0
    ys += y0
    distances2 = (xs - gx) ** 2 + (ys - gy) ** 2
    inside = distances2 <= radius**2
    if not np.any(inside):
        return None
    xs, ys, distances2 = xs[inside], ys[inside], distances2[inside]
    best = int(np.argmin(distances2))
    best_px = np.array([xs[best], ys[best]])
    best_xy = obs_map.px_to_xy(np.array([best_px]))[0]
    return tuple(best_xy), best_px


def find_navigable_frontier_goal(
    obs_map: "ObstacleMap",
    centroid_px: np.ndarray,
) -> Optional[Tuple[Tuple[float, float], np.ndarray]]:
    """Snap a VLFM frontier to the nearest RTNav-navigable approach cell."""
    cx, cy = int(centroid_px[0]), int(centroid_px[1])
    if cx < 0 or cx >= obs_map.size or cy < 0 or cy >= obs_map.size:
        return None

    centroid_xy = obs_map.px_to_xy(np.array([[cx, cy]]))[0]
    return find_navigable_goal(obs_map, tuple(centroid_xy), search_radius_m=2.0)


def is_goal_blocked(obs_map: "ObstacleMap", goal_xy: Tuple[float, float]) -> bool:
    """Whether the exact active goal has become a hard obstacle."""
    px = obs_map.xy_to_px(np.asarray([goal_xy], dtype=float))[0]
    x, y = int(px[0]), int(px[1])
    blocked = getattr(obs_map, "_combined_blocked", None)
    if blocked is not None:
        if not (0 <= x < blocked.shape[1] and 0 <= y < blocked.shape[0]):
            return True
        return bool(blocked[y, x])
    return not obs_map.is_navigable_px(x, y)
