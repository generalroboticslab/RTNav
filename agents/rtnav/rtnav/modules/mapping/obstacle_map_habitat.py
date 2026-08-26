"""VLFM's depth-built obstacle map with RTNav's same-floor safety rule.

The mapping and frontier extraction otherwise track VLFM's ObstacleMap. Points below
``floor_drop_height`` in episode z are additionally treated as obstacles so descending
stairs do not become exploration candidates.
"""

from __future__ import annotations

from typing import Any, Union

import cv2
import numpy as np
from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

# HabitatObstacleMap runs the pure-Python path so its behavior tracks the
# VLFM reference 1:1. The C++ accel module is still used by rtnav's planner
# ObstacleMap, just not here.


def fill_small_holes(depth_img: np.ndarray, area_thresh: int) -> np.ndarray:
    """VLFM ``vlfm/utils/img_utils.py::fill_small_holes`` (verbatim).

    Holes (``depth == 0``) with contour area ``< area_thresh`` are filled
    with 1; larger holes stay 0.
    """
    binary_img = np.where(depth_img == 0, 1, 0).astype("uint8")
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled_holes = np.zeros_like(binary_img)
    for cnt in contours:
        if cv2.contourArea(cnt) < area_thresh:
            cv2.drawContours(filled_holes, [cnt], 0, 1, -1)
    return np.where(filled_holes == 1, 1, depth_img)


def get_point_cloud(depth_image: np.ndarray, mask: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """VLFM ``vlfm/utils/geometry_utils.py::get_point_cloud`` (verbatim).

    Depth pixels → ``(forward, left, up)`` body-frame XYZ via depth + fx + fy.
    """
    v, u = np.where(mask)
    z = depth_image[v, u]
    x = (u - depth_image.shape[1] / 2) * z / fx
    y = (v - depth_image.shape[0] / 2) * z / fy
    return np.stack((z, -x, -y), axis=-1)


def transform_points(transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """VLFM ``vlfm/utils/geometry_utils.py::transform_points`` (verbatim)."""
    homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed = (transformation_matrix @ homogeneous.T).T
    return transformed[:, :3] / transformed[:, 3:]


def extract_yaw(matrix: np.ndarray) -> float:
    """VLFM ``vlfm/utils/geometry_utils.py::extract_yaw`` (verbatim)."""
    assert matrix.shape == (4, 4), "The input matrix must be 4x4"
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def filter_obstacle_points(
    points: np.ndarray,
    min_h: float,
    max_h: float,
    floor_drop_h: float,
) -> np.ndarray:
    heights = points[:, 2]
    body_obstacles = (heights >= min_h) & (heights <= max_h)
    lower_floor = heights < floor_drop_h
    return points[body_obstacles | lower_floor]


class HabitatObstacleMap:
    """Port of VLFM ``vlfm.mapping.obstacle_map.ObstacleMap`` for rtnav.

    Owns its own ``_map`` / ``_navigable_map`` / ``explored_area`` arrays
    and exposes ``frontiers`` (world XY, shape ``(N, 2)``) and
    ``_frontiers_px`` (pixel coords in *this* map's frame).

    Sized in pixels around an origin at ``(size//2, size//2)``; world XY is
    converted through :meth:`_xy_to_px` (same convention as VLFM's BaseMap).
    """

    _map_dtype: np.dtype = np.dtype(bool)

    def __init__(
        self,
        min_height: float = 0.15,
        max_height: float = 0.88,
        floor_drop_height: float = -0.2,
        agent_radius: float = 0.18,
        area_thresh: float = 1.5,  # m²
        hole_area_thresh: int = 100000,  # px²
        size: int = 1000,
        pixels_per_meter: int = 20,
        use_local_height_range: bool = False,
    ):
        self.pixels_per_meter = pixels_per_meter
        self.size = size
        self._episode_pixel_origin = np.array([size // 2, size // 2])

        self._map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)
        self.explored_area = np.zeros((size, size), dtype=bool)

        self._min_height = float(min_height)
        self._max_height = float(max_height)
        self._floor_drop_height = float(floor_drop_height)
        # When True, ``update_map`` interprets ``min_height`` / ``max_height``
        # as offsets from the robot's current base z (extracted from the
        # tf_camera_to_episodic translation). This lets the obstacle map
        # follow the agent up/down stairs — points that count as obstacles
        # on floor 1 stay relative to whichever floor the agent stands on,
        # instead of staying anchored to z=0 of the starting floor.
        #
        # CAVEAT: depends on tf_camera_to_episodic actually reflecting the
        # robot's true world elevation. With the current pipeline
        # (build_T_world_base sets z=0, then vlfm_frontier_detector
        # overrides tf[2,3] to camera_height) the inferred base z is always
        # 0, so the flag's behavior degrades to the absolute-height filter.
        # To get real multi-floor behavior, plumb habitat's agent_state.
        # position[1] (y-up in habitat = world z) through obs.gps (3D) and
        # use it when building T_world_base.
        self._use_local_height_range = bool(use_local_height_range)
        self._area_thresh_in_pixels = int(area_thresh * (pixels_per_meter**2))
        self._hole_area_thresh = int(hole_area_thresh)

        kernel_size = pixels_per_meter * agent_radius * 2
        # round up to the nearest odd number
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)

        self._frontiers_px = np.array([])
        self.frontier_unexplored_directions = []
        self.frontiers = np.array([])

    def reset(self) -> None:
        self._map.fill(0)
        self._navigable_map.fill(0)
        self.explored_area.fill(0)
        self._frontiers_px = np.array([])
        self.frontier_unexplored_directions = []
        self.frontiers = np.array([])

    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        """VLFM BaseMap ``_xy_to_px`` (verbatim)."""
        px = np.rint(points[:, ::-1] * self.pixels_per_meter) + self._episode_pixel_origin
        px[:, 0] = self._map.shape[0] - px[:, 0]
        return px.astype(int)

    def _px_to_xy(self, px: np.ndarray) -> np.ndarray:
        """VLFM BaseMap ``_px_to_xy`` (verbatim)."""
        px_copy = px.copy()
        px_copy[:, 0] = self._map.shape[0] - px_copy[:, 0]
        points = (px_copy - self._episode_pixel_origin) / self.pixels_per_meter
        return points[:, ::-1]

    def update_map(
        self,
        depth: Union[np.ndarray, Any],
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        """Port of ``vlfm.mapping.obstacle_map.ObstacleMap.update_map``.

        Args:
            depth: ``(H, W)`` depth image normalized to ``[0, 1]``. Pixels
                with value 0 are treated as depth holes.
            tf_camera_to_episodic: 4×4 world-from-camera matrix. For
                Habitat-style forward-facing cameras this is the agent
                pose with the camera height in the z translation.
            min_depth/max_depth: depth normalization range (m).
            fx/fy: depth-camera focal lengths in pixels.
            topdown_fov: horizontal FOV (rad) projected onto the topdown
                map for the fog-of-war sweep.
        """
        # When use_local_height_range is set, treat self._{min,max}_height as
        # offsets from the camera-frame translation's z (i.e. the agent base
        # height plus camera height). This shifts the obstacle band up/down
        # with the agent so points on the floor the agent currently stands
        # on count as obstacles instead of being clipped because they sit
        # above the starting floor's [min_height, max_height] window.
        if self._use_local_height_range:
            base_z = float(tf_camera_to_episodic[2, 3])
            eff_min = self._min_height + base_z
            eff_max = self._max_height + base_z
        else:
            eff_min = self._min_height
            eff_max = self._max_height

        if update_obstacles:
            if self._hole_area_thresh == -1:
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0
            else:
                filled_depth = fill_small_holes(depth, self._hole_area_thresh)
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
            mask = scaled_depth < max_depth
            point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
            point_cloud_episodic_frame = transform_points(
                tf_camera_to_episodic, point_cloud_camera_frame
            )
            obstacle_cloud = filter_obstacle_points(
                point_cloud_episodic_frame,
                eff_min,
                eff_max,
                self._floor_drop_height,
            )

            xy_points = obstacle_cloud[:, :2]
            pixel_points = self._xy_to_px(xy_points)
            in_bounds = (
                (pixel_points[:, 0] >= 0)
                & (pixel_points[:, 0] < self._map.shape[1])
                & (pixel_points[:, 1] >= 0)
                & (pixel_points[:, 1] < self._map.shape[0])
            )
            pixel_points = pixel_points[in_bounds]
            self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1

            self._navigable_map = 1 - cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1,
            ).astype(bool)

        if not explore:
            return

        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]

        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        )
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
        self.explored_area[new_explored_area > 0] = 1
        self.explored_area[self._navigable_map == 0] = 0

        # Keep only the connected component containing the agent (matches VLFM)
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) > 1:
            min_dist = np.inf
            best_idx = 0
            for idx, cnt in enumerate(contours):
                dist = cv2.pointPolygonTest(cnt, tuple(int(i) for i in agent_pixel_location), True)
                if dist >= 0:
                    best_idx = idx
                    break
                elif abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_idx = idx
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            cv2.drawContours(new_area, contours, best_idx, 1, -1)  # type: ignore
            self.explored_area = new_area.astype(bool)

        self._frontiers_px = self._get_frontiers()
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)

    def visualize(
        self,
        agent_xy: np.ndarray | None = None,
        selected_goal_xy: np.ndarray | tuple | None = None,
        target_nav_xy: np.ndarray | tuple | None = None,
        sg_confirmed_xys: list | None = None,
    ) -> np.ndarray:
        """VLFM-style BGR debug image of *this* map (port of ``ObstacleMap.visualize``).

        White = unknown, light green = explored, gray = inflated-unnavigable,
        black = obstacles, blue circles = frontier waypoints, red dot = agent.

        Goal markers (two distinct overlays, drawn in this z-order):
          * ``selected_goal_xy`` (gold ★) — the frontier the DecisionThread
            currently picked. ``shared_state.frontier.chosen_frontier_xy``.
          * ``target_nav_xy`` (magenta ◆) — concrete object target during
            exploitation. ``shared_state.nav.goal_xy`` when
            ``nav.goal_source == "target"``.

        Drawing both simultaneously shows the last frontier pick and the actual
        object once it is spotted.

        ``sg_confirmed_xys`` is an optional list of ``(x, y, label)`` tuples
        for confirmed scene-graph nodes matching the current search target.
        Each is drawn as a bright cyan circle + text label so you can see ALL
        candidate nodes, not just the one the agent is currently navigating to.
        """
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        vis_img[self.explored_area == 1] = (200, 255, 200)
        vis_img[self._navigable_map == 0] = (100, 100, 100)
        vis_img[self._map == 1] = (0, 0, 0)
        for frontier in self._frontiers_px:
            cv2.circle(vis_img, tuple(int(i) for i in frontier), 5, (200, 0, 0), 2)
        if selected_goal_xy is not None:
            gp = self._xy_to_px(np.asarray(selected_goal_xy, dtype=np.float32).reshape(1, 2))[0]
            if 0 <= gp[0] < self._map.shape[1] and 0 <= gp[1] < self._map.shape[0]:
                gx, gy = int(gp[0]), int(gp[1])
                cv2.circle(vis_img, (gx, gy), 12, (0, 0, 0), 2)  # outline
                cv2.circle(vis_img, (gx, gy), 11, (0, 215, 255), -1)  # gold fill
                cv2.drawMarker(
                    vis_img,
                    (gx, gy),
                    (255, 255, 255),
                    cv2.MARKER_STAR,
                    14,
                    2,
                )
        if target_nav_xy is not None:
            tp = self._xy_to_px(np.asarray(target_nav_xy, dtype=np.float32).reshape(1, 2))[0]
            if 0 <= tp[0] < self._map.shape[1] and 0 <= tp[1] < self._map.shape[0]:
                tx, ty = int(tp[0]), int(tp[1])
                cv2.circle(vis_img, (tx, ty), 14, (0, 0, 0), 2)  # outline
                cv2.circle(vis_img, (tx, ty), 13, (255, 0, 255), -1)  # magenta fill
                cv2.drawMarker(
                    vis_img,
                    (tx, ty),
                    (255, 255, 255),
                    cv2.MARKER_DIAMOND,
                    16,
                    2,
                )
        if sg_confirmed_xys:
            font = cv2.FONT_HERSHEY_SIMPLEX
            h_map, w_map = vis_img.shape[:2]
            for entry in sg_confirmed_xys:
                try:
                    wx, wy = float(entry[0]), float(entry[1])
                    lbl = str(entry[2]) if len(entry) > 2 else ""
                except Exception:
                    continue
                np_pt = self._xy_to_px(np.array([[wx, wy]], dtype=np.float32))[0]
                px_x, px_y = int(np_pt[0]), int(np_pt[1])
                if not (0 <= px_x < w_map and 0 <= px_y < h_map):
                    continue
                # Cyan filled circle with black outline — distinct from magenta target
                cv2.circle(vis_img, (px_x, px_y), 12, (0, 0, 0), 2)
                cv2.circle(vis_img, (px_x, px_y), 10, (255, 255, 0), -1)  # yellow
                cv2.circle(vis_img, (px_x, px_y), 5, (0, 0, 0), -1)  # dark center
                if lbl:
                    (tw, th), bl = cv2.getTextSize(lbl, font, 0.45, 1)
                    tx = px_x - tw // 2
                    ty = px_y - 14
                    cv2.rectangle(
                        vis_img,
                        (tx - 2, ty - th - 2),
                        (tx + tw + 2, ty + bl + 1),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        vis_img,
                        lbl,
                        (tx, ty),
                        font,
                        0.45,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
        if agent_xy is not None:
            ap = self._xy_to_px(np.asarray(agent_xy, dtype=np.float32).reshape(1, 2))[0]
            if 0 <= ap[0] < self._map.shape[1] and 0 <= ap[1] < self._map.shape[0]:
                cv2.circle(vis_img, (int(ap[0]), int(ap[1])), 6, (0, 0, 255), -1)
        vis_img = cv2.flip(vis_img, 0)
        return vis_img

    def _get_frontiers(self) -> np.ndarray:
        """Return VLFM waypoints and the map-derived unexplored side of each one."""
        nav_u8 = self._navigable_map.astype(np.uint8)
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        frontiers = detect_frontier_waypoints(
            nav_u8,
            explored_area,
            self._area_thresh_in_pixels,
        )
        explored_area[nav_u8 == 0] = 0
        self.frontier_unexplored_directions = [
            self._unexplored_direction_xy(frontier, nav_u8, explored_area)
            for frontier in frontiers
        ]
        return frontiers

    def _unexplored_direction_xy(self, frontier_px, navigable, explored):
        """Point from nearby explored cells toward nearby unexplored cells."""
        x, y = np.rint(frontier_px).astype(int)
        radius = 5
        x0, x1 = max(0, x - radius), min(explored.shape[1], x + radius + 1)
        y0, y1 = max(0, y - radius), min(explored.shape[0], y + radius + 1)
        seen_y, seen_x = np.nonzero(explored[y0:y1, x0:x1])
        unseen_y, unseen_x = np.nonzero(
            (navigable[y0:y1, x0:x1] > 0) & (explored[y0:y1, x0:x1] == 0)
        )
        if len(seen_x) == 0 or len(unseen_x) == 0:
            return None

        seen_px = np.array([seen_x.mean() + x0, seen_y.mean() + y0])
        unseen_px = np.array([unseen_x.mean() + x0, unseen_y.mean() + y0])
        direction_px = unseen_px - seen_px
        if np.linalg.norm(direction_px) <= 1e-6:
            return None
        points_xy = self._px_to_xy(np.stack([seen_px, unseen_px]))
        direction_xy = points_xy[1] - points_xy[0]
        norm = float(np.linalg.norm(direction_xy))
        return direction_xy / norm if norm > 1e-6 else None
