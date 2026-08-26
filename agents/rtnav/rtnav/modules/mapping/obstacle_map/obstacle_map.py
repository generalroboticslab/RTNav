import time
from typing import Optional

import cv2
import numpy as np

from rtnav.core.data_types import MappingOutput, PerceptionOutput
from rtnav.modules.mapping.obstacle_map.terrain_analysis import TerrainAnalyzer
from rtnav.modules.mapping.obstacle_map.utils import (
    find_navigable_frontier_goal as _find_navigable_frontier_goal,
)
from rtnav.modules.mapping.utils import fog_of_war_local_inplace


class ObstacleMap:
    """2.5D elevation-based traversability map for local planning."""

    NO_DATA_CM = -32768  # height-map sentinel for cells with no depth
    UNKNOWN_TRAVERSABILITY = 1.0  # traversability assigned to unobserved cells
    NAVIGABILITY_THRESHOLD = 0.1  # traversability at/above this = navigable
    TRAVERSABILITY_RADIUS_M = 15.0  # radius around robot re-analyzed each frame

    def __init__(
        self,
        size: int = 500,
        pixels_per_meter: float = 20.0,
        config=None,
    ):
        self.size = size
        self.ppm = pixels_per_meter
        self.origin_px = np.array([size // 2, size // 2])

        self._height_map_cm = np.full((size, size), self.NO_DATA_CM, dtype=np.int16)
        self._traversability_u8 = np.full((size, size), 255, dtype=np.uint8)
        self.navigable = np.ones((size, size), np.uint8)
        self.explored = np.zeros((size, size), np.uint8)
        # Raw cumulative FOW polygon (no height gate); frontier detector uses this for unmapped territory.
        self._fow_raw = np.zeros((size, size), np.uint8)

        # Persistent obstacle accumulator: once a cell is True it stays True for the episode.
        self._obstacle_mask = np.zeros((size, size), dtype=bool)

        cfg = config  # mapping sub-config (MappingConfig)
        self._cfg = cfg

        self._last_robot_pose = (0.0, 0.0, 0.0)
        self._last_robot_z = 0.0

        self._terrain_analyzer = TerrainAnalyzer(
            grid_size=1.0 / self.ppm,
            slope_threshold=cfg.slope_threshold,
            erode_boundary=cfg.erode_boundary,
        )

        # Per-tick caches for derived float32 rasters, invalidated via _invalidate_derived_caches.
        self._cached_height_map_f32: Optional[np.ndarray] = None
        self._cached_traversability_f32: Optional[np.ndarray] = None
        self._cached_occupancy_u8: Optional[np.ndarray] = None

    def _invalidate_derived_caches(self) -> None:
        self._cached_height_map_f32 = None
        self._cached_traversability_f32 = None
        self._cached_occupancy_u8 = None

    @property
    def height_map(self) -> np.ndarray:
        if self._cached_height_map_f32 is None:
            result = self._height_map_cm.astype(np.float32) / 100.0
            result[self._height_map_cm == self.NO_DATA_CM] = -np.inf
            self._cached_height_map_f32 = result
        return self._cached_height_map_f32

    @property
    def traversability(self) -> np.ndarray:
        if self._cached_traversability_f32 is None:
            self._cached_traversability_f32 = self._traversability_u8.astype(np.float32) / 255.0
        return self._cached_traversability_f32

    @property
    def occupancy(self):
        if self._cached_occupancy_u8 is not None:
            return self._cached_occupancy_u8
        thresh = self._cfg.occupancy_trav_threshold
        thresh_u8 = int(np.clip(thresh, 0.0, 1.0) * 255)
        has_depth = self._height_map_cm != self.NO_DATA_CM
        occ = ((self._traversability_u8 < thresh_u8) & has_depth).astype(np.uint8)
        min_area = self._cfg.min_obstacle_area_px
        if min_area > 1 and occ.any():
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
            if num_labels > 1:
                areas = stats[:, cv2.CC_STAT_AREA]
                small = np.zeros(num_labels, dtype=bool)
                small[1:] = areas[1:] < min_area
                if small.any():
                    occ[small[labels]] = 0
        self._cached_occupancy_u8 = occ
        return occ

    def update_from_perception(self, perception_out: PerceptionOutput) -> MappingOutput:
        pts_3d = perception_out.points_world
        T = perception_out.robot_pose
        yaw = np.arctan2(T[1, 0], T[0, 0])
        robot_pose = (float(T[0, 3]), float(T[1, 3]), float(yaw))
        self._last_robot_z = float(T[2, 3])

        mcfg = self._cfg
        h_min = float(mcfg.obstacle_height_min)
        h_max = float(mcfg.obstacle_height_max)
        if mcfg.use_local_height_range:
            base_z = self._last_robot_z
            if np.isfinite(h_min):
                h_min += base_z
            if np.isfinite(h_max):
                h_max += base_z
        if pts_3d.size > 0 and (np.isfinite(h_min) or np.isfinite(h_max)):
            z = pts_3d[:, 2]
            pts_3d = pts_3d[(z >= h_min) & (z <= h_max)]

        pts_xy = pts_3d[:, :2] if pts_3d.size > 0 else np.empty((0, 2))
        heights = pts_3d[:, 2] if pts_3d.size > 0 else np.empty(0)

        cam_poses, fov_list, range_list = [], [], []
        for name in perception_out.camera_extrinsics:
            if name not in perception_out.camera_meta:
                continue
            T_wc = perception_out.camera_extrinsics[name]
            meta = perception_out.camera_meta[name]
            cam_poses.append(
                (float(T_wc[0, 3]), float(T_wc[1, 3]), float(np.arctan2(T_wc[1, 2], T_wc[0, 2])))
            )
            fov_list.append(float(meta.get("fov_deg", 90.0)))
            range_list.append(float(meta.get("range_m", 3.0) or 3.0))

        return self._update_core(pts_xy, heights, robot_pose, cam_poses, fov_list, range_list)

    def _update_core(
        self, points_xy, heights, robot_pose, camera_poses, fov_deg, fov_range_m
    ) -> MappingOutput:
        self._invalidate_derived_caches()

        self._last_robot_pose = robot_pose

        if points_xy.size > 0:
            self._integrate_points(points_xy, heights)

        # Terrain analysis: updates _obstacle_mask, _traversability_u8, navigable.
        self._update_traversability()

        self._update_explored(camera_poses, fov_deg, fov_range_m, points_xy)

        # Explored is purely historical: monotonic snapshot of _fow_raw.
        self.explored = self._fow_raw.copy()

        robot_px = self.xy_to_px(np.array([[robot_pose[0], robot_pose[1]]]))[0]
        self._keep_explored_cc_at_agent(robot_px)

        return MappingOutput(
            occupancy=self.occupancy,
            traversability=self.traversability,
            explored=self.explored,
            navigable=self.navigable,
            robot_pose_xyyaw=robot_pose,
            timestamp=time.time(),
        )

    def clear_explored(self) -> None:
        """Wipe cumulative explored/FOW state (frontier-exhaustion restart).
        Obstacles and traversability are kept. Fresh arrays, not in-place fill:
        the published MappingOutput aliases the old ones."""
        self._fow_raw = np.zeros((self.size, self.size), np.uint8)
        self.explored = np.zeros((self.size, self.size), np.uint8)

    def _integrate_points(self, points_xy, heights):
        px = self.xy_to_px(points_xy)
        heights_cm = (heights * 100.0).astype(np.int16)
        np.maximum.at(self._height_map_cm, (px[:, 1], px[:, 0]), heights_cm)

    def _update_explored(self, camera_poses, fov_deg, fov_range_m, points_xy=None):
        for (x, y, yaw), fov, r in zip(camera_poses, fov_deg, fov_range_m):
            if not (r and r > 0 and np.isfinite(r)):
                continue
            if not (fov and fov > 0 and np.isfinite(fov)):
                continue

            # Clip each ray to the farthest depth point in its direction so
            # explored never extends past what the camera actually observed.
            per_ray = self._depth_ray_ranges_px(points_xy, x, y, yaw, fov, r)
            cam_px = tuple(self.xy_to_px(np.array([[x, y]]))[0])
            fog_of_war_local_inplace(
                output_mask=self._fow_raw,
                agent_px=cam_px,
                agent_yaw=yaw,
                fov_deg=fov,
                max_range_px=float(r * self.ppm),
                occupancy_map=self._obstacle_mask.astype(np.uint8),
                per_ray_max_px=per_ray,
            )

    def _depth_ray_ranges_px(self, points_xy, x, y, yaw, fov_deg, r, num_rays=60):
        per_ray = np.zeros(num_rays, dtype=np.float32)
        if points_xy is None or len(points_xy) == 0:
            return per_ray
        dx = points_xy[:, 0] - x
        dy = points_xy[:, 1] - y
        rng = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx) - yaw
        ang = np.arctan2(np.sin(ang), np.cos(ang))
        half = np.radians(fov_deg) / 2.0
        keep = (np.abs(ang) <= half) & (rng <= r)
        if keep.any():
            frac = (ang[keep] + half) / (2 * half)
            b = np.clip(np.round(frac * (num_rays - 1)).astype(int), 0, num_rays - 1)
            np.maximum.at(per_ray, b, rng[keep] * self.ppm)
        return per_ray

    def _keep_explored_cc_at_agent(self, robot_px):
        explored = self.explored
        if not explored.any():
            return
        rx, ry = int(robot_px[0]), int(robot_px[1])
        if not (0 <= rx < self.size and 0 <= ry < self.size):
            return

        # Cut explored along dilated obstacles so islands behind walls form a separate CC.
        obs_thick = cv2.dilate(
            self._obstacle_mask.astype(np.uint8),
            np.ones((3, 3), np.uint8),
        )
        explored_cut = explored.copy()
        explored_cut[obs_thick > 0] = 0

        num_labels, labels = cv2.connectedComponents(explored_cut, connectivity=4)
        if num_labels <= 2:
            # Fall back to the un-cut CC when nothing survived or only one island.
            num_labels, labels = cv2.connectedComponents(
                explored.astype(np.uint8),
                connectivity=4,
            )
            if num_labels <= 2:
                return

        robot_label = int(labels[ry, rx])
        if robot_label == 0:
            # Robot fell inside the dilated obstacle ring: use the nearest labeled cell's CC.
            nz_y, nz_x = np.nonzero(labels)
            if nz_y.size == 0:
                return
            d2 = (nz_x - rx) ** 2 + (nz_y - ry) ** 2
            nearest = int(np.argmin(d2))
            robot_label = int(labels[nz_y[nearest], nz_x[nearest]])
            if robot_label == 0:
                return

        # Dilate the kept CC to recover obstacle-edge cells zeroed by the dilation step.
        keep = (labels == robot_label).astype(np.uint8)
        keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))
        self.explored[keep == 0] = 0

    def _update_traversability(self):
        mcfg = self._cfg
        local_radius_m = self.TRAVERSABILITY_RADIUS_M
        edge_pad = max(10, int(mcfg.safety_margin * self.ppm) + 5)

        robot_px = self.xy_to_px(np.array([[self._last_robot_pose[0], self._last_robot_pose[1]]]))[
            0
        ]
        rx, ry = int(robot_px[0]), int(robot_px[1])
        radius_px = int(local_radius_m * self.ppm)

        inner_x_min = max(0, rx - radius_px)
        inner_x_max = min(self.size, rx + radius_px)
        inner_y_min = max(0, ry - radius_px)
        inner_y_max = min(self.size, ry + radius_px)

        outer_x_min = max(0, inner_x_min - edge_pad)
        outer_x_max = min(self.size, inner_x_max + edge_pad)
        outer_y_min = max(0, inner_y_min - edge_pad)
        outer_y_max = min(self.size, inner_y_max + edge_pad)

        pad_left = inner_x_min - outer_x_min
        pad_top = inner_y_min - outer_y_min
        inner_h = inner_y_max - inner_y_min
        inner_w = inner_x_max - inner_x_min

        local_height_cm = self._height_map_cm[
            outer_y_min:outer_y_max, outer_x_min:outer_x_max
        ].copy()
        valid_local = local_height_cm > self.NO_DATA_CM

        trav_raw, valid_for_nav = self._compute_traversability(local_height_cm, valid_local)

        trav_raw_inner = trav_raw[pad_top : pad_top + inner_h, pad_left : pad_left + inner_w]
        valid_inner = valid_for_nav[pad_top : pad_top + inner_h, pad_left : pad_left + inner_w]

        # Accumulate hard obstacles into persistent mask (raw, no safety margin).
        new_obstacle = valid_inner & (trav_raw_inner < self.NAVIGABILITY_THRESHOLD)

        # Frame-local height-above-robot check (not accumulated).
        robot_z = float(self._last_robot_z)
        height_map_inner = self._height_map_cm[inner_y_min:inner_y_max, inner_x_min:inner_x_max]
        has_depth_inner = height_map_inner != self.NO_DATA_CM
        height_m_inner = height_map_inner.astype(np.float32) / 100.0
        wall_h_thresh = float(self._cfg.wall_height_threshold_m)
        height_obstacle = has_depth_inner & ((height_m_inner - robot_z) > wall_h_thresh)

        # within_radius scopes the trav repaint; obstacle checks run on the full inner window.
        wall_radius_m = float(self._cfg.wall_height_radius_m)
        if wall_radius_m > 0:
            wall_radius_px = wall_radius_m * self.ppm
            rx_in = rx - inner_x_min
            ry_in = ry - inner_y_min
            yy, xx = np.ogrid[0:inner_h, 0:inner_w]
            within_radius = (yy - ry_in) ** 2 + (xx - rx_in) ** 2 <= wall_radius_px**2
        else:
            within_radius = np.ones((inner_h, inner_w), dtype=bool)

        self._obstacle_mask[inner_y_min:inner_y_max, inner_x_min:inner_x_max] |= new_obstacle

        # Trav writes restricted to within the disc; trav stays raw (no inflation).
        trav_u8 = (np.clip(trav_raw_inner, 0.0, 1.0) * 255).astype(np.uint8)
        region = self._traversability_u8[inner_y_min:inner_y_max, inner_x_min:inner_x_max]
        local_obs = self._obstacle_mask[inner_y_min:inner_y_max, inner_x_min:inner_x_max]
        # Safe cells inside the disc: write fresh trav each frame
        safe_inner = within_radius & valid_inner & ~local_obs & ~height_obstacle
        region[safe_inner] = trav_u8[safe_inner]
        region[local_obs] = 0
        region[height_obstacle] = 0

        # Navigable = raw trav above threshold (no inflation, no morphological close).
        thresh_u8 = int(self.NAVIGABILITY_THRESHOLD * 255)
        currently_navigable = (self._traversability_u8 >= thresh_u8).astype(np.uint8)
        np.minimum(self.navigable, currently_navigable, out=self.navigable)

    def _compute_traversability(self, local_height_cm, valid_local):
        """Run terrain analysis. Returns (trav_raw, valid_for_nav)."""
        mcfg = self._cfg
        height_m = local_height_cm.astype(np.float32) / 100.0
        height_for_analysis = np.where(valid_local, height_m, np.nan)

        obstacle = self._terrain_analyzer.analyze(height_for_analysis)
        trav_raw = 1.0 - obstacle  # traversable = 1 - obstacle
        trav_raw[~valid_local] = self.UNKNOWN_TRAVERSABILITY

        # Boundary safe band: force cells within boundary_safe_band_px of the depth edge to safe.
        band_px = int(mcfg.boundary_safe_band_px)
        if band_px > 0 and valid_local.any():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
            interior_strict = cv2.erode(valid_local.astype(np.uint8), k).astype(bool)
            boundary_band = valid_local & ~interior_strict
            if boundary_band.any():
                trav_raw[boundary_band] = self.UNKNOWN_TRAVERSABILITY

        return trav_raw, valid_local

    def xy_to_px(self, xy: np.ndarray) -> np.ndarray:
        px = np.empty_like(xy, float)
        px[:, 0] = self.origin_px[0] + xy[:, 0] * self.ppm
        px[:, 1] = self.origin_px[1] - xy[:, 1] * self.ppm
        return np.clip(px, 0, self.size - 1).astype(int)

    def px_to_xy(self, px: np.ndarray) -> np.ndarray:
        xy = np.empty_like(px, float)
        xy[:, 0] = (px[:, 0] - self.origin_px[0]) / self.ppm
        xy[:, 1] = -(px[:, 1] - self.origin_px[1]) / self.ppm
        return xy

    def is_navigable_px(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        return bool(self.navigable[y, x])

    def find_navigable_frontier_goal(self, centroid_px):
        return _find_navigable_frontier_goal(self, centroid_px)
