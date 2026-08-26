"""VLFM-style frontier detector backed by HabitatObstacleMap.

Feeds depth + camera pose from shared state into the HabitatObstacleMap (VLFM port),
then reprojects the resulting frontiers from its world XY into rtnav's ObstacleMap
pixel space so centroids/strategy see the planner's frame.
"""

from __future__ import annotations

import numpy as np
from frontier_exploration.frontier_detection import (
    contour_to_frontiers,
    get_frontier_midpoint,
)

from rtnav.core.data_types import FrontierMapSnapshot
from rtnav.modules.mapping.obstacle_map_habitat import HabitatObstacleMap


def _warm_frontier_jit() -> None:
    """Compile the Numba frontier kernels before sensor observations arrive."""
    contour = np.array(
        [[[1, 1]], [[2, 1]], [[3, 1]], [[4, 1]], [[5, 1]], [[6, 1]]],
        dtype=np.int32,
    )
    unexplored = np.full((8, 8), 255, dtype=np.uint8)
    contour_to_frontiers(contour, unexplored)
    contour_to_frontiers(np.asfortranarray(contour), unexplored)
    get_frontier_midpoint(contour)
    get_frontier_midpoint(contour[::2])


class VLFMFrontierDetector:
    """VLFM frontier detection on a parallel ObstacleMap built from rtnav inputs.

    Exposes ``detect`` and ``cluster_centroids`` for :class:`FrontierDetectionThread`.
    """

    def __init__(
        self,
        shared_state,
        *,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        floor_drop_height: float = -0.2,
        agent_radius: float = 0.18,
        area_thresh_m2: float = 1.5,
        hole_area_thresh: int = 100000,
        size: int = 1000,
        pixels_per_meter: int = 20,
        use_local_height_range: bool = False,
        only_forward_cameras: bool = True,
        update_every_n_steps: int = 4,
        camera_cfg=None,
        debug: bool = False,
    ):
        _warm_frontier_jit()
        self.shared_state = shared_state
        self.min_obstacle_height = float(min_obstacle_height)
        self.max_obstacle_height = float(max_obstacle_height)
        self.floor_drop_height = float(floor_drop_height)
        self.agent_radius = float(agent_radius)
        self.area_thresh_m2 = float(area_thresh_m2)
        self.hole_area_thresh = int(hole_area_thresh)
        self.size = int(size)
        self.pixels_per_meter = int(pixels_per_meter)
        self.use_local_height_range = bool(use_local_height_range)
        self.only_forward_cameras = bool(only_forward_cameras)
        # Update the HabitatObstacleMap only once per N env steps. Habitat
        # /obs fires at sim rate (~30Hz), but the pointnav policy decides
        # every SUBSTEPS_PER_POLICY=4 substeps (~7.5Hz). Updating at 30Hz
        # means 4 noisy depth passes accumulate between policy decisions
        # — flickers frontiers around tight corners. Throttling to ~policy
        # rate matches VLFM's per-step update pattern.
        self.update_every_n_steps = max(1, int(update_every_n_steps))
        self.debug = bool(debug)
        self._camera_cfg = camera_cfg

        self._build_habitat_map()
        # Separate dedup cursors per source: habitat_obs uses small int
        # step_id, detection_result uses huge Unix timestamps. Sharing one
        # cursor lets the first slow-path tick poison the fast path, since
        # any future step_id will then be < the stored Unix timestamp and
        # get silently skipped.
        self._last_habitat_step_id: float = -1.0
        self._last_detection_ts: float = -1.0
        self._last_logged_frontier_count: int = -1
        self.last_frontier_unexplored_directions: list = []

    def _build_habitat_map(self) -> None:
        """Create a fresh HabitatObstacleMap. Called from __init__ and reset."""
        self._habitat_map = HabitatObstacleMap(
            min_height=self.min_obstacle_height,
            max_height=self.max_obstacle_height,
            floor_drop_height=self.floor_drop_height,
            agent_radius=self.agent_radius,
            area_thresh=self.area_thresh_m2,
            hole_area_thresh=self.hole_area_thresh,
            size=self.size,
            pixels_per_meter=self.pixels_per_meter,
            use_local_height_range=self.use_local_height_range,
        )

    def reset(self) -> None:
        """Drop the accumulated map at episode boundaries."""
        self._build_habitat_map()
        self._last_habitat_step_id = -1.0
        self._last_detection_ts = -1.0
        self._last_logged_frontier_count = -1
        self.last_frontier_unexplored_directions = []

    def _camera_min_max(self) -> tuple[float, float]:
        """Resolve (min_depth, max_depth) from the camera cfg."""
        cam_cfg = self._camera_cfg
        if cam_cfg is None:
            return 0.5, 5.0
        min_d = float(
            getattr(cam_cfg, "sensor_depth_min_m", None) or getattr(cam_cfg, "min_depth", 0.0)
        )
        max_d = float(
            getattr(cam_cfg, "sensor_depth_max_m", None) or getattr(cam_cfg, "max_depth", 5.0)
        )
        if max_d <= min_d:
            max_d = min_d + 1e-3
        return min_d, max_d

    def _is_forward_camera(self) -> bool:
        if not self.only_forward_cameras:
            return True
        cam_cfg = self._camera_cfg
        if cam_cfg is None:
            return True
        return bool(getattr(cam_cfg, "forward_facing", True))

    def _camera_height(self) -> float:
        cam_cfg = self._camera_cfg
        if cam_cfg is None or getattr(cam_cfg, "position_base", None) is None:
            return 0.0
        return float(cam_cfg.position_base[2])

    @staticmethod
    def _normalize_depth(depth_m, min_d, max_d):
        """Depth (m) -> [0,1] for HabitatObstacleMap.update_map; non-finite/<=0 -> 0."""
        denom = max(max_d - min_d, 1e-3)
        depth_norm = depth_m.astype(np.float32, copy=True)
        invalid = ~np.isfinite(depth_norm) | (depth_norm <= 0.0)
        depth_norm = (depth_norm - min_d) / denom
        depth_norm[invalid] = 0.0
        np.clip(depth_norm, 0.0, 1.0, out=depth_norm)
        return depth_norm

    def _push_observations(self) -> None:
        """Feed the freshest camera observation into the HabitatObstacleMap.

        Source priority: (1) sensor.habitat_obs — fast path, fires every /obs frame
        regardless of the detector; (2) perception.detection_result — slower fallback
        (gated by the inference scheduler) for non-Habitat hosts.
        """
        if self._try_push_from_habitat_obs():
            return
        self._try_push_from_detection_result()

    def update_map(self) -> None:
        """Ingest the latest observation without waiting to publish frontiers."""
        self._push_observations()

    def _try_push_from_habitat_obs(self) -> bool:
        """Fast path: read raw HabitatObservation from sensor.habitat_obs.

        Returns True if an observation was consumed (whether or not the
        map actually updated successfully), False if no usable habitat_obs
        was available — letting the caller fall back to the slower
        detection_result path.
        """
        sensor = getattr(self.shared_state, "sensor", None)
        habitat_obs = getattr(sensor, "habitat_obs", None) if sensor is not None else None
        if habitat_obs is None:
            return False
        # Deduplicate + throttle: HabitatObservation carries the env's
        # step_id; only re-push when it has advanced by
        # ``update_every_n_steps`` since the last push. At habitat's ~30Hz
        # /obs rate and policy SUBSTEPS_PER_POLICY=4, throttling by 4
        # gives ~7.5Hz map updates — matches VLFM's per-policy-step rate
        # and stops mid-traversal frontier flicker. Uses its own cursor
        # (separate from the detection_result wall-clock cursor).
        step_id = float(getattr(habitat_obs, "step_id", 0))
        if step_id <= self._last_habitat_step_id:
            return True
        if (
            self._last_habitat_step_id >= 0.0
            and (step_id - self._last_habitat_step_id) < self.update_every_n_steps
        ):
            # Treat as consumed (don't fall through to the slow detection
            # path) but skip the actual map update this tick.
            return True
        self._last_habitat_step_id = step_id

        depth_raw = getattr(habitat_obs, "depth", None)
        if depth_raw is None:
            return True
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]

        # Match PerceptionThread's pose convention exactly:
        # T_world_base = build_T_world_base(gps, compass), plus camera
        # height in z to make this tf_camera_to_episodic (VLFM convention).
        try:
            from rtnav.modules.perception.camera_geometry import (
                build_K,
                build_T_world_base,
                decode_depth_to_meters,
            )
        except Exception:
            return True
        gps = tuple(getattr(habitat_obs, "gps", (0.0, 0.0)))
        compass = float(getattr(habitat_obs, "compass", 0.0))
        T_world_base = build_T_world_base(list(gps), compass)

        for cam_cfg in (self._camera_cfg,):
            if not bool(getattr(cam_cfg, "forward_facing", True)):
                continue
            min_d = float(
                getattr(cam_cfg, "sensor_depth_min_m", None) or getattr(cam_cfg, "min_depth", 0.5)
            )
            max_d = float(
                getattr(cam_cfg, "sensor_depth_max_m", None) or getattr(cam_cfg, "max_depth", 5.0)
            )

            # HabitatObservation.depth is the raw env_node payload, which for
            # habitat (normalize_depth=True) is [0,1] — NOT meters. Decode to
            # meters (depth_normalized decides scale vs pass-through), then
            # renormalize to [0,1] for ObstacleMap.update_map's contract.
            try:
                depth_m = decode_depth_to_meters(
                    depth_raw,
                    getattr(cam_cfg, "depth_normalized", False),
                    getattr(cam_cfg, "sensor_depth_min_m", 0.0),
                    getattr(cam_cfg, "sensor_depth_max_m", 10.0),
                )
            except Exception:
                continue
            if depth_m.ndim == 3:
                depth_m = depth_m[:, :, 0]
            depth_norm = self._normalize_depth(depth_m, min_d, max_d)

            try:
                K = build_K(
                    getattr(cam_cfg, "hfov_deg", 0.0), depth_norm.shape[1], depth_norm.shape[0]
                )
            except Exception:
                continue
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            H, W = depth_norm.shape
            topdown_fov = 2.0 * float(np.arctan(W / (2.0 * fx)))

            tf = T_world_base.copy()
            tf[2, 3] = float(getattr(cam_cfg, "position_base", (0, 0, 0))[2])

            try:
                self._habitat_map.update_map(
                    depth_norm,
                    tf,
                    min_d,
                    max_d,
                    fx,
                    fy,
                    topdown_fov,
                )
            except Exception as e:
                if self.debug:
                    print(f"[VLFMFrontier] update_map raised: {e}")
                continue
            return True
        return True

    def _try_push_from_detection_result(self) -> None:
        """Slow fallback: read from DetectorThread's MultiCameraDetectionResult."""
        perception = getattr(self.shared_state, "perception", None)
        if perception is None:
            return
        det = getattr(perception, "detection_result", None)
        if det is None or not getattr(det, "camera_results", None):
            return

        ts = float(getattr(det, "timestamp", 0.0) or 0.0)
        if ts <= self._last_detection_ts:
            return
        self._last_detection_ts = ts

        T_world_base = getattr(det, "T_world_base", None)
        if T_world_base is None:
            return

        for cam_name, cam_result in det.camera_results.items():
            if not self._is_forward_camera():
                continue
            depth_m = getattr(cam_result, "depth_image", None)
            K = getattr(cam_result, "intrinsics_depth", None)
            if depth_m is None or K is None:
                continue
            if depth_m.ndim == 3:
                depth_m = depth_m[:, :, 0]

            min_d, max_d = self._camera_min_max()
            depth_norm = self._normalize_depth(depth_m, min_d, max_d)

            fx = float(K[0, 0])
            fy = float(K[1, 1])
            H, W = depth_norm.shape
            topdown_fov = 2.0 * np.arctan(W / (2.0 * fx))

            tf = T_world_base.copy()
            tf[2, 3] = self._camera_height()

            try:
                self._habitat_map.update_map(
                    depth_norm,
                    tf,
                    min_d,
                    max_d,
                    fx,
                    fy,
                    topdown_fov,
                )
            except Exception as e:
                if self.debug:
                    print(f"[VLFMFrontier] update_map raised on {cam_name}: {e}")
                continue

    def detect(self, obs_map) -> tuple[np.ndarray, list[np.ndarray]]:
        """Drives the VLFM map from shared_state, reprojects frontiers to rtnav px.

        ``obs_map`` is used only to (a) reproject VLFM frontiers (world XY)
        into rtnav pixel space and (b) size the returned ``frontier_mask``
        to rtnav's map shape.
        """
        self.update_map()

        # Pull frontiers (world XY) from the Habitat-port map.
        frontiers_xy = self._habitat_map.frontiers
        h, w = obs_map.explored.shape if hasattr(obs_map, "explored") else (1000, 1000)
        frontier_mask = np.zeros((h, w), dtype=np.uint8)
        clusters: list[np.ndarray] = []
        directions: list = []

        if frontiers_xy is None or len(frontiers_xy) == 0:
            self._log_count(0, [])
            self.last_frontier_unexplored_directions = []
            return frontier_mask, clusters

        # World XY -> rtnav obstacle map pixel coords.
        try:
            px = obs_map.xy_to_px(np.asarray(frontiers_xy, dtype=np.float32))
        except Exception:
            self.last_frontier_unexplored_directions = []
            return frontier_mask, clusters

        unexplored_directions = self._habitat_map.frontier_unexplored_directions
        boundaries_xy = self._habitat_map.frontier_boundaries
        out_of_bounds = 0
        for i, p in enumerate(px):
            x, y = int(p[0]), int(p[1])
            if 0 <= x < w and 0 <= y < h:
                boundary_xy = boundaries_xy[i] if i < len(boundaries_xy) else None
                boundary_px = (
                    obs_map.xy_to_px(np.asarray(boundary_xy, dtype=np.float32))
                    if boundary_xy is not None and len(boundary_xy)
                    else np.array([[x, y]], dtype=np.int32)
                )
                boundary_px = np.asarray(boundary_px, dtype=np.int32).reshape(-1, 2)
                in_bounds = (
                    (boundary_px[:, 0] >= 0)
                    & (boundary_px[:, 0] < w)
                    & (boundary_px[:, 1] >= 0)
                    & (boundary_px[:, 1] < h)
                )
                boundary_px = boundary_px[in_bounds]
                if not len(boundary_px):
                    boundary_px = np.array([[x, y]], dtype=np.int32)
                frontier_mask[boundary_px[:, 1], boundary_px[:, 0]] = 1
                clusters.append(boundary_px)
                directions.append(
                    unexplored_directions[i] if i < len(unexplored_directions) else None
                )
            else:
                out_of_bounds += 1

        self.last_frontier_unexplored_directions = directions
        self._log_count(len(clusters), list(frontiers_xy), out_of_bounds=out_of_bounds)
        return frontier_mask, clusters

    def map_snapshot(self) -> FrontierMapSnapshot:
        """Copy the VLFM explored map associated with the latest frontiers."""
        habitat_map = self._habitat_map
        return FrontierMapSnapshot(
            explored=np.asarray(habitat_map.explored_area, dtype=bool).copy(),
            pixels_per_meter=float(habitat_map.pixels_per_meter),
            episode_pixel_origin=tuple(
                float(value) for value in habitat_map._episode_pixel_origin[:2]
            ),
        )

    def _log_count(self, n_in: int, frontiers_xy, *, out_of_bounds: int = 0) -> None:
        """Print frontier world coords whenever the count *changes*.

        Helps the user cross-reference VLFM frontiers against rtnav's
        map (since the map_viz_web only renders rtnav's obstacle map,
        not the parallel Habitat-port map).
        """
        n_total = len(frontiers_xy) if frontiers_xy is not None else 0
        if n_total == self._last_logged_frontier_count and out_of_bounds == 0:
            return
        self._last_logged_frontier_count = n_total
        if n_total == 0:
            return
        coords = ", ".join(f"({float(p[0]):+.2f},{float(p[1]):+.2f})" for p in frontiers_xy[:8])
        extra = ""
        if out_of_bounds:
            extra = f" ({out_of_bounds} outside rtnav map)"
        print(
            f"[VLFMFrontier] {n_total} frontier(s) world_xy=[{coords}] "
            f"-> {n_in} in rtnav map{extra}"
        )

    def cluster_centroids(self, clusters: list[np.ndarray]) -> np.ndarray:
        """Return the path-length midpoint of each full frontier boundary."""
        if len(clusters) == 0:
            return np.zeros((0, 2))
        out = np.zeros((len(clusters), 2), dtype=np.float32)
        for i, pts in enumerate(clusters):
            points = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
            points = points[np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]]
            out[i] = (
                points[0]
                if len(points) < 2
                else get_frontier_midpoint(points.reshape(-1, 1, 2))
            )
        return out