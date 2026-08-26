"""HabitatObservation → PerceptionOutput: build camera geometry, project depth
to a world point cloud, and publish it to shared_state."""

import time
from typing import Optional

import numpy as np

from rtnav.core.data_types import CameraFrame, PerceptionOutput
from rtnav.modules.perception.camera_geometry import (
    build_K,
    build_T_base_cam,
    build_T_world_base,
    decode_depth_to_meters,
    depth_to_pointcloud_world,
)
from rtnav.utils.worker_thread import WorkerThread


class PerceptionThread(WorkerThread):
    """Reads sensor.habitat_obs → writes perception_output (+version),
    sensor.latest_odom, sensor.latest_cameras."""

    CAMERA_HISTORY_LIMIT = 32
    CAMERA_HISTORY_MIN_YAW_RAD = np.deg2rad(12.0)
    CAMERA_HISTORY_MIN_DIST_M = 0.25

    def __init__(self, shared_state, shutdown_event, cfg, verbose: bool = False):
        super().__init__(shared_state, shutdown_event, name="PerceptionThread")
        self.cam_cfg = cfg.camera
        self.verbose = verbose
        self._last_step_id: Optional[int] = None

    def _step(self) -> None:
        obs = self._next_observation()
        if obs is None:
            time.sleep(0.005)
            return
        self._process(obs)

    def _reset(self) -> None:
        """Clear per-episode camera history when the runner advances the episode."""
        if not self._episode_advanced():
            return
        with self.shared_state.lock:
            self.shared_state.sensor.camera_history = []
        self._last_step_id = None
        print("[Perception] episode reset")

    def _next_observation(self):
        """The latest habitat_obs if its step_id changed, else None
        (an episode boundary rewinds step_id to 0, which counts as changed)."""
        with self.shared_state.lock:
            obs = self.shared_state.sensor.habitat_obs
        if obs is None or obs.step_id == self._last_step_id:
            return None
        if self._last_step_id is not None and obs.step_id < self._last_step_id:
            self._last_step_id = None
        return obs

    def _process(self, obs) -> None:
        """Build perception for one observation and publish it to shared_state."""
        t0 = time.time()
        try:
            perc_out, T_world_base, cam_frame = self._build_perception(obs)
        except Exception as e:
            if self.verbose:
                print(f"[Perception] build failed: {e}")
            self._last_step_id = obs.step_id
            return
        self._publish(obs, perc_out, T_world_base, cam_frame)
        self._last_step_id = obs.step_id
        if self.verbose:
            print(
                f"[Perception] step={obs.step_id} pts={len(perc_out.points_world)} "
                f"{(time.time() - t0) * 1000:.1f}ms"
            )

    # ── build ─────────────────────────────────────────────────────────

    def _build_perception(self, obs):
        """One HabitatObservation → (PerceptionOutput, T_world_base, CameraFrame|None)."""
        cam = self.cam_cfg
        depth_m = self._to_metric_depth(obs)

        K = build_K(cam.hfov_deg, depth_m.shape[1], depth_m.shape[0])
        T_world_base = build_T_world_base(list(obs.gps), float(obs.compass))
        T_world_cam = T_world_base @ build_T_base_cam(cam.position_base, cam.forward_facing)
        pts_world = self._project_to_world(obs, depth_m, T_world_cam, K)

        perc_out = PerceptionOutput(
            points_world=pts_world,
            camera_extrinsics={cam.name: T_world_cam},
            robot_pose=T_world_base,
            camera_meta={cam.name: {"fov_deg": cam.hfov_deg, "range_m": cam.max_depth}},
        )
        cam_frame = None
        if obs.rgb is not None:
            cam_frame = CameraFrame(
                timestamp=obs.timestamp,
                rgb=obs.rgb,
                depth=depth_m,
                T_world_cam=T_world_cam,
                intrinsics=K,
                min_depth_m=float(cam.min_depth),
                max_depth_m=float(cam.max_depth),
            )
        return perc_out, T_world_base, cam_frame

    def _to_metric_depth(self, obs) -> np.ndarray:
        """Raw depth → metric depth: fill small zero-holes, then decode to meters."""
        raw = obs.depth
        hole_thresh = int(self.cam_cfg.depth_hole_area_thresh_px2)
        if hole_thresh > 0 and raw is not None:
            try:
                from rtnav.modules.perception.cpp_accel import fill_small_holes

                rd = raw[:, :, 0] if raw.ndim == 3 else raw
                raw = fill_small_holes(np.ascontiguousarray(rd, dtype=np.float32), hole_thresh)
            except Exception as exc:
                if self.verbose:
                    print(f"[Perception] fill_small_holes failed: {exc}")
        cam = self.cam_cfg
        depth_m = decode_depth_to_meters(
            raw, cam.depth_normalized, cam.sensor_depth_min_m, cam.sensor_depth_max_m
        )
        return depth_m[:, :, 0] if depth_m.ndim == 3 else depth_m

    def _project_to_world(self, obs, depth_m, T_world_cam, K) -> np.ndarray:
        cam = self.cam_cfg
        if obs.rgb is not None:
            pts_world, _ = depth_to_pointcloud_world(
                depth_m, T_world_cam, K, cam.min_depth, cam.max_depth, rgb=obs.rgb
            )
        else:
            pts_world = depth_to_pointcloud_world(
                depth_m, T_world_cam, K, cam.min_depth, cam.max_depth
            )
        if self.verbose and len(pts_world) == 0:
            finite = depth_m[np.isfinite(depth_m)]
            rng = (
                f"[{float(finite.min()):.2f}, {float(finite.max()):.2f}]" if finite.size else "none"
            )
            print(
                f"[Perception] empty cloud: depth={rng} "
                f"valid=[{self.cam_cfg.min_depth}, {self.cam_cfg.max_depth}]"
            )
        return pts_world

    # ── publish ───────────────────────────────────────────────────────

    def _publish(self, obs, perc_out, T_world_base, cam_frame) -> None:
        """Atomically write the perception output + odom + camera frame to shared_state."""
        yaw = float(np.arctan2(np.sin(obs.compass), np.cos(obs.compass)))
        with self.shared_state.lock:
            self.shared_state.perception.perception_output = perc_out
            self.shared_state.perception.perception_version += 1
            sensor = self.shared_state.sensor
            sensor.latest_odom = (float(T_world_base[0, 3]), float(T_world_base[1, 3]), yaw)
            if cam_frame is not None:
                sensor.latest_cameras[self.cam_cfg.name] = cam_frame
                self._append_camera_history(cam_frame)

    def _append_camera_history(self, cam_frame) -> None:
        """Snapshot cam_frame into camera_history if the view moved enough. Caller holds the lock."""
        history = list(getattr(self.shared_state.sensor, "camera_history", []) or [])
        if self._view_moved_enough(history, cam_frame):
            history.append({self.cam_cfg.name: cam_frame})
            self.shared_state.sensor.camera_history = history[-self.CAMERA_HISTORY_LIMIT :]

    def _view_moved_enough(self, history, cam_frame) -> bool:
        """True if the camera moved/rotated past the history thresholds since its last snapshot."""
        prev = None
        for snapshot in reversed(history):
            prev = (snapshot or {}).get(self.cam_cfg.name)
            if prev is not None:
                break
        if prev is None:
            return True
        T_prev = np.asarray(prev.T_world_cam, dtype=float)
        T_cur = np.asarray(cam_frame.T_world_cam, dtype=float)
        if float(np.linalg.norm(T_cur[:2, 3] - T_prev[:2, 3])) >= self.CAMERA_HISTORY_MIN_DIST_M:
            return True
        f_prev, f_cur = T_prev[:2, 2], T_cur[:2, 2]
        n_prev, n_cur = float(np.linalg.norm(f_prev)), float(np.linalg.norm(f_cur))
        if n_prev <= 1e-6 or n_cur <= 1e-6:
            return True
        cos_yaw = float(np.clip(np.dot(f_prev, f_cur) / (n_prev * n_cur), -1.0, 1.0))
        return float(np.arccos(cos_yaw)) >= self.CAMERA_HISTORY_MIN_YAW_RAD
