"""Rerun 3D scene-graph visualizer thread."""

import colorsys
import threading
import time
from collections import deque

import numpy as np
import rerun as rr


class SceneGraphRerunVisualizer(threading.Thread):
    def __init__(
        self,
        shared_state,
        shutdown_event,
        update_rate=15.0,
        grpc_port: int = 9876,
        # Accumulated point cloud (back-projected per-frame depth → world voxels).
        accumulate_points: bool = True,
        voxel_size_m: float = 0.05,
        max_voxels: int = 500_000,
        pixel_stride: int = 2,
    ):
        super().__init__(name="SceneGraphRerunVisualizer", daemon=True)
        self.shared_state = shared_state
        self.shutdown_event = shutdown_event
        self.update_rate = update_rate
        self.trajectory = deque(maxlen=5000)
        self.last_pos = None

        self.grpc_port = int(grpc_port)

        self.accumulate_points = bool(accumulate_points)
        self.voxel_size_m = float(voxel_size_m)
        self.max_voxels = int(max_voxels)
        self.pixel_stride = max(1, int(pixel_stride))
        # voxel_idx (ix, iy, iz) -> uint8 RGB (3,)
        self._voxels: dict = {}
        self._last_cam_ts: dict = {}

    def _label_to_color(self, label):
        h = (hash(label) % 1000) / 1000
        r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
        return [int(r * 255), int(g * 255), int(b * 255)]

    def _accumulate_depth_points(self, cameras: dict) -> bool:
        """Back-project each new CameraFrame's depth → world voxels.

        Mutates ``self._voxels`` in place. Returns True if any voxel changed
        (signal for the caller to re-log the cumulative cloud)."""
        if not cameras:
            return False
        v = self.voxel_size_m
        stride = self.pixel_stride
        changed = False
        for cam_name, cam in cameras.items():
            ts = float(getattr(cam, "timestamp", 0.0))
            if self._last_cam_ts.get(cam_name) == ts:
                continue
            self._last_cam_ts[cam_name] = ts

            depth = getattr(cam, "depth", None)
            rgb = getattr(cam, "rgb", None)
            K = getattr(cam, "intrinsics", None)
            T_world_cam = getattr(cam, "T_world_cam", None)
            if depth is None or K is None or T_world_cam is None:
                continue

            # Subsample first → cheap unprojection.
            d = depth[::stride, ::stride]
            if rgb is not None and rgb.shape[:2] == depth.shape:
                c = rgb[::stride, ::stride]
            elif rgb is not None and rgb.shape[:2] == d.shape:
                c = rgb
            else:
                c = None

            mask = np.isfinite(d) & (d > 0.1) & (d < 9.0)
            if not mask.any():
                continue
            vs, us = np.where(mask)
            # Recompute fx/fy/cx/cy on the subsampled grid.
            fx = float(K[0, 0]) / stride
            fy = float(K[1, 1]) / stride
            cx = float(K[0, 2]) / stride
            cy = float(K[1, 2]) / stride
            z = d[vs, us].astype(np.float64)
            x_cam = (us - cx) * z / fx
            y_cam = (vs - cy) * z / fy
            pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)
            pts_world = (T_world_cam @ pts_cam.T).T[:, :3]
            colors = (
                c[vs, us].astype(np.uint8)
                if c is not None
                else np.full((pts_world.shape[0], 3), 200, dtype=np.uint8)
            )

            ix = np.round(pts_world[:, 0] / v).astype(np.int32)
            iy = np.round(pts_world[:, 1] / v).astype(np.int32)
            iz = np.round(pts_world[:, 2] / v).astype(np.int32)
            for k in range(ix.size):
                key = (int(ix[k]), int(iy[k]), int(iz[k]))
                if key not in self._voxels:
                    self._voxels[key] = colors[k]
                    changed = True

        # Cap voxel count: drop ~10% oldest (dict preserves insertion order).
        if len(self._voxels) > self.max_voxels:
            n_drop = max(1, len(self._voxels) - int(self.max_voxels * 0.9))
            for k in list(self._voxels.keys())[:n_drop]:
                self._voxels.pop(k, None)
            changed = True
        return changed

    def _log_accumulated_points(self):
        if not self._voxels:
            return
        v = self.voxel_size_m
        keys = np.fromiter(
            (c for k in self._voxels for c in k), dtype=np.int32, count=len(self._voxels) * 3
        ).reshape(-1, 3)
        pts = keys.astype(np.float32) * v
        cols = np.stack(list(self._voxels.values()), axis=0).astype(np.uint8)
        # static=True: viewer keeps only the latest snapshot.
        rr.log(
            "world/cumulative_points",
            rr.Points3D(positions=pts, colors=cols, radii=v * 0.5),
            static=True,
        )

    def run(self):
        print("SceneGraphRerunVisualizer starting...")

        rr.init("scene_graph", spawn=False)
        try:
            # Rerun 0.20+ uses gRPC; URL format: rerun+http://host:port/proxy
            _connect = getattr(rr, "connect_grpc", None) or getattr(rr, "connect", None)
            if _connect is not None:
                _connect(f"rerun+http://127.0.0.1:{self.grpc_port}/proxy")
                print(f"[RR] Connected to Rerun viewer at 127.0.0.1:{self.grpc_port}")
            else:
                print("[RR] No connect method found in rerun SDK")
        except Exception as e:
            print(f"[RR] Could not connect to Rerun viewer: {e}")

        # Ground grid
        grid_lines = []
        for i in range(-20, 21, 2):
            grid_lines.append([[i, -20, 0], [i, 20, 0]])
            grid_lines.append([[-20, i, 0], [20, i, 0]])
        rr.log(
            "world/grid",
            rr.LineStrips3D(grid_lines, colors=[[60, 60, 60]]),
            static=True,
        )

        print("Rerun viewer opened")

        while not self.shutdown_event.is_set():
            t0 = time.time()

            try:
                with self.shared_state.lock:
                    odom = self.shared_state.sensor.latest_odom
                    sg = self.shared_state.scenegraph.scene_graph
                    det_result = getattr(self.shared_state.perception, "detection_result", None)
                    # Snapshot camera frames under the lock; re-projection happens outside.
                    cameras_snapshot = (
                        dict(self.shared_state.sensor.latest_cameras)
                        if self.accumulate_points
                        else {}
                    )

                if self.accumulate_points:
                    if self._accumulate_depth_points(cameras_snapshot):
                        self._log_accumulated_points()

                # Camera RGB + detections (bboxes in detector-input image space)
                if det_result is not None:
                    for cam_name, cam_det in getattr(det_result, "camera_results", {}).items():
                        img = cam_det.rgb_image_detector_input
                        dets = cam_det.detections_detector_input
                        if img is None:
                            img = cam_det.rgb_image
                            dets = cam_det.detections
                        if img is not None:
                            rr.log(f"camera/{cam_name}/rgb", rr.Image(img))
                        if dets:
                            mins = [[d.bbox[0], d.bbox[1]] for d in dets]
                            sizes = [[d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]] for d in dets]
                            labels = [f"{d.name} {d.confidence:.2f}" for d in dets]
                            rr.log(
                                f"camera/{cam_name}/detections",
                                rr.Boxes2D(
                                    mins=mins,
                                    sizes=sizes,
                                    labels=labels,
                                    colors=[[0, 255, 0]] * len(mins),
                                ),
                            )

                # Robot
                if odom is not None:
                    x, y, yaw = odom

                    theta = np.linspace(0, 2 * np.pi, 32)
                    circle = [[x + 0.3 * np.cos(t), y + 0.3 * np.sin(t), 0.05] for t in theta]
                    circle.append(circle[0])
                    rr.log(
                        "world/robot/circle",
                        rr.LineStrips3D([circle], colors=[[255, 0, 0]], radii=[0.02]),
                    )

                    # Heading arrow
                    rr.log(
                        "world/robot/arrow",
                        rr.Arrows3D(
                            origins=[[x, y, 0.05]],
                            vectors=[[0.5 * np.cos(yaw), 0.5 * np.sin(yaw), 0]],
                            colors=[[255, 0, 0]],
                        ),
                    )

                    # Trajectory
                    if (
                        self.last_pos is None
                        or np.hypot(x - self.last_pos[0], y - self.last_pos[1]) > 0.05
                    ):
                        self.trajectory.append([x, y, 0.02])
                        self.last_pos = [x, y]

                    if len(self.trajectory) >= 2:
                        rr.log(
                            "world/trajectory",
                            rr.LineStrips3D([list(self.trajectory)], colors=[[0, 200, 255]]),
                        )

                # Objects - show both confirmed and unconfirmed
                if sg is not None and len(sg.nodes) > 0:
                    confirmed = [n for n in sg.nodes if n.is_confirmed]
                    unconfirmed = [n for n in sg.nodes if not n.is_confirmed]

                    # Confirmed nodes - solid colors
                    if confirmed:
                        centers = []
                        sizes = []
                        colors = []
                        labels = []
                        for node in confirmed:
                            bmin, bmax = node.bbox_3d[0], node.bbox_3d[1]
                            centers.append(((bmin + bmax) / 2).tolist())
                            sizes.append((bmax - bmin).tolist())
                            node_label = getattr(
                                node, "chosen_label", getattr(node, "label", "unknown")
                            )
                            colors.append(self._label_to_color(node_label))
                            labels.append(f"{node.node_id}:{node_label}")

                        rr.log(
                            "world/objects/confirmed",
                            rr.Boxes3D(
                                centers=centers,
                                sizes=sizes,
                                colors=colors,
                                labels=labels,
                            ),
                        )
                    else:
                        rr.log("world/objects/confirmed", rr.Clear(recursive=False))

                    # Unconfirmed nodes - semi-transparent gray
                    if unconfirmed:
                        centers = []
                        sizes = []
                        colors = []
                        labels = []
                        for node in unconfirmed:
                            bmin, bmax = node.bbox_3d[0], node.bbox_3d[1]
                            centers.append(((bmin + bmax) / 2).tolist())
                            sizes.append((bmax - bmin).tolist())
                            node_label = getattr(
                                node, "chosen_label", getattr(node, "label", "unknown")
                            )
                            # Dim color with alpha for unconfirmed nodes
                            base_color = self._label_to_color(node_label)
                            dim_color = [c // 2 for c in base_color]
                            colors.append(dim_color + [100])
                            labels.append(f"?{node.node_id}:{node_label}")

                        rr.log(
                            "world/objects/unconfirmed",
                            rr.Boxes3D(
                                centers=centers,
                                sizes=sizes,
                                colors=colors,
                                labels=labels,
                            ),
                        )
                    else:
                        rr.log("world/objects/unconfirmed", rr.Clear(recursive=False))

            except Exception as e:
                print(f"Rerun error: {e}")

            elapsed = time.time() - t0
            time.sleep(max(0, 1.0 / self.update_rate - elapsed))

        print("SceneGraphRerunVisualizer stopped")
