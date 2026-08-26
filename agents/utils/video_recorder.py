"""Episode video recorders for ROS2 distributed evaluation.

EnvVideoRecorder:              one frame per Habitat step.
AgentVideoRecorder:            one frame per agent decision.
AgentObservationVideoRecorder: one frame per streamed observation.

Set VIDEO_DIR env var to enable saving; empty/unset = no-op unless display mode
is requested.
"""

import os
import threading
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


# ── Compositing helpers ──────────────────────────────────────────────────


def _write_mp4(frames: List[np.ndarray], path: str, fps: int = 10):
    if not frames:
        return
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)
    h += h % 2  # ensure even
    w += w % 2
    writer = imageio.get_writer(
        path, format="FFMPEG", fps=fps, quality=5, macro_block_size=1,
        # ultrafast preset: ~3-5x faster encode, ~30% larger files.
        # Worth it since save runs off the env's hot thread anyway.
        output_params=["-preset", "ultrafast"],
    )
    try:
        for f in frames:
            if f.shape[0] != h or f.shape[1] != w:
                f = _letterbox_to(f, h, w)
            f = np.ascontiguousarray(f[: h - h % 2, : w - w % 2])
            writer.append_data(f)
    finally:
        writer.close()


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    d = np.clip(depth, 0, None)
    if d.max() > 0:
        d = d / d.max()
    return cv2.cvtColor((d * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


def _to_rgb_u8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    img = _to_rgb_u8(img)
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _letterbox_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    img = _to_rgb_u8(img)
    ih, iw = img.shape[:2]
    if ih <= 0 or iw <= 0:
        return np.zeros((h, w, 3), np.uint8)
    scale = min(1.0, h / ih, w / iw)
    new_h = max(int(round(ih * scale)), 1)
    new_w = max(int(round(iw * scale)), 1)
    if (new_h, new_w) != (ih, iw):
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y = (h - new_h) // 2
    x = (w - new_w) // 2
    canvas[y : y + new_h, x : x + new_w] = img[:new_h, :new_w]
    return canvas


def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    return cv2.resize(
        img, (max(int(img.shape[1] * h / img.shape[0]), 1), h),
        interpolation=cv2.INTER_AREA,
    )


def _resize_w(img: np.ndarray, w: int) -> np.ndarray:
    if img.shape[1] == w:
        return img
    return cv2.resize(
        img, (w, max(int(img.shape[0] * w / img.shape[1]), 1)),
        interpolation=cv2.INTER_AREA,
    )


def _overlay(img: np.ndarray, lines: List[str], scale: float = 0.4) -> np.ndarray:
    out = img.copy()
    for i, line in enumerate(lines):
        cv2.putText(
            out, line, (4, 14 + 16 * i),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


def _hstack(imgs: List[np.ndarray]) -> np.ndarray:
    if not imgs:
        return np.zeros((100, 100, 3), np.uint8)
    h = max(i.shape[0] for i in imgs)
    return np.hstack([_resize_h(i, h) for i in imgs])


def _vstack(imgs: List[np.ndarray]) -> np.ndarray:
    if not imgs:
        return np.zeros((100, 100, 3), np.uint8)
    w = max(i.shape[1] for i in imgs)
    return np.vstack([_resize_w(i, w) for i in imgs])


# ── Base recorder ────────────────────────────────────────────────────────


class _BaseRecorder:
    """Per-episode video recorder.

    Frames are streamed straight to an mp4 via imageio as soon as
    ``record_step()`` produces them — there is **no in-memory frame
    buffer**. The previous design (``self._frames.append(...)`` then a
    bulk encode at ``save_episode`` time) held every recorded frame as a
    numpy array for the full episode; at HD resolution × 30 Hz × a few
    hundred seconds this added up to ~100 GB per episode of unreclaimed
    RSS (the per-callback ``np.hstack`` was actually the bulk of the
    leak we hunted to via tracemalloc). Streaming removes both the
    growing buffer and the periodic glibc-fragmenting bulk alloc.

    Two output layouts are supported:

      - **Explicit episode folder**: caller passes ``output_dir`` to
        :meth:`new_episode` and the file lands at ``output_dir/video.mp4``.

      - **Flat layout**: when ``output_dir`` is omitted, the file is
        written to a temp path during the episode, then renamed at
        :meth:`save_episode` time with success/spl in the filename.

    The first ``record_step`` locks in the frame dimensions; later
    frames are resized to match (with the same even-dim handling
    ``_write_mp4`` used to do).
    """

    def __init__(self, video_dir: Optional[str], prefix: str, fps: int = 10):
        self._dir = video_dir or ""
        self._prefix = prefix
        self._fps = fps
        self._ep_id = self._scene_id = ""
        self._episode_output_dir: Optional[str] = None
        self._initial_heading: Optional[float] = None
        self._initial_pitch: Optional[float] = None
        # Streaming writer state.
        self._writer: Optional[Any] = None
        self._writer_path: Optional[str] = None     # final path or temp path
        self._writer_target_path: Optional[str] = None  # path to rename to on save
        self._writer_hw: Optional[tuple] = None     # (h, w) locked at first frame
        self._writer_frame_count: int = 0
        self._episode_active: bool = False
        # Disabled = stop trying to open/append after a failure. Warned =
        # we already printed the reason once, suppress further spam.
        self._writer_disabled: bool = False
        self._writer_warned: bool = False

    @property
    def enabled(self) -> bool:
        # Enabled when EITHER VIDEO_DIR env var is set (legacy flat layout)
        # OR new_episode was called with an explicit output_dir (per-episode).
        return bool(self._episode_active and (self._dir or self._episode_output_dir))

    def new_episode(
        self, scene_id: str, episode_id: str, output_dir: Optional[str] = None,
    ):
        if self._writer is not None:
            self.save_episode()
        self._scene_id = os.path.basename(scene_id).replace(".glb", "")
        self._ep_id = episode_id
        self._episode_output_dir = output_dir
        self._initial_heading = None
        self._initial_pitch = None
        self._writer_hw = None
        self._writer_frame_count = 0
        # Re-enable recording for the new episode. If the previous episode
        # failed due to e.g. a transient FS error, give it another shot.
        self._episode_active = True
        self._writer_disabled = False
        # Keep _writer_warned True across episodes so we don't spam the
        # same disk-full / permission warning every episode.

    def _open_writer_if_needed(self, frame: np.ndarray) -> None:
        """Open the imageio writer on the first frame, locking dimensions.

        Any failure here (permission denied on the output dir, missing
        ffmpeg, imageio import error, etc.) disables recording for this
        episode rather than propagating — the recorder runs on the ROS
        spin thread, and an uncaught exception there kills the executor.
        """
        if self._writer is not None or self._writer_disabled or not self.enabled:
            return
        try:
            h, w = frame.shape[:2]
            h += h % 2
            w += w % 2
            self._writer_hw = (h, w)
            if self._episode_output_dir:
                target_path = os.path.join(self._episode_output_dir, "video.mp4")
                self._writer_path = target_path
            else:
                target_dir = self._dir
                tmp = f".{self._prefix}_ep{self._ep_id}_sc{self._scene_id}.partial.mp4"
                self._writer_path = os.path.join(target_dir, tmp)
            target_dir = os.path.dirname(self._writer_path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            import imageio.v2 as imageio
            self._writer = imageio.get_writer(
                self._writer_path, format="FFMPEG", fps=self._fps,
                quality=5, macro_block_size=1,
                output_params=["-preset", "ultrafast"],
            )
        except Exception as exc:
            self._writer = None
            self._writer_path = None
            self._writer_hw = None
            if not self._writer_warned:
                self._writer_warned = True
                print(
                    f"[VideoRecorder] disabling video recording — "
                    f"could not open writer at {self._writer_path!r}: {exc}"
                )
            self._writer_disabled = True

    def _append_frame(self, frame: np.ndarray) -> None:
        """Stream one frame to the open writer, resizing to locked HxW.

        Wrapped so the ROS-thread callback never raises out of this layer.
        """
        if not self.enabled or self._writer_disabled:
            return
        try:
            self._open_writer_if_needed(frame)
            if self._writer is None or self._writer_hw is None:
                return
            h, w = self._writer_hw
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = _letterbox_to(frame, h, w)
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            self._writer.append_data(frame)
            self._writer_frame_count += 1
        except Exception as exc:
            if not self._writer_warned:
                self._writer_warned = True
                print(f"[VideoRecorder] append_data failed, disabling: {exc}")
            self._writer_disabled = True
            self._close_writer()

    def _close_writer(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.close()
        except Exception:
            pass
        self._writer = None

    def save_episode(self, metrics: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Close the streaming writer and (for flat layout) rename to the
        metrics-tagged final filename. Returns the saved path, or None if
        nothing was recorded.
        """
        if self._writer is None:
            self._episode_active = False
            return None
        if not (self._dir or self._episode_output_dir):
            self._close_writer()
            self._episode_active = False
            return None
        partial_path = self._writer_path
        frame_count = self._writer_frame_count
        self._close_writer()
        self._episode_active = False
        if frame_count == 0:
            # Nothing recorded — clean up empty file.
            try:
                if partial_path and os.path.exists(partial_path):
                    os.remove(partial_path)
            except Exception:
                pass
            return None
        if self._episode_output_dir:
            # Already at final path.
            return partial_path
        # Flat layout — rename temp file with success/spl suffix.
        s = int(metrics.get("success", 0)) if metrics else 0
        spl = float(metrics.get("spl", 0)) if metrics else 0.0
        final_path = os.path.join(
            self._dir,
            f"{self._prefix}_ep{self._ep_id}_sc{self._scene_id}_succ{s}_spl{spl:.3f}.mp4",
        )
        try:
            os.replace(partial_path, final_path)
        except Exception as exc:
            print(f"[VideoRecorder] rename {partial_path}->{final_path} failed: {exc}")
            return partial_path
        return final_path

    def _pose_lines(self, obs: dict) -> List[str]:
        heading = _obs_scalar(obs.get("compass"))
        pitch = _obs_scalar(obs.get("camera_pitch"))
        if heading is None:
            heading = _obs_scalar(obs.get("heading"))
        if heading is None and pitch is None:
            return []
        if heading is not None and self._initial_heading is None:
            self._initial_heading = heading
        if pitch is not None and self._initial_pitch is None:
            self._initial_pitch = pitch

        lines = []
        if heading is not None and self._initial_heading is not None:
            rot = _angle_diff(heading, self._initial_heading)
            lines.append(f"rot: {np.degrees(rot):+.1f} deg")
        if pitch is not None and self._initial_pitch is not None:
            lines.append(f"tilt: {np.degrees(pitch - self._initial_pitch):+.1f} deg")
        return lines


def _obs_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[0])


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > np.pi:
        d -= 2 * np.pi
    while d < -np.pi:
        d += 2 * np.pi
    return d


# ── Environment-side recorder ────────────────────────────────────────────


class EnvVideoRecorder(_BaseRecorder):
    """Records RGB + depth + top-down map per Habitat step.

    fps defaults to 30 to match the sim tick rate — env_node calls
    record_step() on every step, so encoding at the sim rate gives
    real-time playback. Drop to 10 for a smaller file at 3× slow-motion.
    """

    def __init__(self, video_dir: Optional[str], fps: int = 30):
        super().__init__(video_dir, "env", fps)

    def record_step(self, obs: dict, info: dict, step_id: int):
        if not self.enabled:
            return
        rgb = obs["rgb"]
        depth_rgb = _depth_to_rgb(obs["depth"])

        # Top-down map
        tdm = info.get("top_down_map")
        if tdm is not None:
            try:
                from habitat.utils.visualizations import maps
                map_img = maps.colorize_draw_agent_and_fit_to_height(
                    tdm, rgb.shape[0] * 2
                )
            except Exception:
                map_img = np.ones(
                    (rgb.shape[0] * 2, rgb.shape[0] * 2, 3), np.uint8
                ) * 200
        else:
            map_img = np.ones(
                (rgb.shape[0] * 2, rgb.shape[0] * 2, 3), np.uint8
            ) * 200

        # Metrics overlay on depth
        lines = [f"step: {step_id}", *self._pose_lines(obs)]
        for k in ("distance_to_goal", "success", "spl", "soft_spl"):
            if k in info:
                try:
                    lines.append(f"{k}: {float(info[k]):.3f}")
                except (TypeError, ValueError):
                    pass

        left = _vstack([rgb, _overlay(depth_rgb, lines)])
        self._append_frame(_hstack([left, map_img]))


# ── Agent-side recorder ──────────────────────────────────────────────────


class AgentVideoRecorder(_BaseRecorder):
    """Records annotated views + obstacle/value maps per agent decision."""

    def __init__(self, video_dir: Optional[str], fps: int = 10):
        super().__init__(video_dir, "agent", fps)

    def record_step(self, obs: dict, policy_info: dict, step: int):
        if not self.enabled:
            return
        placeholder = np.ones((240, 320, 3), np.uint8) * 200

        def _get(k, fallback):
            return np.asarray(policy_info[k], np.uint8) if k in policy_info else fallback

        rgb = _get(
            "annotated_rgb",
            np.asarray(obs.get("rgb", placeholder), np.uint8),
        )
        depth = _get(
            "annotated_depth",
            _depth_to_rgb(obs["depth"]) if "depth" in obs else placeholder,
        )
        obs_map = _get("obstacle_map", np.ones_like(rgb) * 255)
        val_map = _get("value_map", np.ones_like(rgb) * 255)
        h, w = rgb.shape[:2]
        depth = _resize_to(depth, h, w)
        obs_map = _resize_to(obs_map, h, w)
        val_map = _resize_to(val_map, h, w)

        lines = [f"step: {step}", *self._pose_lines(obs)]
        target = policy_info.get("target_object", "")
        if target:
            lines.append(f"target: {target}")

        left = _vstack([_overlay(rgb, lines), depth])
        right = _vstack([obs_map, val_map])
        self._append_frame(_hstack([left, right]))


class AgentObservationVideoRecorder(_BaseRecorder):
    """Displays or saves raw agent observations with latest agent maps."""

    def __init__(self, video_dir: Optional[str], mode: str = "none", fps: int = 30):
        self._mode = mode
        self._warned_display = False
        super().__init__(video_dir if mode == "save" else "", "agent_obs", fps)

    @property
    def enabled(self) -> bool:
        if self._mode == "display":
            return True
        if self._mode == "save":
            return super().enabled
        return False

    def record_step(self, obs: dict, policy_info: dict, step: int):
        if not self.enabled:
            return
        placeholder = np.ones((240, 320, 3), np.uint8) * 200
        rgb = np.asarray(obs.get("rgb", placeholder), np.uint8)
        depth = _depth_to_rgb(obs["depth"]) if "depth" in obs else placeholder
        lines = [f"step: {step}", *self._pose_lines(obs)]

        if "map_panel" in policy_info:
            frame = self._single_map_frame(
                rgb,
                depth,
                policy_info["map_panel"],
                lines,
                policy_info,
            )
            if self._mode == "display":
                self._show(frame)
            else:
                self._append_frame(frame)
            return

        obs_map = self._map_image(policy_info, "obstacle_map", rgb)
        frontier_map = self._map_image(policy_info, "frontier_map", rgb)
        value_map = self._map_image(policy_info, "value_map", rgb)
        h, w = rgb.shape[:2]
        depth = _resize_to(depth, h, w)
        obs_map = _resize_to(obs_map, h, w)
        frontier_map = _resize_to(frontier_map, h, w)
        value_map = _resize_to(value_map, h, w)
        frame = _hstack(
            [
                _vstack([_overlay(rgb, lines), depth]),
                _vstack([obs_map, frontier_map, value_map]),
            ]
        )
        if self._mode == "display":
            self._show(frame)
        else:
            self._append_frame(frame)

    @staticmethod
    def _map_image(policy_info: dict, key: str, rgb: np.ndarray) -> np.ndarray:
        if key in policy_info:
            return np.asarray(policy_info[key], np.uint8)
        return np.ones_like(rgb) * 255

    def _single_map_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        map_img: np.ndarray,
        lines: List[str],
        policy_info: dict,
    ) -> np.ndarray:
        rgb = _to_rgb_u8(rgb)
        h, w = rgb.shape[:2]
        depth = _resize_to(depth, h, w)
        left = _vstack([_overlay(rgb, lines), depth])
        map_panel = _resize_h(_to_rgb_u8(map_img), left.shape[0])
        panels = [(left, "Observations"), (map_panel, "RtNav Map")]

        if "detection_panel" in policy_info:
            det_panel = _to_rgb_u8(policy_info["detection_panel"])
            if det_panel.shape[0] < left.shape[0]:
                det_panel = _resize_h(det_panel, left.shape[0])
            panels.append(
                (det_panel, str(policy_info.get("detection_title", "Detections")))
            )

        title_h = 42
        pad = 14
        gap = 16
        panel_h = max(panel.shape[0] for panel, _ in panels)
        canvas_h = title_h + panel_h + pad
        canvas_w = (
            pad
            + sum(p.shape[1] for p, _ in panels)
            + gap * (len(panels) - 1)
            + pad
        )
        canvas = np.ones((canvas_h, canvas_w, 3), np.uint8) * 255

        panel_y = title_h
        target = str(policy_info.get("target_object", "") or "").strip()
        if target:
            panels[0] = (panels[0][0], f"Observations (Goal: {target})")
        panels[1] = (panels[1][0], str(policy_info.get("map_title", "RtNav Map")))

        border = (100, 100, 100)
        x = pad
        for panel, title in panels:
            panel_w = panel.shape[1]
            canvas[panel_y:panel_y + panel.shape[0], x:x + panel_w] = panel
            self._draw_centered_title(canvas, title, x, panel_w, 27)
            cv2.rectangle(
                canvas,
                (x - 1, panel_y - 1),
                (x + panel_w, panel_y + panel.shape[0]),
                border,
                1,
            )
            x += panel_w + gap
        return canvas

    @staticmethod
    def _draw_centered_title(
        img: np.ndarray, text: str, x: int, width: int, baseline_y: int
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        thickness = 2
        max_w = max(width - 8, 1)
        text_w = cv2.getTextSize(text, font, scale, thickness)[0][0]
        if text_w > max_w:
            scale = max(0.35, scale * max_w / text_w)
            text_w = cv2.getTextSize(text, font, scale, thickness)[0][0]
        tx = x + max((width - text_w) // 2, 0)
        cv2.putText(
            img,
            text,
            (tx, baseline_y),
            font,
            scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )

    def _show(self, frame: np.ndarray):
        try:
            cv2.imshow("Agent observations", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        except cv2.error as exc:
            if not self._warned_display:
                print(f"[AgentVideo] display failed: {exc}")
                self._warned_display = True
