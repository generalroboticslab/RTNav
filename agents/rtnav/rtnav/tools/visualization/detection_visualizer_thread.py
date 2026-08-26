import asyncio
import base64
import os
import threading
import time
from types import SimpleNamespace
from typing import Any, List, Optional, Set, Tuple

import cv2
import numpy as np

_DETECTION_COLOR = (245, 135, 66)  # Blue in BGR.
_TARGET_COLOR = (65, 80, 245)  # Red in BGR.
_DETECTION_BOX_THICKNESS = 2
_DETECTION_LABEL_FONT_SCALE = 0.5
_DETECTION_LABEL_THICKNESS = 1
_DETECTION_LABEL_PADDING = 3


def _get_detector_input_view(
    rgb: np.ndarray, detections: List[Any]
) -> Tuple[np.ndarray, List[Any]]:
    """
    Fallback detector-input view when old detection results lack the stored
    input image. Current detector inputs preserve source aspect ratio; old
    results only have original-frame detections, so keep those coordinates.
    """
    return _as_rgb3(rgb), list(detections or [])


def _as_rgb3(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim != 3:
        raise RuntimeError(f"Expected HxW or HxWxC image, got shape={arr.shape}")
    else:
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] != 3:
            raise RuntimeError(f"Expected 1, 3, or 4 image channels, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        scale = (
            255.0 if np.issubdtype(arr.dtype, np.floating) and arr.size and arr.max() <= 1 else 1.0
        )
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _with_scaled_bbox(det: Any, sx: float, sy: float) -> Any:
    x1, y1, x2, y2 = det.bbox
    attrs = {
        "name": det.name,
        "confidence": det.confidence,
        "bbox": (x1 * sx, y1 * sy, x2 * sx, y2 * sy),
    }
    for attr in ("location", "embeddings"):
        if hasattr(det, attr):
            attrs[attr] = getattr(det, attr)
    return SimpleNamespace(**attrs)


def _scale_detections(detections: List[Any], sx: float, sy: float) -> List[Any]:
    if sx == 1.0 and sy == 1.0:
        return list(detections or [])
    for det in detections or []:
        if not hasattr(det, "bbox"):
            raise RuntimeError(f"Detection missing bbox: {det!r}")
    return [_with_scaled_bbox(det, sx, sy) for det in detections or []]


def _resize_detection_panel_with_detections(
    img: np.ndarray, detections: List[Any], display_height: int
) -> Tuple[np.ndarray, List[Any]]:
    """Downscale for display and apply the same scale to detection boxes."""
    img = _as_rgb3(img)
    h, w = img.shape[:2]
    target_height = int(display_height or 0)
    if target_height <= 0 or target_height >= h:
        return img, list(detections or [])
    target_width = max(int(round(w * target_height / h)), 1)
    sx = target_width / w
    sy = target_height / h
    resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, _scale_detections(detections, sx, sy)


def _resize_detection_panel(img: np.ndarray, display_height: int) -> np.ndarray:
    """Downscale for display without changing aspect ratio or upscaling."""
    return _resize_detection_panel_with_detections(img, [], display_height)[0]


def _pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    canvas = np.zeros((height, w, 3), dtype=img.dtype)
    y = (height - h) // 2
    canvas[y : y + h, :w] = img
    return canvas


def _hstack_same_height(imgs: List[np.ndarray]) -> np.ndarray:
    if not imgs:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    height = max(img.shape[0] for img in imgs)
    return np.hstack([_pad_to_height(img, height) for img in imgs])


class WebDetectionVisualizerThread(threading.Thread):
    """Web visualization of detections via WebSocket streaming.

    Doubles as the agent-side video recorder: every rendered frame is also
    appended to a per-episode buffer that gets written to ``video.mp4`` on
    :meth:`save_episode`. The runner wires its video lifecycle hooks here
    when this thread is active.
    """

    def __init__(
        self,
        shared_state,
        shutdown_event: threading.Event,
        port: int = 8765,
        jpeg_quality: int = 80,
        update_rate: int = 10,
        display_height: int = 0,
        video_fps: int = 10,
        video_max_frames: int = 6000,
    ):
        threading.Thread.__init__(self, name="WebDetectionVisualizerThread", daemon=True)
        self.shared_state = shared_state
        self.shutdown_event = shutdown_event
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.update_rate = update_rate
        self.display_height = display_height
        self._clients: Set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Per-episode video recording (driven by runner via new_episode/save_episode).
        self._video_fps = video_fps
        self._video_max_frames = video_max_frames
        self._video_frames: List[np.ndarray] = []
        self._video_output_dir: Optional[str] = None
        self._video_lock = threading.Lock()

    def new_episode(self, scene_id: str, episode_id: str, output_dir: Optional[str] = None):
        """Start a fresh per-episode buffer. Called by the runner at episode start."""
        with self._video_lock:
            self._video_frames = []
            self._video_output_dir = output_dir

    def save_episode(self, metrics: Optional[dict] = None) -> Optional[str]:
        """Flush the per-episode buffer to ``{output_dir}/video.mp4`` in a
        background thread so the eval loop never waits on mp4 encoding.
        """
        with self._video_lock:
            frames = self._video_frames
            out_dir = self._video_output_dir
            self._video_frames = []
        if not frames or not out_dir:
            return None
        from video_recorder import _write_mp4

        path = os.path.join(out_dir, "video.mp4")
        threading.Thread(
            target=_write_mp4,
            args=(frames, path, self._video_fps),
            daemon=True,
            name=f"det_viz_save_{os.path.basename(out_dir)}",
        ).start()
        return path

    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            print(f"[WebDetectionViz] Server error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self._loop.close()

    async def _serve(self):
        """Start aiohttp server with HTTP and WebSocket support."""
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/", self._handle_http)
        app.router.add_get("/ws", self._handle_websocket)

        runner = web.AppRunner(app)
        await runner.setup()

        # Probe [port, port+15] for a free one — matches the other web viz
        # threads and lets multiple parallel workers each grab a port.
        site = None
        for candidate in range(self.port, self.port + 16):
            try:
                site = web.TCPSite(runner, "0.0.0.0", candidate)
                await site.start()
                if candidate != self.port:
                    print(f"[WebDetectionViz] port {self.port} in use → fell back to {candidate}")
                self.port = candidate
                break
            except OSError as e:
                if e.errno != 98:  # EADDRINUSE
                    raise
                site = None
        if site is None:
            raise OSError(f"[WebDetectionViz] no free port in {self.port}..{self.port + 15}")

        print(f"[WebDetectionViz] Server running on http://0.0.0.0:{self.port}")

        frame_interval = 1.0 / self.update_rate
        last_render = time.time()
        while not self.shutdown_event.is_set():
            now = time.time()
            if now - last_render >= frame_interval:
                await self._broadcast_frame()
                last_render = now
            else:
                await asyncio.sleep(0.005)

        await runner.cleanup()

    async def _handle_http(self, request):
        """Serve HTML page."""
        from aiohttp import web

        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Detection Visualizer</title></head>
        <body style="margin:0; display:flex; justify-content:center; align-items:center; height:100vh; background:#000;">
            <img id="frame" style="max-width:90%; max-height:90%; border:2px solid #0f0;" />
            <script>
                const ws = new WebSocket("ws://" + location.host + "/ws");
                ws.binaryType = "arraybuffer";
                ws.onmessage = (e) => {
                    document.getElementById("frame").src = "data:image/jpeg;base64," + e.data;
                };
                ws.onerror = () => { console.log("WebSocket error"); };
            </script>
        </body>
        </html>
        """
        return web.Response(body=html, content_type="text/html")

    async def _handle_websocket(self, request):
        """Handle WebSocket connection."""
        from aiohttp import web

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._clients.add(ws)
        print(f"[WebDetectionViz] Client connected ({len(self._clients)} total)")

        try:
            async for _ in ws:
                pass
        finally:
            self._clients.discard(ws)
            print(f"[WebDetectionViz] Client disconnected ({len(self._clients)} total)")

        return ws

    def _draw_detections_on_image(self, rgb_img, detections, target_labels=None):
        """Draw blue boxes, using red for the current target and its aliases."""
        img = rgb_img.copy()
        h, w = img.shape[:2]
        target_labels = {str(label).strip().lower() for label in target_labels or ()}

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            x1 = int(np.clip(round(x1), 0, max(0, w - 1)))
            y1 = int(np.clip(round(y1), 0, max(0, h - 1)))
            x2 = int(np.clip(round(x2), 0, max(0, w - 1)))
            y2 = int(np.clip(round(y2), 0, max(0, h - 1)))
            if x2 <= x1 or y2 <= y1:
                continue
            color = (
                _TARGET_COLOR
                if str(detection.name).strip().lower() in target_labels
                else _DETECTION_COLOR
            )

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color,
                _DETECTION_BOX_THICKNESS,
            )

            label_text = f"{detection.name} {detection.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                _DETECTION_LABEL_FONT_SCALE,
                _DETECTION_LABEL_THICKNESS,
            )

            label_height = text_h + baseline + 2 * _DETECTION_LABEL_PADDING
            label_width = text_w + 2 * _DETECTION_LABEL_PADDING
            label_x1 = min(x1, max(0, w - label_width))
            label_x2 = min(w - 1, label_x1 + label_width)
            if y1 >= label_height:
                label_y1 = y1 - label_height
                label_y2 = y1
            elif y2 + label_height < h:
                label_y1 = y2
                label_y2 = y2 + label_height
            else:
                label_y1 = y1
                label_y2 = min(h - 1, y1 + label_height)

            cv2.rectangle(
                img,
                (label_x1, label_y1),
                (label_x2, label_y2),
                color,
                -1,
            )

            cv2.putText(
                img,
                label_text,
                (
                    label_x1 + _DETECTION_LABEL_PADDING,
                    label_y1 + _DETECTION_LABEL_PADDING + text_h,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                _DETECTION_LABEL_FONT_SCALE,
                (0, 0, 0),
                _DETECTION_LABEL_THICKNESS,
                cv2.LINE_AA,
            )

        return img

    def _create_detection_grid(self, detection_result, target_labels=None):
        """Create a grid showing all camera views with detections."""
        if detection_result is None or not detection_result.camera_results:
            return None

        camera_views = []
        camera_labels = []
        cam_names = sorted(detection_result.camera_results.keys())

        for cam_name in cam_names:
            cam_result = detection_result.camera_results[cam_name]

            if getattr(cam_result, "rgb_image_detector_input", None) is not None:
                cropped_rgb = cam_result.rgb_image_detector_input
                crop_detections = getattr(cam_result, "detections_detector_input", None) or []
            else:
                rgb = cam_result.rgb_image
                detections = cam_result.detections
                if rgb is None:
                    continue
                cropped_rgb, crop_detections = _get_detector_input_view(rgb, detections)

            vis_img = cv2.cvtColor(_as_rgb3(cropped_rgb), cv2.COLOR_RGB2BGR)
            vis_img, crop_detections = _resize_detection_panel_with_detections(
                vis_img, crop_detections, self.display_height
            )
            if len(crop_detections) > 0:
                vis_img = self._draw_detections_on_image(vis_img, crop_detections, target_labels)

            target_width = vis_img.shape[1]

            detections = cam_result.detections
            cam_label = f"{cam_name}"
            if len(detections) > 0:
                cam_label += f" ({len(detections)})"

            camera_views.append(vis_img)
            camera_labels.append((cam_label, target_width))

        if len(camera_views) == 0:
            return None

        num_cams = len(camera_views)
        cams_per_row = (num_cams + 1) // 2

        row1_views = camera_views[:cams_per_row]
        row1_labels = camera_labels[:cams_per_row]
        row2_views = camera_views[cams_per_row:]
        row2_labels = camera_labels[cams_per_row:]

        row1_combined = _hstack_same_height(row1_views) if row1_views else None

        if row2_views:
            row2_combined = _hstack_same_height(row2_views)
            if row1_combined is not None and row2_combined.shape[1] < row1_combined.shape[1]:
                pad_width = row1_combined.shape[1] - row2_combined.shape[1]
                padding = np.zeros((row2_combined.shape[0], pad_width, 3), dtype=np.uint8)
                row2_combined = np.hstack([row2_combined, padding])
        else:
            row2_combined = None

        label_height = 40
        panel_height = max(v.shape[0] for v in camera_views)
        font_scale = panel_height / 360 * 0.7
        thickness = max(1, int(panel_height / 360 * 2))

        def create_label_panel(labels, width):
            panel = np.zeros((label_height, width, 3), dtype=np.uint8)
            panel[:, :] = (0, 0, 0)

            x_offset = 0
            for cam_label, cam_width in labels:
                (text_w, text_h), baseline = cv2.getTextSize(
                    cam_label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
                )
                text_x = x_offset + (cam_width - text_w) // 2
                text_y = (label_height + text_h) // 2

                cv2.putText(
                    panel,
                    cam_label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )
                x_offset += cam_width
            return panel

        row1_label = create_label_panel(row1_labels, row1_combined.shape[1])
        row1_with_label = np.vstack([row1_combined, row1_label])

        if row2_combined is not None:
            row2_label = create_label_panel(row2_labels, row2_combined.shape[1])
            row2_with_label = np.vstack([row2_combined, row2_label])
            final_grid = np.vstack([row1_with_label, row2_with_label])
        else:
            final_grid = row1_with_label

        return final_grid

    async def _broadcast_frame(self):
        """Render and send frame to all clients."""
        if not self._clients:
            return

        with self.shared_state.lock:
            detection_result = self.shared_state.perception.detection_result
            target_labels = set(getattr(self.shared_state.task, "synonym_to_canonical", {}) or {})

        if detection_result is None:
            return

        img = self._create_detection_grid(detection_result, target_labels)

        if img is None:
            return

        cv2.circle(img, (img.shape[1] - 20, 20), 8, (0, 0, 255), -1)
        cv2.putText(
            img, "LIVE", (img.shape[1] - 60, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )

        # Append to per-episode video buffer (runner flushes on episode end).
        # imageio expects RGB but the grid is BGR (OpenCV native); convert.
        with self._video_lock:
            if self._video_output_dir is not None:
                self._video_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if len(self._video_frames) > self._video_max_frames:
                    self._video_frames = self._video_frames[-self._video_max_frames :]

        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        frame_data = base64.b64encode(buffer).decode("utf-8")

        dead = set()
        for client in self._clients.copy():
            try:
                await client.send_str(frame_data)
            except Exception:
                dead.add(client)
        self._clients -= dead
