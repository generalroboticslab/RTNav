"""Agent-side video recorder: augments the BaseAgentNode video with rtnav
map + detection panels, rendered off the hot path on a worker thread.

The node owns the framework recorder (BaseAgentNode) and hands it here; this
class owns the frame queue, the panel-render worker, and the panel caches, and
delegates frame writes to that recorder. The node just forwards on_obs /
new_episode / save / reset.
"""

import os
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
from ros_helpers import DONE_SENTINEL, obs_msg_to_obs_dict


class AgentVideoRecorder:
    _VIDEO_MAP_PERIOD_S = 0.25

    def __init__(self, video_recorder, agent_getter):
        self._video_recorder = video_recorder
        self._agent_getter = agent_getter

        self._video_map_viz = None
        self._video_det_viz = None
        self._video_policy_info_cache: dict = {}
        self._video_policy_info_cache_t: float = 0.0
        self._video_policy_info_lock = threading.Lock()
        self._video_policy_info_rendering = False
        self._video_policy_info_generation = 0
        self._video_det_panel_cache = None
        self._video_det_panel_key = None
        self._video_frame_queue = deque()
        video_queue_default = os.environ.get("AGENT_VIDEO_QOS_DEPTH", "300")
        self._video_frame_queue_limit = max(
            1, int(os.environ.get("AGENT_VIDEO_QUEUE_SIZE", video_queue_default))
        )
        self._video_frame_cv = threading.Condition()
        self._video_frame_inflight = False
        self._video_recorder_lock = threading.Lock()
        self._frame_error_warned = False
        self._panel_error_warned = False
        self._shutdown_event = threading.Event()
        self._video_frame_worker = threading.Thread(
            target=self._frame_worker_loop,
            daemon=True,
            name="agent_video_frame_writer",
        )
        self._video_frame_worker.start()

    # ── public API (node delegates here) ─────────────────────────────────────
    def on_obs(self, msg):
        if msg.step_id == DONE_SENTINEL:
            return
        self._schedule_refresh()
        with self._video_policy_info_lock:
            policy_info = dict(self._video_policy_info_cache)
        if "map_panel" not in policy_info or "detection_panel" not in policy_info:
            return
        self._enqueue(msg, policy_info)

    def new_episode(self, scene_id, episode_id):
        self._drain()  # flush Episode N frames before switching to Episode N+1
        self.reset()
        with self._video_recorder_lock:
            self._video_recorder.new_episode(scene_id, episode_id)

    def save(self, timeout: float = 5.0):
        if not self._drain(timeout):
            print(
                "[agent_video] warning: timed out waiting for video frames — episode video may be incomplete"
            )
        with self._video_recorder_lock:
            self._video_recorder.save_episode()

    def shutdown(self) -> None:
        """Stop the frame writer thread cleanly."""
        self._shutdown_event.set()
        with self._video_frame_cv:
            self._video_frame_cv.notify_all()
        self._video_frame_worker.join(timeout=3.0)

    def reset(self) -> None:
        with self._video_policy_info_lock:
            self._video_policy_info_cache = {}
            self._video_policy_info_cache_t = 0.0
            self._video_policy_info_generation += 1
            self._video_det_panel_cache = None
            self._video_det_panel_key = None
            clear_viz = not self._video_policy_info_rendering
        self._frame_error_warned = False
        self._panel_error_warned = False
        if clear_viz:
            self._video_map_viz = None
            self._video_det_viz = None

    # ── frame queue + writer thread ──────────────────────────────────────────
    def _enqueue(self, msg, policy_info: dict) -> None:
        with self._video_frame_cv:
            if len(self._video_frame_queue) >= self._video_frame_queue_limit:
                self._video_frame_queue.popleft()
            self._video_frame_queue.append((msg, policy_info, int(msg.step_id)))
            self._video_frame_cv.notify()

    def _frame_worker_loop(self) -> None:
        while not self._shutdown_event.is_set():
            with self._video_frame_cv:
                while not self._video_frame_queue and not self._shutdown_event.is_set():
                    self._video_frame_cv.wait()
                if self._shutdown_event.is_set():
                    return
                msg, policy_info, step_id = self._video_frame_queue.popleft()
                self._video_frame_inflight = True
            try:
                with self._video_recorder_lock:
                    self._video_recorder.record_step(obs_msg_to_obs_dict(msg), policy_info, step_id)
            except Exception as exc:
                if not self._frame_error_warned:
                    print(f"[agent_video] frame record failed: {exc}")
                    self._frame_error_warned = True
            finally:
                with self._video_frame_cv:
                    self._video_frame_inflight = False
                    if not self._video_frame_queue:
                        self._video_frame_cv.notify_all()

    def _drain(self, timeout: float = 5.0) -> bool:
        """Wait for the frame queue to empty. Returns True if fully drained, False on timeout."""
        deadline = time.time() + float(timeout)
        with self._video_frame_cv:
            while self._video_frame_queue or self._video_frame_inflight:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                self._video_frame_cv.wait(timeout=min(0.05, remaining))
        return True

    # ── panel rendering (off the hot path) ───────────────────────────────────
    def _schedule_refresh(self) -> None:
        now = time.time()
        with self._video_policy_info_lock:
            if (
                self._video_policy_info_cache
                and now - self._video_policy_info_cache_t < self._VIDEO_MAP_PERIOD_S
            ):
                return
            if self._video_policy_info_rendering:
                return
            self._video_policy_info_rendering = True
            generation = self._video_policy_info_generation
        threading.Thread(
            target=self._refresh_worker,
            args=(generation,),
            daemon=True,
            name="agent_video_panel_render",
        ).start()

    def _refresh_worker(self, generation: int) -> None:
        policy_info = None
        try:
            policy_info = self._build_policy_info()
        except Exception as exc:
            if not self._panel_error_warned:
                print(f"[agent_video] panel render failed; keeping last video panel: {exc}")
                self._panel_error_warned = True
        stale_generation = False
        with self._video_policy_info_lock:
            stale_generation = generation != self._video_policy_info_generation
            complete = bool(
                policy_info
                and "map_panel" in policy_info
                and "detection_panel" in policy_info
            )
            if complete and not stale_generation:
                self._video_policy_info_cache = policy_info
                self._video_policy_info_cache_t = time.time()
            if stale_generation:
                self._video_det_panel_cache = None
                self._video_det_panel_key = None
            self._video_policy_info_rendering = False
        if stale_generation:
            self._video_map_viz = None
            self._video_det_viz = None

    def _build_policy_info(self) -> dict:
        agent = self._agent_getter()
        ss = getattr(agent, "shared_state", None) if agent is not None else None
        if ss is None:
            return {}
        policy_info = {}
        target = self._target_text(ss)
        if target:
            policy_info["target_object"] = target
        map_img = self._render_map(ss)
        if map_img is not None:
            policy_info["map_panel"] = map_img
            policy_info["map_title"] = "RtNav Frontier Map"
        det_img = self._render_detection(ss)
        if det_img is None:
            det_img = self._detection_placeholder(ss)
        if det_img is not None:
            policy_info["detection_panel"] = det_img
            policy_info["detection_title"] = "Object Detections"
        return policy_info

    def _detection_placeholder(self, ss) -> np.ndarray | None:
        import cv2

        try:
            with ss.lock:
                cameras = dict(getattr(ss.sensor, "latest_cameras", {}) or {})
        except Exception:
            cameras = {}
        camera_results = {}
        for name, camera in sorted(cameras.items()):
            rgb = self._rgb_u8(getattr(camera, "rgb", None))
            if rgb is None:
                continue
            camera_results[name] = SimpleNamespace(
                rgb_image_detector_input=rgb,
                detections_detector_input=[],
                rgb_image=rgb,
                detections=[],
            )
        if not camera_results:
            return None
        from rtnav.tools.visualization.detection_visualizer_thread import (
            WebDetectionVisualizerThread,
        )

        if self._video_det_viz is None or self._video_det_viz.shared_state is not ss:
            self._video_det_viz = WebDetectionVisualizerThread(
                ss, threading.Event(), display_height=0
            )
        image = self._bgr_to_rgb(
            self._video_det_viz._create_detection_grid(
                SimpleNamespace(camera_results=camera_results)
            )
        )
        if image is not None:
            cv2.putText(
                image,
                "waiting for detector",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return image

    def _target_text(self, ss) -> str:
        try:
            with ss.lock:
                return str(getattr(ss.task, "goal_category", "") or "").strip()
        except Exception:
            return ""

    def _render_map(self, ss):
        from rtnav.tools.visualization.map_visualizer_thread import MapVisualizerBase

        if self._video_map_viz is None or self._video_map_viz.shared_state is not ss:
            self._video_map_viz = MapVisualizerBase(ss)
        state = self._video_map_viz._get_state()
        return self._bgr_to_rgb(self._video_map_viz._render_frame(state))

    def _render_detection(self, ss):
        with ss.lock:
            detection_result = getattr(ss.perception, "detection_result", None)
            target_labels = set(getattr(ss.task, "synonym_to_canonical", {}) or {})
        if detection_result is None:
            return None
        cache_key = (
            getattr(detection_result, "timestamp", None),
            getattr(detection_result, "episode_index", None),
            getattr(detection_result, "total_detections", None),
            tuple(sorted(str(label) for label in target_labels)),
        )
        if self._video_det_panel_cache is not None and self._video_det_panel_key == cache_key:
            return self._video_det_panel_cache
        from rtnav.tools.visualization.detection_visualizer_thread import (
            WebDetectionVisualizerThread,
        )

        if self._video_det_viz is None or self._video_det_viz.shared_state is not ss:
            self._video_det_viz = WebDetectionVisualizerThread(
                ss, threading.Event(), display_height=0
            )
        img = self._bgr_to_rgb(
            self._video_det_viz._create_detection_grid(detection_result, target_labels)
        )
        self._video_det_panel_cache = img
        self._video_det_panel_key = cache_key
        return img

    @staticmethod
    def _bgr_to_rgb(img):
        if img is None:
            return None
        import cv2

        arr = AgentVideoRecorder._rgb_u8(img)
        if arr is None:
            return None
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _rgb_u8(img):
        if img is None:
            return None
        array = np.asarray(img)
        if array.ndim != 3 or array.shape[2] < 3:
            return None
        array = array[:, :, :3]
        if array.dtype != np.uint8:
            scale = (
                255.0
                if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1
                else 1.0
            )
            array = np.clip(array * scale, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)
