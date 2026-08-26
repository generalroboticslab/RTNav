"""latest_cameras → detection_result: run OWLv2, project
boxes to world XY, and publish detections to shared_state."""

import time

import numpy as np

from rtnav.config.detection_classes import OBJECT_CLASSES
from rtnav.config.model_paths import OWLV2_MODEL_DIR
from rtnav.core.data_types import (
    CameraDetectionResult,
    DetectionEntity,
    MultiCameraDetectionResult,
)
from rtnav.utils.worker_thread import WorkerThread


class OpenVocabDetectorThread(WorkerThread):
    """Detects the target + object classes in the latest camera frames, projects
    them to world XY, and publishes detection_result to shared_state."""

    MIN_TARGET_RANGE_M = 0.5  # reject near-field phantoms (camera near-clip / robot body)

    def __init__(self, shared_state, shutdown_event, cfg):
        super().__init__(shared_state, shutdown_event, name="DetectorThread")
        self.threshold = cfg.detection.threshold
        self.inference_res = cfg.detection.inference_res
        self.classes = list(OBJECT_CLASSES)
        self._detector = None
        self._to_tensor = None
        self._last_skip = ""
        self._resumed_from_vllm = False
        print(f"[Detector] {len(self.classes)} classes")

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _setup(self) -> bool:
        # Load the GPU model once in-thread (CUDA/fork safety) before the loop.
        return self._load_model()

    def _reset(self) -> None:
        """Reset per-episode skip/log state when the runner advances the episode."""
        if self._episode_advanced():
            self._last_skip = ""
            self._resumed_from_vllm = False

    def _load_model(self) -> bool:
        import torch
        from torchvision.transforms import ToTensor

        from rtnav.modules.perception.object_detector import OpenVocabularyDetector

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Detector] loading OWLv2 from {OWLV2_MODEL_DIR}")
            self._detector = OpenVocabularyDetector(
                owl_dir=OWLV2_MODEL_DIR,
                device=device,
                inference_res=self.inference_res,
            )
            if hasattr(self._detector, "warmup"):
                self._detector.warmup()  # prime CUDA so the first inference isn't slow
            self._to_tensor = ToTensor()
            print(f"[Detector] ready on {self._detector.device}")
            return True
        except Exception as e:
            print(f"[Detector] load failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    # ── one detection cycle ──────────────────────────────────────────────────
    def _step(self):
        skip, reason = self._should_skip()
        if skip:
            if reason != self._last_skip:
                print(f"[Detector] skipped: {reason}")
                self._last_skip = reason
            time.sleep(0.1)
            return
        if self._last_skip:
            print(f"[Detector] resumed (was: {self._last_skip})")
            self._last_skip = ""

        with self.shared_state.lock:
            cameras = dict(getattr(self.shared_state.sensor, "latest_cameras", {}) or {})
            odom = self.shared_state.sensor.latest_odom
            synonym_to_canonical = dict(
                getattr(self.shared_state.task, "synonym_to_canonical", {}) or {}
            )
            reset_counter = getattr(self.shared_state.system, "episode_index", 0)
        if not cameras:
            time.sleep(0.05)
            return

        cam_names, cam_obs_list, tensors = self._collect_batch(cameras)
        if not tensors:
            time.sleep(0.01)
            return

        if self._should_skip()[0]:  # re-check right before GPU work
            time.sleep(0.01)
            return
        if not self._acquire_gpu():  # paused for a vLLM caller
            return
        outputs = self._run_batch(tensors, synonym_to_canonical)
        if outputs is None or self._batch_is_stale(reset_counter):
            return

        ts = time.time()  # after inference, so it exceeds any flush_time set during it
        T_world_base = cam_obs_list[0].T_world_cam if cam_obs_list else np.eye(4)
        pose = self._detection_pose(odom, T_world_base)

        camera_results = {}
        for cam_name, cam_obs, output in zip(cam_names, cam_obs_list, outputs):
            camera_results[cam_name] = self._build_camera_result(
                cam_name, cam_obs, output, pose, ts
            )

        total = sum(len(r.detections) for r in camera_results.values())
        self._publish(
            MultiCameraDetectionResult(
                timestamp=ts,
                robot_odom=odom or (0, 0, 0),
                T_world_base=T_world_base,
                camera_results=camera_results,
                total_detections=total,
                has_detections=total > 0,
                episode_index=reset_counter,
            )
        )
        time.sleep(0.001)

    # ── cycle helpers ────────────────────────────────────────────────────────
    def _should_skip(self):
        """Return whether detection should wait for the episode target."""
        if not self.shared_state.task_ready.is_set():
            return True, "task_not_ready"
        return False, ""

    def _collect_batch(self, cameras):
        """Each camera with a valid RGB frame → a (rotated) input tensor."""
        cam_names, cam_obs_list, tensors = [], [], []
        for cam_name, cam_obs in cameras.items():
            rgb = cam_obs.rgb
            if rgb is None or not hasattr(rgb, "ndim") or rgb.ndim < 2:
                continue
            det_rgb = self._prepare_detector_rgb(rgb, rot_k=getattr(cam_obs, "rotation_k", 0))
            cam_names.append(cam_name)
            cam_obs_list.append(cam_obs)
            tensors.append(self._to_tensor(det_rgb))
        return cam_names, cam_obs_list, tensors

    def _acquire_gpu(self) -> bool:
        """Mark the shared GPU busy; if a vLLM caller needs it, release + wait and
        return False. mark_busy() precedes should_pause() to close the race window."""
        gate = self.shared_state.inference
        gate.detector_mark_busy()
        if gate.detector_should_pause():
            gate.detector_mark_idle()
            if self._last_skip != "vllm_pause":
                print("[Detector] paused for vLLM")
                self._last_skip = "vllm_pause"
            gate.detector_wait_for_resume(timeout=30.0)
            self._resumed_from_vllm = True
            return False
        return True

    def _run_batch(self, tensors, synonym_to_canonical):
        """Batched detector forward pass; always releases the GPU gate."""
        import torch

        classes = {"targets": synonym_to_canonical, "additional": self.classes}
        t0 = time.time()
        try:
            images = torch.stack(tensors, dim=0)
            outputs = self._detector.detect_batch(
                image=images, detection_classes=classes, threshold=self.threshold
            )
            del images
            ms = int((time.time() - t0) * 1000)
            if ms > 500:
                print(
                    f"[Detector] slow detect_batch: {ms}ms"
                    + (" (post-vLLM)" if self._resumed_from_vllm else "")
                )
            self._resumed_from_vllm = False
            return outputs
        finally:
            self.shared_state.inference.detector_mark_idle()

    def _batch_is_stale(self, reset_counter) -> bool:
        """True if an episode reset or task-gate drop landed during inference."""
        with self.shared_state.lock:
            current = getattr(self.shared_state.system, "episode_index", 0)
        task_ready = getattr(self.shared_state, "task_ready", None)
        stale = current != reset_counter or (task_ready is not None and not task_ready.is_set())
        if stale:
            print(f"[Detector] dropping stale batch (frame={reset_counter}, current={current})")
        return stale

    @staticmethod
    def _detection_pose(odom, T_world_base):
        """Robot (x, y, yaw): prefer odom, else extract from T_world_base."""
        if odom and odom != (0, 0, 0):
            return odom
        if T_world_base is not None and not np.allclose(T_world_base, np.eye(4)):
            return (
                float(T_world_base[0, 3]),
                float(T_world_base[1, 3]),
                float(np.arctan2(T_world_base[1, 0], T_world_base[0, 0])),
            )
        return None

    def _build_camera_result(self, cam_name, cam_obs, output, pose, ts):
        """One camera's detections → CameraDetectionResult."""
        orig_rgb = cam_obs.rgb
        h, w = orig_rgb.shape[:2]
        T_world_cam = cam_obs.T_world_cam
        dets = output.detections
        rot_k = getattr(cam_obs, "rotation_k", 0)
        det_rgb = self._prepare_detector_rgb(orig_rgb, rot_k=rot_k)

        if dets:
            from rtnav.modules.perception.cpp_accel import batch_postprocess_detections

            raw = np.array([d.bbox for d in dets], dtype=np.float64)
            # rot_k rotates bboxes back to the original orientation; per-camera
            # depth range lets the projection reject Habitat clip-to-min/max sentinels.
            adj_bboxes, valid, _world_xy, _has_world = batch_postprocess_detections(
                raw,
                rot_k,
                w,
                h,
                cam_obs.depth,
                cam_obs.intrinsics,
                T_world_cam,
                float(cam_obs.min_depth_m),
                float(cam_obs.max_depth_m),
            )
        else:
            adj_bboxes = np.zeros((0, 4))
            valid = np.zeros(0, bool)

        adjusted = []
        for idx, det in enumerate(dets):
            if not valid[idx]:
                continue
            x1, y1, x2, y2 = adj_bboxes[idx]
            adjusted.append(
                DetectionEntity(
                    name=det.name,
                    confidence=det.confidence,
                    bbox=(x1, y1, x2, y2),
                    location=((x1 + x2) / 2, (y1 + y2) / 2),
                    embeddings=det.embeddings,
                    top_k_label_probs=getattr(det, "top_k_label_probs", None),
                )
            )

        return CameraDetectionResult(
            camera_name=cam_name,
            timestamp=ts,
            detections=adjusted,
            rgb_image=orig_rgb,
            depth_image=cam_obs.depth,
            T_world_camera=T_world_cam,
            intrinsics_depth=cam_obs.intrinsics,
            intrinsics_rgb=getattr(cam_obs, "intrinsics_rgb", None),
            min_depth_m=float(cam_obs.min_depth_m),
            max_depth_m=float(cam_obs.max_depth_m),
            rotation_k=rot_k,
            rgb_image_detector_input=det_rgb,
            detections_detector_input=[
                DetectionEntity(
                    name=d.name,
                    confidence=d.confidence,
                    bbox=d.bbox,
                    location=((d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2),
                    embeddings=d.embeddings,
                    top_k_label_probs=getattr(d, "top_k_label_probs", None),
                )
                for d in dets
            ],
        )

    # ── publish ──────────────────────────────────────────────────────────────
    def _publish(self, detection_result):
        with self.shared_state.lock:
            current_reset = getattr(self.shared_state.system, "episode_index", 0)
        if detection_result.episode_index != current_reset:
            print(
                f"[Detector] not publishing stale result "
                f"(result={detection_result.episode_index}, current={current_reset})"
            )
            return

        with self.shared_state.lock:
            self.shared_state.perception.detection_result = detection_result

    @staticmethod
    def _as_rgb3(rgb):
        """Coerce any HxW / HxWxC image to HxWx3 uint8."""
        arr = np.asarray(rgb)
        if arr.ndim == 2:
            return np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3:
            raise RuntimeError(f"Detector RGB must be HxW or HxWxC, got shape={arr.shape}")
        channels = arr.shape[2]
        if channels == 1:
            return np.repeat(arr, 3, axis=2)
        if channels == 3:
            return arr
        if channels == 4:
            return arr[:, :, :3]
        raise RuntimeError(f"Detector RGB must have 1, 3, or 4 channels, got shape={arr.shape}")

    @staticmethod
    def _prepare_detector_rgb(rgb, rot_k=0):
        """Image fed to the detector: rotate by rot_k*90° (axes 0,1). Boxes are
        rotated back to the original frame downstream via batch_postprocess."""
        rgb = OpenVocabDetectorThread._as_rgb3(rgb)
        return np.rot90(rgb, k=rot_k, axes=(0, 1)).copy() if rot_k else rgb.copy()
