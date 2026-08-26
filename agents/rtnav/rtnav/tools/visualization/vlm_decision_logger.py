"""Asynchronous per-episode visual logs for VLM decisions."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import cv2
import numpy as np


class VLMDecisionLogger:
    """Save frontier choices and target-verification evidence off the hot path."""

    def __init__(self):
        self._episode_dir: Path | None = None
        self._episode = {}
        self._queue: queue.Queue = queue.Queue(maxsize=256)
        self._ids = {"frontier": 0, "verification": 0}
        self._id_lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True, name="vlm_decision_logger")
        self._worker.start()

    def new_episode(self, scene_id, episode_id, output_dir=None):
        with self._id_lock:
            self._episode_dir = None
        self.flush()
        episode_dir = Path(output_dir) / "vlm_decisions" if output_dir else None
        if episode_dir is not None:
            (episode_dir / "frontier").mkdir(parents=True, exist_ok=True)
            (episode_dir / "verification").mkdir(parents=True, exist_ok=True)
            (episode_dir / "frontier" / "logprobs.txt").write_text("", encoding="utf-8")
        with self._id_lock:
            self._episode_dir = episode_dir
            self._episode = {"scene_id": scene_id, "episode_id": episode_id}
            self._ids = {"frontier": 0, "verification": 0}

    def record_frontier(self, input_rgb, selected_rgb, event, rejected_rgb=None):
        self._enqueue("frontier", input_rgb, selected_rgb, event, rejected_rgb)

    def record_verification(self, rgb, event):
        self._enqueue("verification", rgb, None, event)

    def save_episode(self, metrics=None):
        self.flush()

    def flush(self):
        self._queue.join()

    def shutdown(self):
        self.flush()
        self._queue.put(None)
        self._worker.join(timeout=2.0)

    def _enqueue(self, kind, image, second_image, event, rejected_image=None):
        with self._id_lock:
            episode_dir = self._episode_dir
            if episode_dir is None:
                return
            episode = dict(self._episode)
            self._ids[kind] += 1
            event_id = self._ids[kind]
        item = (
            episode_dir,
            kind,
            event_id,
            self._copy_image(image),
            self._copy_image(second_image),
            self._copy_image(rejected_image),
            self._json_value({**episode, **event}),
        )
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            print(f"[VLMLog] queue full; dropped {kind} event")

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._write(*item)
            except Exception as exc:
                print(f"[VLMLog] write failed: {exc}")
            finally:
                self._queue.task_done()

    @staticmethod
    def _copy_image(image):
        if image is None:
            return None
        array = np.asarray(image)
        if array.size == 0:
            return None
        return np.ascontiguousarray(array.copy())

    @classmethod
    def _json_value(cls, value):
        if isinstance(value, dict):
            return {str(key): cls._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_value(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _write(self, episode_dir, kind, event_id, image, second_image, rejected_image, event):
        event = {"id": event_id, "type": kind, **event}
        folder = episode_dir / kind
        stem = f"{event_id:04d}"
        if kind == "frontier":
            self._write_rgb(folder / f"{stem}_input.jpg", image)
            self._write_rgb(folder / f"{stem}_selected.jpg", second_image)
            self._write_rgb(folder / f"{stem}_rejected.jpg", rejected_image)
            self._write_frontier_log(folder / "logprobs.txt", event)
        else:
            verdict = "yes" if event.get("accepted") else "no"
            self._write_rgb(folder / f"{stem}_{verdict}.png", image)
        with (episode_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    @staticmethod
    def _write_frontier_log(path, event):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"Decision {int(event['id']):04d} | query_id={event.get('query_id', 'n/a')} "
                f"| scope={event.get('candidate_scope', 'n/a')} "
                f"| response={event.get('response', '')!r}\n"
            )
            for rank, item in enumerate(event.get("top_tokens", [])[:10], start=1):
                stream.write(
                    f"  {rank:2d}. token={item.get('token', '')!r} "
                    f"probability={float(item.get('probability', 0.0)):.6f}\n"
                )
            excluded = event.get("excluded_frontier_indices") or []
            missing = event.get("missing_frontier_indices") or []
            if excluded:
                stream.write(f"  excluded_not_queried={excluded}\n")
            if missing:
                stream.write(f"  missing_image_not_queried={missing}\n")
            stream.write("\n")

    @staticmethod
    def _write_rgb(path, image):
        if image is None:
            return
        array = np.asarray(image)
        if array.dtype != np.uint8:
            scale = (
                255.0
                if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1
                else 1.0
            )
            array = np.clip(array * scale, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        else:
            array = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), array):
            raise OSError(f"could not write {path}")
