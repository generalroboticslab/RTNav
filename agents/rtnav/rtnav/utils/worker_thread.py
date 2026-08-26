"""Shared lifecycle for the agent's long-lived worker threads.

A worker is started once and runs for the whole process (many episodes). It
loads any heavy, thread-affine state in ``_setup`` (e.g. a GPU model), then loops
until ``shared_state.task_ready`` and runs one cycle. The runner drives episode
boundaries by calling ``reset_episode()`` synchronously on each worker.
"""

import threading
import time
import traceback

from rtnav.utils.task_gate import wait_for_task_ready


class WorkerThread(threading.Thread):
    def __init__(self, shared_state, shutdown_event, name):
        super().__init__(name=name, daemon=True)
        self.shared_state = shared_state
        self.shutdown_event = shutdown_event
        with shared_state.lock:
            self._last_episode_index = shared_state.system.episode_index

    def run(self):
        print(f"[{self.name}] started")
        if self._setup() is False:  # e.g. model load failed
            return
        while not self.shutdown_event.is_set():
            if not self._await_ready():
                break
            try:
                self._step()
            except Exception as e:
                print(f"[{self.name}] error: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    # ── subclass hooks ─────────────────────────────────────────────────────────
    def _setup(self) -> bool:
        """One-time in-thread init before the loop. Return False to abort."""
        return True

    def _step(self) -> None:
        """Run one work cycle. Called repeatedly while the task is active."""
        raise NotImplementedError

    def _reset(self) -> None:
        """Clear per-episode state; default does nothing."""

    # ── public API ─────────────────────────────────────────────────────────────
    def reset_episode(self) -> None:
        """Synchronous episode-boundary reset, called by the runner."""
        self._reset()

    # ── helpers ────────────────────────────────────────────────────────────────
    def _await_ready(self) -> bool:
        """Block until the episode goal is applied; False if shutdown intervenes."""
        return wait_for_task_ready(self.shared_state, self.name, self.shutdown_event)

    def _episode_advanced(self) -> bool:
        """True once per episode boundary — when the runner has bumped
        ``episode_index`` since we last saw it. Only the first caller per
        boundary sees the advance."""
        with self.shared_state.lock:
            counter = self.shared_state.system.episode_index
            if counter <= self._last_episode_index:
                return False
            self._last_episode_index = counter
        return True
