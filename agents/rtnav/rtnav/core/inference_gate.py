"""
Inference gate — GPU exclusivity between detector and vLLM callers.

================================================================================
DESIGN
================================================================================
The OWLv2 detector and the vLLM server share the same GPU.
Running them simultaneously causes severe contention (detector inference
balloons from ~300 ms to >1 s).  Solution: they never overlap.

The detector runs as fast as possible.  When any module needs the vLLM server
it *pauses* the detector, waits for the current inference cycle to finish,
makes the API call, then *resumes* the detector.

Key primitives
--------------
_vllm_pause   : threading.Event, SET = detector may run, CLEAR = detector must pause.
_detector_idle: threading.Event, SET = detector is between cycles (safe to use vLLM).

Detector thread
~~~~~~~~~~~~~~~
- At the top of each cycle: if _vllm_pause is clear -> signal _detector_idle, wait.
- Before inference: _detector_idle.clear()  (mark busy)
- After  inference: _detector_idle.set()    (mark idle)

vLLM callers
~~~~~~~~~~~~
gate.pause_for_vllm()          # clear _vllm_pause, wait for _detector_idle
try:
    # ... make vLLM API call ...
finally:
    gate.resume_after_vllm()   # set _vllm_pause so detector resumes

"""

import threading


class InferenceGate:
    """Keep OWLv2 and vLLM from running on the GPU simultaneously."""

    def __init__(self):
        # SET = detector may run;  CLEAR = detector must pause.
        self._vllm_pause = threading.Event()
        self._vllm_pause.set()  # detector starts freely

        # SET = detector is idle (between cycles);  CLEAR = detector is mid-inference.
        self._detector_idle = threading.Event()
        self._detector_idle.set()  # idle at startup

    def detector_should_pause(self) -> bool:
        """Return True if a vLLM caller wants the detector to pause."""
        return not self._vllm_pause.is_set()

    def detector_wait_for_resume(self, timeout: float = 30.0) -> bool:
        """Block until vLLM caller says detector may resume.  Returns False on timeout."""
        self._detector_idle.set()  # let the vLLM caller know we stopped
        return self._vllm_pause.wait(timeout=timeout)

    def detector_mark_busy(self) -> None:
        """Detector is about to start a GPU inference cycle."""
        self._detector_idle.clear()

    def detector_mark_idle(self) -> None:
        """Detector finished its GPU inference cycle."""
        self._detector_idle.set()

    def pause_for_vllm(self, timeout: float = 10.0) -> bool:
        """Pause the detector and wait for it to be idle before a vLLM call.

        Returns True if the detector is confirmed idle (safe to call vLLM),
        False on timeout (caller should still proceed but expect contention).
        """
        self._vllm_pause.clear()
        return self._detector_idle.wait(timeout=timeout)

    def resume_after_vllm(self) -> None:
        """Let the detector run again after a vLLM call completes.

        Includes a CUDA cool-down so the detector's first cycle after resume
        doesn't hit residual GPU context-switch overhead from the vLLM server
        (separate process, shared GPU).

        The vLLM server runs in a *separate* process — torch.cuda.synchronize()
        from our process does NOT fence vLLM's CUDA kernels.  We need a longer
        sleep to let the vLLM server's GPU work (KV cache cleanup, memory pool
        compaction) fully drain before the detector fires up.
        """
        import time as _time

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()  # flush OUR process's pending CUDA work
        except Exception:
            pass
        # 150ms cooldown: vLLM's separate CUDA context needs time to fully
        # release GPU resources (kernel queues, memory allocator).
        # 20ms was far too short and left residual contention.
        _time.sleep(0.15)

        self._vllm_pause.set()
