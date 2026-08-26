"""vLLM server lifecycle and client helpers.

`ensure_vllm_server` starts the local OpenAI-compatible server if it isn't
already running, then blocks until it's healthy. Call it once before
loading TaskParser or making any LLM calls.

`get_vllm_base_url` / `get_vllm_model_id` are client helpers for decision calls.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time as _time

_cached_model_id = None

_vllm_proc = None
_vllm_ensured = False


def get_vllm_base_url() -> str:
    """Return the vLLM base URL, reading VLLM_PORT from the environment (default 8000)."""
    port = int(os.environ.get("VLLM_PORT", "8000"))
    return f"http://localhost:{port}/v1"


def get_vllm_model_id(base_url: str | None = None) -> str:
    """Return the first model ID currently served by the vLLM server.

    Results are cached after the first successful call so subsequent calls are free.
    """
    global _cached_model_id
    if _cached_model_id is not None:
        return _cached_model_id
    if base_url is None:
        base_url = get_vllm_base_url()
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key="EMPTY")
        models = client.models.list()
        if models and models.data:
            _cached_model_id = models.data[0].id
            print(f"[vllm_utils] Using model: {_cached_model_id}")
            return _cached_model_id
    except Exception as e:
        msg = (
            "[vllm_utils] FATAL: could not fetch vLLM model list; "
            f"base_url={base_url!r} VLLM_PORT={os.environ.get('VLLM_PORT')!r} error={e!r}"
        )
        print(msg)
        raise RuntimeError(msg) from e
    msg = (
        "[vllm_utils] FATAL: vLLM returned no served models; "
        f"base_url={base_url!r} VLLM_PORT={os.environ.get('VLLM_PORT')!r}"
    )
    print(msg)
    raise RuntimeError(msg)


def kill_vllm_process() -> None:
    """Kill tracked vLLM subprocess and any stale api_server on ``VLLM_PORT``."""
    global _vllm_proc
    import signal as _sig

    port = int(os.environ.get("VLLM_PORT", "8000"))

    if _vllm_proc is not None:
        try:
            os.killpg(os.getpgid(_vllm_proc.pid), _sig.SIGKILL)
        except Exception:
            pass
        _vllm_proc = None

    try:
        result = subprocess.run(
            ["pgrep", "-f", f"vllm.entrypoints.openai.api_server.*--port {port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for pid in result.stdout.strip().split("\n"):
            try:
                os.kill(int(pid.strip()), 9)
            except (ValueError, ProcessLookupError):
                pass
    except Exception:
        pass


def vllm_process_running() -> bool:
    """True if vLLM api_server is listening on ``VLLM_PORT`` (may still be loading)."""
    port = int(os.environ.get("VLLM_PORT", "8000"))
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"vllm.entrypoints.openai.api_server.*--port {port}"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def ensure_vllm_server(
    model_dir: str,
    gpu_mem: float = 0.40,
    max_wait: float = 300.0,
) -> None:
    """Start vLLM if needed; wait until /health returns 200. Idempotent."""
    global _vllm_proc, _vllm_ensured
    if _vllm_ensured:
        return
    import requests

    port = int(os.environ.get("VLLM_PORT", "8000"))
    # Prefer a host-mounted dir (parallel workers) so the log survives the
    # container; fall back to /tmp for manual single-worker runs.
    _log_dir = os.environ.get("VLLM_CACHE_ROOT") or "/tmp"
    try:
        os.makedirs(_log_dir, exist_ok=True)
    except Exception:
        _log_dir = "/tmp"
    log_path = os.path.join(_log_dir, f"vllm_{port}.log")

    def _log_tail(n: int = 30) -> str:
        try:
            with open(log_path) as f:
                lines = f.readlines()
            return "".join(lines[-n:]).rstrip()
        except Exception:
            return "(no vllm log captured)"

    def _ready() -> bool:
        try:
            return requests.get(f"http://localhost:{port}/health", timeout=1).status_code == 200
        except Exception:
            return False

    if _ready():
        print(f"[agent] vllm server already running on port {port} — reusing")
        _vllm_ensured = True
        return

    if vllm_process_running():
        print(f"[agent] vllm server still loading — waiting for it to become ready on port {port}…")
    else:
        kill_vllm_process()
        _time.sleep(0.5)
        print(f"[agent] starting vllm server (gpu_mem={gpu_mem}, port={port}) for {model_dir} …")
        print(
            f"[agent] vllm stdout/stderr -> {log_path} "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})"
        )
        flashinfer_workspace = shlex.quote(os.environ.get("FLASHINFER_WORKSPACE_BASE", "/tmp"))
        # Inherit the process's CUDA_VISIBLE_DEVICES (the worker's GPU) — do NOT
        # hardcode 0, or every parallel worker's vLLM lands on GPU 0 and OOMs.
        # Capture output to a per-port log so startup failures are diagnosable
        # (otherwise the traceback is interleaved across workers' stdout).
        _vllm_log = open(log_path, "w")
        _vllm_proc = subprocess.Popen(
            f"PYTHONNOUSERSITE=1 FLASHINFER_WORKSPACE_BASE={flashinfer_workspace} "
            f"python3 -m vllm.entrypoints.openai.api_server "
            f"--model {model_dir} "
            f"--dtype bfloat16 --tensor-parallel-size 1 "
            f"--gpu-memory-utilization {gpu_mem} "
            f"--max-model-len 4096 "
            f"--max-num-seqs 32 "
            f"--port {port} "
            f"--host 127.0.0.1",
            shell=True,
            start_new_session=True,
            stdout=_vllm_log,
            stderr=subprocess.STDOUT,
        )

    start = _time.time()
    try:
        while _time.time() - start < max_wait:
            if _vllm_proc is not None and _vllm_proc.poll() is not None:
                raise RuntimeError(
                    f"[agent] vllm server process exited with code {_vllm_proc.returncode}\n"
                    f"--- last lines of {log_path} ---\n{_log_tail()}"
                )
            if _vllm_proc is None and not vllm_process_running():
                raise RuntimeError("[agent] vllm server process disappeared while waiting")
            if _ready():
                print(f"[agent] vllm server ready in {_time.time() - start:.1f}s")
                _vllm_ensured = True
                return
            _time.sleep(3.0)
    except KeyboardInterrupt:
        print("\n[agent] interrupted — leaving vllm server running")
        raise

    raise RuntimeError(
        f"[agent] vllm server did not become ready in time\n"
        f"--- last lines of {log_path} ---\n{_log_tail()}"
    )
