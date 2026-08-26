"""Resolve the agent's output directory from the RUN_DIR environment variable."""

import datetime
import os
from pathlib import Path


def resolve_agent_run_dir() -> Path:
    """Return the run directory for this worker.

    In Docker (parallel eval): RUN_DIR is set to /opt/worker_logs by the compose
    launcher and returned directly.

    For local single-worker runs: falls back to outputs/{mode}_{benchmark}_{baseline}/{stamp}.
    """
    if run_dir := os.environ.get("RUN_DIR"):
        return Path(run_dir)

    mode = os.environ["MODE"]
    benchmark = os.environ["BENCHMARK"]
    baseline = os.environ["BASELINE"]
    stamp = os.environ.get("RUN_STAMP") or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    here = Path(__file__).resolve()
    root = next(
        (p for p in here.parents if (p / "envs" / "env_node.py").is_file()),
        here.parents[4],
    )
    run_dir = root / "outputs" / f"{mode}_{benchmark}_{baseline}" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RUN_DIR"] = str(run_dir)
    return run_dir
