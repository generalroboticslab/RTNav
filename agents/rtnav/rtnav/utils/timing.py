"""Tiny opt-in timing helper."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Optional


def _threshold_ms() -> int:
    try:
        return int(os.environ.get("TIMING_MS", "0"))
    except ValueError:
        return 0


@contextmanager
def timing(name: str, warn_ms: Optional[int] = None):
    threshold = warn_ms if warn_ms is not None else _threshold_ms()
    if threshold <= 0:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms >= threshold:
            print(f"[SLOW] {name} {elapsed_ms:.0f}ms", flush=True)
