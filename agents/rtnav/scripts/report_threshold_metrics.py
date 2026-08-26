#!/usr/bin/env python3
"""Report ObjectNav metrics at configurable success-distance thresholds.

An episode is successful when the agent called STOP and its final distance to
the goal is within the threshold.  The log does not store ``stop_called``
directly, so it is inferred from OVON's terminal failure mode: ``success``,
``stop_too_far``, and ``misidentification`` all indicate a STOP action.

For episodes already marked successful, their logged SPL/SCT values are kept.
For newly successful episodes, SPL path efficiency is recovered from SoftSPL
and SCT is reconstructed from its timing fields.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_THRESHOLDS = (0.1, 0.2, 1.0)
STOPPED_FAILURE_MODES = frozenset({"success", "stop_too_far", "misidentification"})


def _finite_number(episode: dict[str, Any], field: str, episode_id: str) -> float:
    value = episode.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Episode {episode_id!r} has no numeric {field!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Episode {episode_id!r} has non-finite {field!r}")
    return value


def _called_stop(episode: dict[str, Any]) -> bool:
    """Infer whether the episode ended with STOP from the logged fields."""
    if _success_value(episode) > 0.5:
        return True
    return episode.get("ovon_failure_mode") in STOPPED_FAILURE_MODES


def _success_value(episode: dict[str, Any]) -> float:
    value = episode.get("success", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _spl_for_success(episode: dict[str, Any], episode_id: str) -> float:
    """Return SPL assuming this episode is successful at the new threshold."""
    if _success_value(episode) > 0.5:
        return _finite_number(episode, "spl", episode_id)

    # Habitat's SoftSPL is soft_success * path_efficiency, where
    # soft_success = max(0, 1 - final_distance / initial_distance).  Recovering
    # the efficiency this way uses Habitat's own path-length accounting.
    distance = _finite_number(episode, "distance_to_goal", episode_id)
    initial_distance = _finite_number(episode, "geodesic_distance", episode_id)
    soft_spl = _finite_number(episode, "soft_spl", episode_id)
    soft_success = max(0.0, 1.0 - distance / initial_distance) if initial_distance > 0 else 0.0
    if soft_success > 0:
        return min(1.0, max(0.0, soft_spl / soft_success))

    # This fallback is only needed when no positive progress was made.
    path_length = _finite_number(episode, "path_length", episode_id)
    return initial_distance / max(path_length, initial_distance) if initial_distance > 0 else 0.0


def _sct_for_success(episode: dict[str, Any], episode_id: str) -> float:
    """Return SCT assuming this episode is successful at the new threshold."""
    if _success_value(episode) > 0.5:
        return _finite_number(episode, "sct", episode_id)
    optimal_time = _finite_number(episode, "optimal_time", episode_id)
    completion_time = _finite_number(episode, "completion_time", episode_id)
    return optimal_time / max(completion_time, optimal_time) if optimal_time > 0 else 0.0


def calculate_metrics(
    episodes: dict[str, dict[str, Any]], thresholds: Iterable[float]
) -> list[dict[str, float | int]]:
    """Calculate SR, SPL, and SCT for every requested threshold."""
    if not episodes:
        raise ValueError("The input contains no per-episode records")

    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(f"Invalid success threshold: {threshold}")

        successes = 0
        spl_sum = 0.0
        sct_sum = 0.0
        for episode_id, episode in episodes.items():
            distance = _finite_number(episode, "distance_to_goal", episode_id)
            success = _called_stop(episode) and distance <= threshold
            if not success:
                continue

            successes += 1
            spl_sum += _spl_for_success(episode, episode_id)
            sct_sum += _sct_for_success(episode, episode_id)

        count = len(episodes)
        rows.append(
            {
                "threshold": threshold,
                "episodes": count,
                "successes": successes,
                "success_rate": 100.0 * successes / count,
                "spl": spl_sum / count,
                "sct": sct_sum / count,
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report success rate, SPL, and SCT at multiple distance thresholds"
    )
    parser.add_argument("stats", type=Path, help="Path to a live.json/stats JSON file")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        metavar="METERS",
        help="Success distances in meters (default: 0.1 0.2 1.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with args.stats.open(encoding="utf-8") as file:
        payload = json.load(file)

    episodes = payload.get("per_episode")
    if not isinstance(episodes, dict):
        raise ValueError("Expected a top-level 'per_episode' object")

    rows = calculate_metrics(episodes, args.thresholds)
    print(f"Episodes: {len(episodes)}")
    print(f"{'Threshold':>10}  {'Successes':>9}  {'SR (%)':>8}  {'SPL':>8}  {'SCT':>8}")
    print(f"{'-' * 10}  {'-' * 9}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for row in rows:
        print(
            f"{row['threshold']:>9.3f}m  {row['successes']:>9d}  "
            f"{row['success_rate']:>8.2f}  {row['spl']:>8.4f}  {row['sct']:>8.4f}"
        )


if __name__ == "__main__":
    main()
