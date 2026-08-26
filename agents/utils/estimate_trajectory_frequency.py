#!/usr/bin/env python3
"""Estimate simulator tick frequency from a merged trajectories JSON file."""

import argparse
import json
from pathlib import Path


def estimate(data: dict) -> tuple[float, str, float, float, int]:
    total_steps = 0
    total_seconds = 0.0
    slowest = None

    for key, record in data.items():
        trajectory = record.get("traj", [])
        episode_steps = 0
        episode_seconds = 0.0
        for previous, current in zip(trajectory, trajectory[1:]):
            try:
                seconds = float(current[3]) - float(previous[3])
                steps = int(current[4]) - int(previous[4])
            except (IndexError, TypeError, ValueError):
                continue
            if seconds <= 0 or steps <= 0:
                continue
            episode_steps += steps
            episode_seconds += seconds

        if episode_seconds == 0:
            continue

        frequency = episode_steps / episode_seconds
        total_steps += episode_steps
        total_seconds += episode_seconds
        if slowest is None or frequency < slowest[1]:
            slowest = (key, frequency, episode_seconds, episode_steps)

    if slowest is None:
        raise ValueError("no episodes contain two valid trajectory samples")

    return total_steps / total_seconds, *slowest


def self_test() -> None:
    point = lambda timestamp, step: [0, 0, 0, timestamp, step]
    result = estimate(
        {
            "fast_0": {"traj": [point(0, 0), point(1, 30)]},
            "slow_0": {"traj": [point(0, 0), point(1, 15)]},
        }
    )
    assert result == (22.5, "slow_0", 15.0, 1.0, 15), result
    print("Self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_json", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.trajectory_json is None:
        parser.error("trajectory_json is required unless --self-test is used")

    try:
        with args.trajectory_json.open() as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("top-level JSON value must be an object")
        overall_hz, key, minimum_hz, seconds, steps = estimate(data)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))

    print(f"Estimated frequency: {overall_hz:.3f} Hz")
    print(f"Minimum-frequency episode: {key}")
    print(f"  Frequency: {minimum_hz:.3f} Hz")
    print(f"  Covered time: {seconds:.2f} s")
    print(f"  Step delta: {steps}")


if __name__ == "__main__":
    main()
