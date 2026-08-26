#!/usr/bin/env python3
"""Merge per-subset stats JSON files from parallel evaluation.

Usage:
    python merge_stats.py --input-dir logs_parallel --num-workers 8
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _stats_path(input_dir: Path, worker_idx: int) -> Path:
    candidates = [
        input_dir / f"subset_{worker_idx}" / f"stats_{worker_idx}.json",
        input_dir / f"worker_{worker_idx}" / "env_stats.json",
    ]
    for path in candidates:
        if path.exists():
            return path

    worker_dir = input_dir / f"worker_{worker_idx}"
    for root in sorted(worker_dir.glob("*_async")):
        latest = root / "LATEST"
        if latest.exists():
            target = latest.read_text().strip()
            if target:
                path = Path(target) / "env_stats.json"
                if path.exists():
                    return path
                path = root / Path(target).name / "env_stats.json"
                if path.exists():
                    return path
    return candidates[0]


def _numeric_mean(all_eps: dict) -> dict:
    keys = sorted({k for ep in all_eps.values() for k in ep})
    agg = {}
    for key in keys:
        values = [
            ep[key]
            for ep in all_eps.values()
            if isinstance(ep.get(key), (int, float)) and not isinstance(ep.get(key), bool)
        ]
        if values:
            agg[f"mean_{key}"] = float(sum(values) / len(values))
    return agg


def _count_field(all_eps: dict, key: str) -> dict:
    counts = {}
    for ep in all_eps.values():
        value = ep.get(key)
        if value:
            counts[value] = counts.get(value, 0) + 1
    return counts


def main():
    p = argparse.ArgumentParser(description="Merge parallel eval stats")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--num-workers", type=int, default=8)
    # Collect episodes whose success flipped vs a baseline run. Off by default.
    p.add_argument("--compare", action="store_true")
    # Baseline run dir for --compare. Omit to auto-pick the previous run on the
    # same split (or set BASELINE_RUN).
    p.add_argument("--baseline", default=None)
    args = p.parse_args()

    args.output = Path(args.input_dir) / "live.json"

    all_eps = {}
    episode_counts = []
    for i in range(args.num_workers):
        path = _stats_path(Path(args.input_dir), i)
        if not path.exists():
            print(f"[WARN] Missing: {path}")
            episode_counts.append(0)
            continue
        data = json.loads(path.read_text())
        per_ep = data.get("per_episode", {})
        episode_counts.append(len(per_ep))
        overlap = set(per_ep) & set(all_eps)
        if overlap:
            print(f"[WARN] Worker {i}: {len(overlap)} overlapping episodes")
        all_eps.update(per_ep)

    print(f"[INFO] Episode counts by worker: {episode_counts}")

    if not all_eps:
        print("No episodes found.")
        sys.exit(1)

    agg = _numeric_mean(all_eps)
    successes = sum(1 for v in all_eps.values() if v.get("success", 0) > 0.5)
    n = len(all_eps)
    agg.update(num_episodes=n, num_successes=successes, success_rate=successes / n * 100)
    ovon_failure_modes = _count_field(all_eps, "ovon_failure_mode")
    if ovon_failure_modes:
        agg["ovon_failure_modes"] = ovon_failure_modes

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    Path(args.output).write_text(json.dumps({"aggregated": agg, "per_episode": all_eps}, indent=2))

    # Merge trajectories: subset_<i>/trajectories/*.json → single trajectories.json
    # keyed by "<scene>_<ep>" (the per-file stem).
    all_trajs = {}
    for i in range(args.num_workers):
        traj_src = Path(args.input_dir) / f"subset_{i}" / "trajectories"
        if not traj_src.is_dir():
            continue
        for f in traj_src.glob("*.json"):
            all_trajs[f.stem] = json.loads(f.read_text())
    if all_trajs:
        traj_path = Path(args.input_dir) / "trajectories.json"
        traj_path.write_text(json.dumps(all_trajs, indent=2))
        print(f"[INFO] Merged {len(all_trajs)} trajectories → {traj_path}")

    print(
        f"\nMerged {n} episodes: SR={agg['success_rate']:.1f}% "
        f"SPL={agg.get('mean_spl', 0):.4f} SCT={agg.get('mean_sct', 0):.4f} "
        f"DTG={agg.get('mean_distance_to_goal', 0):.4f}"
    )

    threshold_report = (
        Path(__file__).resolve().parents[1]
        / "rtnav"
        / "scripts"
        / "report_threshold_metrics.py"
    )
    if threshold_report.exists():
        sys.stdout.flush()
        subprocess.run(
            [
                sys.executable,
                str(threshold_report),
                str(args.output),
                "--thresholds",
                "0.1",
                "0.2",
            ],
            check=True,
        )

    # Collect episodes whose success flipped vs a baseline into <input-dir>/flipped/.
    # Only when --compare is passed; collect_flipped auto-resolves the baseline
    # unless --baseline is given.
    if args.compare:
        cf = Path(__file__).resolve().parents[2] / "experiments" / "collect_flipped.py"
        if cf.exists():
            cmd = [sys.executable, str(cf), str(args.input_dir)]
            if args.baseline:
                cmd.append(args.baseline)
            sys.stdout.flush()  # so the flipped breakdown prints after the merge summary
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
