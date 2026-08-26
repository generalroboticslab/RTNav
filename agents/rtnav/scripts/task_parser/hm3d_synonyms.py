"""Standalone test: synonym expansion for the 6 HM3D ObjectNav targets.

Calls TaskParser.parse_target() directly so the printed output is exactly
what the runtime agent sees in expanded_objs.

Pipeline: exact target, conservative LLM aliases, embedding filter, strict
yes/no VLM filter.

Requires a running vLLM server.

Usage (inside the agent container):

  python agents/rtnav/scripts/task_parser/hm3d_synonyms.py
  python agents/rtnav/scripts/task_parser/hm3d_synonyms.py --targets bed couch
"""

import argparse
import sys
from typing import List

# HM3D ObjectNav v1 goal categories.
HM3D_TARGETS: List[str] = [
    "chair",
    "bed",
    "potted plant",
    "toilet",
    "tv",
    "couch",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--targets",
        nargs="+",
        default=HM3D_TARGETS,
        help="Override the target list (default: 6 HM3D goals).",
    )
    args = ap.parse_args()

    from rtnav.task.task_parser.synonyms import SynonymModel
    from rtnav.task.task_parser.task_parser import TaskParser

    print("[test] Loading SynonymModel …")
    model = SynonymModel()
    print(f"[test] {len(model.candidates)} OBJECT_CLASSES candidates loaded\n")

    print("[test] Loading TaskParser …")
    parser = TaskParser(synonym_model=model)
    print("[test] TaskParser ready\n")

    for target in args.targets:
        print("=" * 70)
        print(f"TARGET: {target!r}")
        print("=" * 70)

        expanded = parser.parse_target(target)

        # parse_target returns {exact target: [target, aliases]}.
        primary = list(expanded.keys())
        print(f"  primary objects: {primary}")
        for p in primary:
            syns = expanded[p]
            print(f"  {p!r} → {syns}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
