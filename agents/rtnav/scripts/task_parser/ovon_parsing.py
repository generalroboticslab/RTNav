"""OVON object-goal parsing & synonym test.

Run from inside the agent container:

    cd /opt/rt_ovn
    python agents/rtnav/scripts/task_parser/ovon_parsing.py
"""

import time

from rtnav.config.model_paths import get_llm_model_dir
from rtnav.task.task_parser.synonyms import SynonymModel
from rtnav.task.task_parser.task_parser import TaskParser
from rtnav.utils.vllm_utils import ensure_vllm_server as _ensure_vllm_server

CASES = [
    "Find the picture.",
    "Find the sculpture.",
    "Find the decorative plant.",
]

if __name__ == "__main__":
    model_dir = get_llm_model_dir()

    print(f"\nEnsuring vllm server is running ({model_dir})…")
    _ensure_vllm_server(model_dir)
    print("vllm ready.\n")

    print("Loading TaskParser…")
    t0 = time.time()
    parser = TaskParser()
    print(f"Ready in {time.time() - t0:.1f}s\n")

    synonym_model = SynonymModel()

    results = []
    for task_input in CASES:
        print("=" * 60)
        print(f"Input    : {task_input!r}")
        t0 = time.time()
        target = task_input.removeprefix("Find the ").removesuffix(".")
        expanded = parser.parse_target(target)
        elapsed = time.time() - t0
        print(f"Time     : {elapsed:.2f}s")
        print(f"Expanded : {dict(expanded)}")

        sm_out = {}
        for obj in expanded.keys():
            raw = synonym_model.get_synonyms(obj, threshold=0.70)
            filtered = parser.filter_object_class_candidates(obj, raw)
            sm_out[obj] = filtered
            print(f"SynonymModel({obj!r})  raw={raw}")
            print(f"  → after LLM filter: {filtered}")

        results.append(dict(input=task_input, elapsed=elapsed, expanded=expanded, sm=sm_out))

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    W = 28
    print(
        f"{'Input':<{W}}  {'Time':>5}  {'Primary':<18}  {'LLM synonyms':<30}  OBJECT_CLASSES matches"
    )
    print("-" * 120)
    for r in results:
        for obj, all_syns in r["expanded"].items():
            obj_classes = r["sm"].get(obj, [])
            obj_classes_set = set(obj_classes)
            # LLM synonyms = everything in expanded that is NOT in OBJECT_CLASSES and NOT the obj itself
            llm_syns = sorted(s for s in all_syns if s != obj and s not in obj_classes_set)
            print(
                f"{r['input']:<{W}}  {r['elapsed']:>5.2f}s"
                f"  {obj:<18}"
                f"  {str(llm_syns):<30}"
                f"  {sorted(obj_classes)}"
            )
    print("=" * 60)
