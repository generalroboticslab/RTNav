"""Parse the env's goal info into shared_state.task fields."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtnav.core.shared_state import SharedState

from rtnav.config.detection_classes import OBJECT_CLASSES
from rtnav.task.task_parser.synonyms import SynonymModel
from rtnav.task.task_parser.task_parser import TaskParser


class GoalParser:
    """Wraps TaskParser. Constructed after vLLM is ready; call set_goal each episode."""

    def __init__(self, task_parser):
        self._task_parser = task_parser

    @classmethod
    def build(cls) -> "GoalParser":
        """Load TaskParser (requires vLLM to already be running)."""
        parser = TaskParser(
            synonym_model=SynonymModel(candidates=OBJECT_CLASSES),
        )
        parser.parse_target("cup")  # warmup
        return cls(parser)

    def set_goal(self, shared_state: "SharedState", info: dict) -> str:
        target = str(info.get("target") or "").strip()
        if not target:
            raise ValueError(f"set_goal: missing 'target' in info={info!r}")

        gate = shared_state.inference
        gate.pause_for_vllm()
        try:
            object_synonyms = self._task_parser.parse_target(target)
        finally:
            gate.resume_after_vllm()
        if not object_synonyms:
            raise RuntimeError("TaskParser returned no target aliases")

        # synonym.lower() → canonical  (used by detector and scene graph)
        lookup = {c.lower(): c for c in object_synonyms}
        for canonical, synonyms in object_synonyms.items():
            for s in synonyms:
                lookup[str(s).lower()] = canonical

        with shared_state.lock:
            task = shared_state.task
            task.synonym_to_canonical = lookup
            task.goal_category = target
            task.complete = False
        shared_state.task_ready.set()

        total = sum(len(s) for s in object_synonyms.values())
        print(f"[goal] {target!r} — {len(object_synonyms)} primary, {total} synonyms")
        return target
