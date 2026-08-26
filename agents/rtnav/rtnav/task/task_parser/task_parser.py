"""Expand one ObjectNav target into conservative detector aliases."""

from __future__ import annotations

import ast
import math
import re

from rtnav.config.detection_classes import LVIS_CATEGORIES, OBJECT_CLASSES
from rtnav.utils.vllm_utils import get_vllm_base_url, get_vllm_model_id

EMBEDDING_THRESHOLD = 0.82
VLM_KEEP_THRESHOLD = 0.5


def _normalize(label: str) -> str:
    return str(label).replace("_", " ").strip().lower()


_OBJECT_CLASS_BY_LABEL = {_normalize(label): label for label in OBJECT_CLASSES}
_LVIS_ALIASES: dict[str, list[str]] = {}
for category in LVIS_CATEGORIES:
    object_class = _OBJECT_CLASS_BY_LABEL.get(_normalize(category.get("name", "")))
    if object_class is None:
        continue
    for alias in [category.get("name", ""), *category.get("synonyms", [])]:
        key = _normalize(alias)
        if key:
            _LVIS_ALIASES.setdefault(key, [])
            if object_class not in _LVIS_ALIASES[key]:
                _LVIS_ALIASES[key].append(object_class)


def _top_logprobs(choice):
    logprobs = choice.logprobs
    if logprobs is None:
        raise RuntimeError("vLLM response did not include token logprobs")
    if getattr(logprobs, "top_logprobs", None):
        return logprobs.top_logprobs[0].items()
    if getattr(logprobs, "content", None):
        return ((entry.token, entry.logprob) for entry in logprobs.content[0].top_logprobs)
    raise RuntimeError("vLLM response did not include first-token logprobs")


def _yes_probability(choice) -> float:
    yes = no = 0.0
    for token, logprob in _top_logprobs(choice):
        probability = math.exp(float(logprob))
        token = str(token).strip().lower()
        if token == "yes":
            yes += probability
        elif token == "no":
            no += probability
    return yes / (yes + no) if yes + no > 0.0 else 0.0


class TaskParser:
    """Resolve aliases for the literal task ``Find the {target}.``"""

    def __init__(
        self,
        synonym_model=None,
        dummy: bool = False,
    ):
        if synonym_model is None:
            from rtnav.task.task_parser.synonyms import SynonymModel

            synonym_model = SynonymModel()
        self.synonym_model = synonym_model
        self.client = None
        if not dummy:
            from openai import OpenAI

            self.client = OpenAI(base_url=get_vllm_base_url(), api_key="EMPTY", timeout=120.0)

    def _query(self, prompt: str, *, max_tokens: int = 64):
        if self.client is None:
            return None
        return self.client.chat.completions.create(
            model=get_vllm_model_id(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    def _common_aliases(self, target: str) -> list[str]:
        response = self._query(
            "Given one indoor object, return a Python list containing zero to two "
            "common, unambiguous alternative names for exactly the same visible "
            "physical object. Do not return room/room containing the object/correlatetd or associated objects, "
            f"or explanations. Object: {target!r}"
        )
        if response is None:
            return []
        text = str(response.choices[0].message.content or "")
        match = re.search(r"\[[\s\S]*?\]", text)
        if match is None:
            return []
        try:
            aliases = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            return []
        if not isinstance(aliases, list):
            return []
        target_norm = _normalize(target)
        return list(
            dict.fromkeys(
                _normalize(alias)
                for alias in aliases[:2]
                if _normalize(alias) and _normalize(alias) != target_norm
            )
        )

    def _same_object_probability(self, target: str, candidate: str) -> float:
        if self.client is None:
            return 0.0
        response = self.client.chat.completions.create(
            model=get_vllm_model_id(),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"""A robot is asked to search for: {target!r}, the indoor object, in a navigation task. It detected the object {candidate!r}. The detector is not perfect, it might label the {target!r} as {candidate!r} if they are visually similar, but it might also label a wrong object as {candidate!r} if only a small part of it looks like it. Think about the intent of the user in the context of indoor object navigation when they asked for {target!r}. Will the user be satisfied with {candidate!r}? Respond with one word: yes/no.\nAnswer:\n\n"""
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return _yes_probability(response.choices[0])

    def filter_object_class_candidates(self, target: str, candidates: list[str]) -> list[str]:
        target_norm = _normalize(target)
        kept = []
        for candidate in dict.fromkeys(_normalize(item) for item in candidates):
            if not candidate or candidate == target_norm:
                continue
            if round(self._same_object_probability(target, candidate), 3) > VLM_KEEP_THRESHOLD:
                kept.append(candidate)
        return kept

    def parse_target(self, target: str):
        """Return detector aliases for one target."""
        target = str(target).strip()
        if not target:
            raise ValueError("Target must be non-empty")

        candidate_vocab = list(
            dict.fromkeys(
                [
                    *OBJECT_CLASSES,
                    *_LVIS_ALIASES.get(_normalize(target), []),
                    *self._common_aliases(target),
                ]
            )
        )
        candidates = self.synonym_model.filter_candidates(
            target, candidate_vocab, threshold=EMBEDDING_THRESHOLD
        )
        aliases = self.filter_object_class_candidates(target, candidates)
        expanded = {target: [target, *aliases]}
        print(f"[TaskParser] {target!r} aliases: {aliases}")
        return expanded

    def parse_task(self, task_description: str):
        """Parse the one accepted sentence shape."""
        match = re.fullmatch(r"\s*Find\s+the\s+(.+?)\s*\.\s*", str(task_description), re.IGNORECASE)
        if match is None:
            raise ValueError("RTNav supports one task form only: 'Find the {target}.'")
        return self.parse_target(match.group(1))
