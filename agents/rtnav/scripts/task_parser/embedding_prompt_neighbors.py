"""Probe task-parser synonym embeddings under different prompt templates.

Usage inside the RtNav agent container:
    python /opt/rtnav/scripts/task_parser/embedding_prompt_neighbors.py
"""

import argparse
import os

import torch
from sentence_transformers import SentenceTransformer

from rtnav.config.detection_classes import OBJECT_CLASSES
from rtnav.config.model_paths import EMBEDDING_MODEL_DIR

PROMPTS = [
    "{}",
    "meaning: {}",
    "object: {}",
    "target object: {}",
    "find {}",
    "find the {}",
    "I found {}",
    "a photo of a {}",
    "a detected {}",
    "the object is {}",
]

DEFAULT_TARGETS = ["potted plant", "tv", "couch", "toilet", "chair", "bed"]

RELEVANT_CANDIDATES = {
    "potted plant": {"flowerpot"},
    "tv": {"television set"},
    "couch": {"sofa"},
    "toilet": {"toilet"},
    "chair": {"chair", "armchair", "deck chair"},
    "bed": {"bed"},
}


def _format(prompt: str, word: str) -> str:
    return prompt.format(word)


def ranked_neighbors(model, candidates: list[str], target: str, prompt: str, top_k: int):

    query = _format(prompt, target)
    candidate_texts = [_format(prompt, candidate) for candidate in candidates]

    with torch.inference_mode():
        query_embedding = model.encode(query)
        candidate_embeddings = model.encode(candidate_texts)
        similarities = model.similarity(query_embedding, candidate_embeddings)[0]

    order = torch.argsort(similarities, descending=True)[:top_k]
    return [(candidates[i], float(similarities[i])) for i in order]


def first_relevant(ranked: list[tuple[str, float]], relevant: set[str]):
    for rank, (word, score) in enumerate(ranked, start=1):
        if word in relevant:
            return rank, word, score
    return None, None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    targets = args.targets or args.target or DEFAULT_TARGETS

    assert "I found {}" in PROMPTS
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidates = list(OBJECT_CLASSES)

    print(f"embedding_model={EMBEDDING_MODEL_DIR}")
    print(f"device={device}")
    print(f"targets={targets!r}")
    print(f"num_candidates={len(candidates)}")
    print()

    model = SentenceTransformer(EMBEDDING_MODEL_DIR, device="cpu").float()
    if device != "cpu":
        model = model.to(device)

    summary = {prompt: [] for prompt in PROMPTS}

    for target in targets:
        relevant = RELEVANT_CANDIDATES.get(target, {target})
        print("=" * 80)
        print(f"target={target!r} relevant={sorted(relevant)}")
        print()

        for prompt in PROMPTS:
            ranked = ranked_neighbors(model, candidates, target, prompt, args.top_k)
            rel_rank, rel_word, rel_score = first_relevant(ranked, relevant)
            summary[prompt].append((target, rel_rank, rel_word, rel_score))

            print("-" * 80)
            print(f"prompt={prompt!r}")
            if rel_rank is None:
                print(f"first_relevant=not_in_top_{args.top_k}")
            else:
                print(f"first_relevant=rank {rel_rank}: {rel_word} {rel_score:.6f}")
            for rank, (word, score) in enumerate(ranked, start=1):
                marker = "*" if word in relevant else " "
                print(f"{rank:02d}{marker} {word:<32} {score:.6f}")
            print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    rows = []
    for prompt, results in summary.items():
        reciprocal_ranks = [1.0 / r for _, r, _, _ in results if r is not None]
        hits = len(reciprocal_ranks)
        mrr = sum(reciprocal_ranks) / len(results)
        hits_at_5 = sum(1 for _, r, _, _ in results if r is not None and r <= 5)
        mean_score = sum(s for _, _, _, s in results if s is not None) / hits if hits else 0.0
        rows.append((mrr, hits_at_5, hits, mean_score, prompt, results))

    rows.sort(reverse=True)
    print("prompt                         mrr     hits@5  hits  mean_relevant_score")
    for mrr, hits_at_5, hits, mean_score, prompt, _ in rows:
        print(
            f"{prompt:<30} {mrr:.4f}  {hits_at_5:>2}/{len(targets)}    {hits:>2}/{len(targets)}   {mean_score:.6f}"
        )

    print()
    print("best_by_target")
    for target in targets:
        target_rows = []
        for prompt, results in summary.items():
            for result_target, rank, word, score in results:
                if result_target == target and rank is not None:
                    target_rows.append((rank, -score, prompt, word, score))
        target_rows.sort()
        if target_rows:
            rank, _, prompt, word, score = target_rows[0]
            print(f"{target:<8} {prompt!r:<20} rank={rank:<2} {word:<16} {score:.6f}")
        else:
            print(f"{target:<8} no relevant candidate in top {args.top_k}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
