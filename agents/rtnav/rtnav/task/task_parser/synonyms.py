import torch
from sentence_transformers import SentenceTransformer

from rtnav.config.detection_classes import OBJECT_CLASSES
from rtnav.config.model_paths import EMBEDDING_MODEL_DIR


class SynonymModel:
    def __init__(
        self,
        model_name=EMBEDDING_MODEL_DIR,
        candidates=OBJECT_CLASSES,
    ):
        import os

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SynonymModel] Loading embedding model on {device}...")
        # Build on CPU first then move — works on both newer and older
        # transformers/accelerate stacks (direct device= can leave meta tensors).
        self.model = SentenceTransformer(model_name, device="cpu").float()
        if device != "cpu":
            self.model = self.model.to(device)
        self.candidates = list(candidates)
        print(f"[SynonymModel] Encoding {len(self.candidates)} candidates...")
        with torch.inference_mode():
            self.candidate_embeddings = self.model.encode(
                [f"meaning: {c}, an indoor object" for c in self.candidates]
            )
        self.candidate_to_index = {c: i for i, c in enumerate(self.candidates)}
        print("[SynonymModel] Ready")

    def get_synonyms(self, term, threshold=0.82):
        term_embedding = self.model.encode(f"meaning: {term}, an indoor object")
        similarities = self.model.similarity(term_embedding, self.candidate_embeddings)[0]
        return [self.candidates[i] for i, sim in enumerate(similarities) if sim >= threshold]

    def filter_candidates(self, term, candidates, threshold=0.8):
        candidates = [
            str(candidate).replace("_", " ").strip()
            for candidate in candidates
            if str(candidate).strip()
        ]
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            return []

        with torch.inference_mode():
            term_embedding = self.model.encode(f"meaning: {term}, an indoor object")
            scores = {}

            known = [
                (candidate, self.candidate_to_index[candidate])
                for candidate in candidates
                if candidate in self.candidate_to_index
            ]
            if known:
                known_embeddings = self.candidate_embeddings[[idx for _, idx in known]]
                similarities = self.model.similarity(term_embedding, known_embeddings)[0]
                for (candidate, _), sim in zip(known, similarities):
                    scores[candidate] = float(sim)

            extras = [
                candidate for candidate in candidates if candidate not in self.candidate_to_index
            ]
            if extras:
                extra_embeddings = self.model.encode(
                    [f"meaning: {candidate}, an indoor object" for candidate in extras]
                )
                similarities = self.model.similarity(term_embedding, extra_embeddings)[0]
                for candidate, sim in zip(extras, similarities):
                    scores[candidate] = float(sim)

        return [candidate for candidate in candidates if scores[candidate] >= threshold]
