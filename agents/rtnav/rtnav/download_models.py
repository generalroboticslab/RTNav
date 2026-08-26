#!/usr/bin/env python3
"""Download the model weights the RTNav agent needs (from HuggingFace)."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download

_PKG = Path(__file__).resolve().parent  # agents/rtnav/rtnav

# name -> (HuggingFace repo, destination dir under this package)
MODELS = {
    "owlv2": ("google/owlv2-base-patch16-ensemble", "modules/perception/weights/owlv2/base"),
    "qwen": ("cyankiwi/Qwen3.5-9B-AWQ-4bit", "task/task_parser/weights/qwen3.5/9b-4bit"),
    "embedding": ("google/embeddinggemma-300m", "task/task_parser/weights/embedding/gemma-300m"),
    "gemma": ("cyankiwi/gemma-4-12B-it-AWQ-INT4", "task/task_parser/weights/gemma4/12b-int4"),
}


def weights_present(model_dir: Path) -> bool:
    """True if the actual weight files (not just config/tokenizer) are on disk."""
    if not model_dir.is_dir():
        return False
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():  # sharded model: every shard the index lists must exist
        try:
            shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        except Exception:
            return False
        return bool(shards) and all((model_dir / s).is_file() for s in shards)
    return any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))


def download(name: str) -> None:
    repo, subdir = MODELS[name]
    model_dir = _PKG / subdir
    if weights_present(model_dir):
        print(f"[{name}] already downloaded: {model_dir}")
        return

    print(f"[{name}] downloading {repo} -> {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=str(model_dir))


def main() -> None:
    for name in MODELS:
        download(name)


if __name__ == "__main__":
    main()
