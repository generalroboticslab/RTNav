"""Local paths for models used by RTNav."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OWLV2_MODEL_DIR = str(PACKAGE_ROOT / "modules/perception/weights/owlv2/base")
QWEN_MODEL_DIR = str(PACKAGE_ROOT / "task/task_parser/weights/qwen3.5/9b-4bit")
QWEN_4B_MODEL_DIR = str(PACKAGE_ROOT / "task/task_parser/weights/qwen3.5/4b-4bit")
GEMMA_MODEL_DIR = str(PACKAGE_ROOT / "task/task_parser/weights/gemma4/12b-int4")
EMBEDDING_MODEL_DIR = str(PACKAGE_ROOT / "task/task_parser/weights/embedding/gemma-300m")


_LLM_MODEL_DIRS = {
    "qwen": QWEN_MODEL_DIR,
    "qwen3.5": QWEN_MODEL_DIR,
    "qwen3.5-4b": QWEN_4B_MODEL_DIR,
    "gemma": GEMMA_MODEL_DIR,
    "gemma4": GEMMA_MODEL_DIR,
}


def get_llm_model_dir() -> str:
    """Resolve the configured task-parser LLM checkpoint directory."""
    model_name = os.environ.get("RTNAV_LLM", "qwen3.5").strip().lower()
    try:
        return _LLM_MODEL_DIRS[model_name]
    except KeyError as exc:
        choices = ", ".join(("qwen3.5", "qwen3.5-4b", "gemma4"))
        raise ValueError(f"RTNAV_LLM must be one of: {choices} (got {model_name!r})") from exc
