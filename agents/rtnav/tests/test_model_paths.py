import pytest

from rtnav.config.model_paths import (
    GEMMA_MODEL_DIR,
    QWEN_4B_MODEL_DIR,
    QWEN_MODEL_DIR,
    get_llm_model_dir,
)


@pytest.mark.parametrize("value", [None, "qwen", "QWEN3.5"])
def test_qwen_model_selection(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("RTNAV_LLM", raising=False)
    else:
        monkeypatch.setenv("RTNAV_LLM", value)

    assert get_llm_model_dir() == QWEN_MODEL_DIR


def test_qwen_4b_model_selection(monkeypatch):
    monkeypatch.setenv("RTNAV_LLM", "QWEN3.5-4B")

    assert get_llm_model_dir() == QWEN_4B_MODEL_DIR


@pytest.mark.parametrize("value", ["gemma", "GEMMA4"])
def test_gemma_model_selection(monkeypatch, value):
    monkeypatch.setenv("RTNAV_LLM", value)

    assert get_llm_model_dir() == GEMMA_MODEL_DIR


def test_unknown_model_is_rejected(monkeypatch):
    monkeypatch.setenv("RTNAV_LLM", "unknown")

    with pytest.raises(ValueError, match="qwen3.5, qwen3.5-4b, gemma4"):
        get_llm_model_dir()
