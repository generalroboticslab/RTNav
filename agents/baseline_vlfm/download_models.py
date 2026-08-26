import json
import os
from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="google/owlv2-base-patch16-ensemble",
    local_dir="vlfm/data/owlv2/owlv2-base-patch16-ensemble",
)

# snapshot_download(
#     repo_id="google/siglip2-base-patch16-224",
#     local_dir="ovnav/modules/perception/weights/siglip2/base",
# )

# # snapshot_download(
# #     repo_id="google/gemma-3-12b-it",
# #     local_dir="ovnav/task/task_parser/weights/gemma3/12b-it",
# # )

# # # Text embedding model for scene graph target matching
# snapshot_download(
#     repo_id="sentence-transformers/all-MiniLM-L6-v2",
#     local_dir="ovnav/task/weights/minilm-l6-v2",
# )

# OPTIONAL:
# largerobject detector
# snapshot_download(repo_id="google/owlv2-large-patch14-ensemble", local_dir="ovnav/perception/weights/owlv2/large")
# CLIP-based evaluation
# snapshot_download(repo_id="openai/clip-vit-base-patch32", local_dir="ovnav/eval/weights/clip-base")
# snapshot_download(
#     repo_id="google/gemma-3-4b-it-qat-q4_0-gguf", local_dir="ovnav/task/task_parser/weights/gemma3/4b-it-4bit"
# )
# snapshot_download(
#     repo_id="google/gemma-3n-E4B-it", local_dir="ovnav/task/task_parser/weights/gemma3n/e4b-it"
# )

# snapshot_download(
#     repo_id="abhishekchohan/gemma-3-12b-it-quantized-W4A16",
#     # repo_id="RedHatAI/gemma-3-12b-it-quantized.w4a16",
#     # repo_id="pytorch/gemma-3-12b-it-QAT-INT4",
#     # repo_id="pytorch/gemma-3-12b-it-AWQ-INT4",
#     local_dir="ovnav/task/task_parser/weights/gemma3/12b-it-4bit",
# )

# snapshot_download(
#     repo_id="facebook/sam3",
#     local_dir="ovnav/modules/perception/weights/sam3",
# )


# snapshot_download(
#     repo_id="unsloth/Ministral-3-14B-Instruct-2512-bnb-4bit",
#     # repo_id="mlx-community/Ministral-3-14B-Instruct-2512-4bit",
#     local_dir="ovnav/task/task_parser/weights/ministral3/14b-it-4bit",
# )


# snapshot_download(
#     repo_id="unsloth/Ministral-3-8B-Instruct-2512-bnb-4bit",
#     local_dir="ovnav/task/task_parser/weights/ministral3/8b-it-4bit",
# )

# def _patch_ministral3_text_config(model_dir: str) -> None:
#     """Add missing 'architectures' to text_config so vLLM can resolve the inner LM."""
#     config_path = os.path.join(model_dir, "config.json")
#     if not os.path.exists(config_path):
#         return
#     with open(config_path, "r") as f:
#         config = json.load(f)
#     text_cfg = config.get("text_config", {})
#     if text_cfg and not text_cfg.get("architectures"):
#         text_cfg["architectures"] = ["MistralForCausalLM"]
#         config["text_config"] = text_cfg
#         with open(config_path, "w") as f:
#             json.dump(config, f, indent=2)
#         print(f"[patch] Added architectures to text_config in {config_path}")


# for ministral_dir in [
#     "ovnav/task/task_parser/weights/ministral3/8b-it-4bit",
#     "ovnav/task/task_parser/weights/ministral3/14b-it-4bit",
# ]:
#     _patch_ministral3_text_config(ministral_dir)