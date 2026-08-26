from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="google/owlv2-base-patch16-ensemble",
    local_dir="vlfm/data/owlv2/owlv2-base-patch16-ensemble",
)
