"""Generates a CLIP text embedding cache for OVON goal categories.

Reads all OVON episode files, extracts unique goal category names,
computes CLIP text embeddings for each, and saves a pickle cache.
"""

import glob
import gzip
import json
import os
import pickle
import sys

import clip
import torch


def main():
    episodes_dir = sys.argv[1] if len(sys.argv) > 1 else "data/datasets/ovon/hm3d/val_unseen/content"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/clip_embeddings/ovon_hm3d_cache.pkl"

    # Collect all unique category names from episodes
    categories = set()
    episode_files = glob.glob(os.path.join(episodes_dir, "*.json.gz"))
    print(f"Scanning {len(episode_files)} episode files in {episodes_dir}...")

    for ep_file in episode_files:
        try:
            with gzip.open(ep_file, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))
        except Exception as e:
            print(f"  Skipping {ep_file}: {e}")
            continue

        # Extract categories from goals_by_category keys
        if "goals_by_category" in data:
            for key in data["goals_by_category"]:
                # Keys are like "scene.glb_category_name"
                parts = key.split("_", 1)
                if len(parts) == 2:
                    cat = parts[1]
                else:
                    # Try splitting by last dot then underscore
                    dot_parts = key.rsplit(".", 1)
                    if len(dot_parts) == 2:
                        cat = dot_parts[1].split("_", 1)[-1] if "_" in dot_parts[1] else dot_parts[1]
                    else:
                        cat = key
                categories.add(cat)

        # Also extract from episodes directly
        if "episodes" in data:
            for ep in data["episodes"]:
                if "object_category" in ep:
                    categories.add(ep["object_category"])

    print(f"Found {len(categories)} unique categories")
    if len(categories) == 0:
        print("No categories found! Check your episode files.")
        sys.exit(1)

    # Sort for reproducibility
    categories = sorted(categories)
    print("Sample categories:", categories[:10])

    # Load CLIP model and compute embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)

    cache = {}
    prompt_template = "Find and go to {category}"

    print("Computing CLIP embeddings...")
    with torch.no_grad():
        for cat in categories:
            text = prompt_template.format(category=cat)
            tokens = clip.tokenize([text]).to(device)
            embedding = model.encode_text(tokens).cpu().numpy().flatten()
            cache[cat] = embedding

    # Save cache
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"Saved cache with {len(cache)} entries to {output_path}")


if __name__ == "__main__":
    main()
