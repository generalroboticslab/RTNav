"""Creates a minimal top-level OVON episodes .json.gz from the content/ directory.
Also removes macOS resource fork files (._*) that cause gzip errors."""

import gzip
import json
import os
import sys


def main():
    if len(sys.argv) > 1:
        content_dir = sys.argv[1]
    else:
        content_dir = "data/datasets/ovon/hm3d/val_unseen/content"

    output_dir = os.path.dirname(content_dir)
    output_path = os.path.join(output_dir, "val_unseen.json.gz")

    # Remove macOS resource fork files that cause gzip errors
    removed = 0
    for f in os.listdir(content_dir):
        if f.startswith("._"):
            os.remove(os.path.join(content_dir, f))
            removed += 1
    if removed:
        print(f"Removed {removed} macOS resource fork files (._*)")

    # Collect all scene IDs from the content directory
    scene_files = sorted(f for f in os.listdir(content_dir) if f.endswith(".json.gz"))
    print(f"Found {len(scene_files)} scene files in {content_dir}")

    # Verify each content file is valid gzip
    bad_files = []
    for sf in scene_files:
        path = os.path.join(content_dir, sf)
        try:
            with gzip.open(path, "rb") as gf:
                gf.read(1)
        except Exception as e:
            print(f"  WARNING: corrupt file {sf}: {e}")
            bad_files.append(path)

    for bf in bad_files:
        print(f"  Removing corrupt file: {bf}")
        os.remove(bf)

    if bad_files:
        scene_files = sorted(f for f in os.listdir(content_dir) if f.endswith(".json.gz"))
        print(f"After cleanup: {len(scene_files)} valid scene files")

    # Create minimal top-level file with empty episodes list
    data = {"episodes": []}
    json_bytes = json.dumps(data).encode("utf-8")

    with gzip.open(output_path, "wb") as f:
        f.write(json_bytes)

    # Verify the file is readable
    with gzip.open(output_path, "rb") as f:
        content = json.loads(f.read().decode("utf-8"))
        assert "episodes" in content

    print(f"Created and verified {output_path} ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    main()
