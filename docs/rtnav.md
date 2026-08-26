# RTNav Setup

Run all commands from the repository root.

## Data


```bash
conda activate rtnav
export DATA_DIR=/absolute/path/to/data
```

## Models

Accept the
[EmbeddingGemma usage terms](https://huggingface.co/google/embeddinggemma-300m),
then download RTNav's Hugging Face models:

```bash
HF_TOKEN=hf_your_token python agents/rtnav/rtnav/download_models.py
```

Download MobileSAM and PointNav checkpoints into `DATA_DIR`:

```bash
wget -nc -P "$DATA_DIR" \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
wget -nc -P "$DATA_DIR" \
  https://raw.githubusercontent.com/rai-opensource/vlfm/main/data/pointnav_weights.pth
```

After model downloading, follow the [main doc](../README.md) to build the docker container for RTNav.


## RTNav specific evaluation arguments


| Option | Values |
|---|---|
| `--llm` | `qwen3.5`, `qwen3.5-4b`, `gemma4` |
| `--video` | `none`, `display`, `save` |

## Real-time Visualization

Pass visualization flags to RTNav through `RTNAV_ARGS`:

```bash
RTNAV_ARGS="--map_viz_web --det_viz_web" \
DATA_DIR="$DATA_DIR" bash agents/launch_parallel.sh \
  --baseline rtnav --benchmark hm3d_v2 \
  --episodes 1 --workers 1 --gpus 0 --launch
```

| Flag | Output |
|---|---|
| `--map_viz_web` | Map and frontier view at `http://localhost:8766` |
| `--det_viz_web` | Detector view at `http://localhost:8765` |
| `--rr_viz` | Rerun scene graph at `127.0.0.1:9876` |
| `--record_decisions` | Decision records in the episode output directory |

