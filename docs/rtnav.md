# RTNav Setup

Run all commands from the repository root.

## Data

Prepare the datasets first using [datasets.md](datasets.md), then export the
same absolute path for every command:

```bash
export DATA_DIR=/absolute/path/to/data
```

## Models

Accept the
[EmbeddingGemma usage terms](https://huggingface.co/google/embeddinggemma-300m),
then download RTNav's Hugging Face models:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
HF_TOKEN=hf_your_token python agents/rtnav/rtnav/download_models.py
```

Download MobileSAM and PointNav checkpoints into `DATA_DIR`:

```bash
wget -nc -P "$DATA_DIR" \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
wget -nc -P "$DATA_DIR" \
  https://raw.githubusercontent.com/rai-opensource/vlfm/main/data/pointnav_weights.pth
```

## Build

```bash
DATA_DIR="$DATA_DIR" docker compose -f agents/rtnav/docker-compose.yml \
  build env agent
```

For a CUDA 12 host, build the alternate agent image from the repository root:

```bash
docker build -f agents/rtnav/docker/Dockerfile.cuda12 \
  -t rt-ovn-agent:latest .
```

## Evaluation

```bash
DATA_DIR="$DATA_DIR" bash agents/launch_parallel.sh \
  --baseline rtnav --benchmark hm3d_v2 \
  --episodes 1000 --workers 4 --gpus 0,1,2,3 --launch
```

RTNav uses asynchronous evaluation automatically. Results are merged into
`agents/rtnav/logs_parallel/live.json` when all workers finish.

| Option | Values |
|---|---|
| `--benchmark` | `hm3d_v1`, `hm3d_v2`, `ovon` |
| `--episodes` | Total episodes across all workers |
| `--workers` | Number of parallel simulator-agent pairs |
| `--gpus` | Comma-separated GPU IDs |
| `--llm` | `qwen3.5`, `qwen3.5-4b`, `gemma4` |
| `--video` | `none`, `display`, `save` |

## Visualization

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

## Development

```bash
cd agents/rtnav
python3 -m venv ../../.venv
source ../../.venv/bin/activate
python -m pip install -r requirements-dev.txt
make check
```

The available targets are `make test`, `make lint`, `make format`, and
`make check`.
