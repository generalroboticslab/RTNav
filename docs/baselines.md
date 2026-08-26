# Baseline Setup

## Build

Build the shared simulator and agent base images once:

```bash
docker build -f docker/Dockerfile.env -t rt-ovn-env:latest .
docker build -f docker/Dockerfile.agent -t rt-ovn-agent-base:latest agents
```

Build the selected baseline image. Replace `vlfm` with `l3mvn`, `trihelper`,
`gamap`, `openfmnav`, or `beliefmapnav`:

```bash
BASELINE=vlfm
docker build -f "agents/baseline_${BASELINE}/Dockerfile" \
  -t "${BASELINE}-agent:latest" .
```

## Evaluation

All baselines are launched through `agents/launch_parallel.sh`:

```bash
DATA_DIR="$DATA_DIR" bash agents/launch_parallel.sh \
  --baseline vlfm --benchmark hm3d_v2 --mode async \
  --episodes 1 --workers 1 --gpus 0 --launch
```

Valid baseline names are `vlfm`, `l3mvn`, `trihelper`, `gamap`, `openfmnav`,
and `beliefmapnav`. Increase `--episodes`, `--workers`, and `--gpus` for a full
evaluation.

## Download models required by baselines

Download only the files required by the baseline you plan to evaluate.

### VLFM

```bash
wget -nc -P "$DATA_DIR" \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
wget -nc -P "$DATA_DIR" \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -nc -P "$DATA_DIR" \
  https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
wget -nc -P "$DATA_DIR" \
  https://raw.githubusercontent.com/rai-opensource/vlfm/main/data/pointnav_weights.pth
```

### L3MVN

```bash
mkdir -p agents/baseline_l3mvn/L3MVN/RedNet/model
gdown 1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv \
  -O agents/baseline_l3mvn/L3MVN/RedNet/model/rednet_semmap_mp3d_40.pth
```

### TriHelper

TriHelper uses the same RedNet checkpoint as L3MVN:

```bash
mkdir -p agents/baseline_trihelper/TriHelper/RedNet/model
cp agents/baseline_l3mvn/L3MVN/RedNet/model/rednet_semmap_mp3d_40.pth \
  agents/baseline_trihelper/TriHelper/RedNet/model/
hf download Qwen/Qwen-VL-Chat-Int4 \
  --local-dir agents/baseline_trihelper/TriHelper/pretrained_models/Qwen-VL
```

### GAMap

```bash
mkdir -p agents/baseline_gamap/GAMap/RedNet/model
gdown 1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv \
  -O agents/baseline_gamap/GAMap/RedNet/model/rednet_semmap_mp3d_40.pth
```

### OpenFMNav

OpenFMNav requires `OPENAI_API_KEY` when launched.

```bash
mkdir -p agents/baseline_openfmnav/OpenFMNav/Grounded_SAM
wget -nc -P agents/baseline_openfmnav/OpenFMNav/Grounded_SAM \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -nc -P agents/baseline_openfmnav/OpenFMNav/Grounded_SAM \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### BeliefMapNav

BeliefMapNav reuses the shared MobileSAM, GroundingDINO, YOLOv7, and PointNav
weights. It needs one local RedNet checkpoint and `OPENAI_API_KEY`:

```bash
mkdir -p agents/baseline_beliefmapnav/BeliefMapNav/data
gdown 1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv \
  -O agents/baseline_beliefmapnav/BeliefMapNav/data/rednet_semmap_mp3d_40.pth
```
