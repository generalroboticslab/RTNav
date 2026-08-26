# Download Baseline Models

PointNav policy is used by VLFM, BeliefMapNav, and RTNav. RedNet is used by L3MVN,
TriHelper, GAMap, and BeliefMapNav. Download the models into the shared
`DATA_DIR`:

```bash
wget -nc -P "$DATA_DIR" \
  https://raw.githubusercontent.com/rai-opensource/vlfm/main/data/pointnav_weights.pth
gdown 1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv \
  -O "$DATA_DIR/rednet_semmap_mp3d_40.pth"
```

Download the remaining baseline-specific checkpoints as needed.

### VLFM

```bash
wget -nc -P "$DATA_DIR" \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
wget -nc -P "$DATA_DIR" \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -nc -P "$DATA_DIR" \
  https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
```

For OVON evaluation, download OWLv2 into the shared data directory:

```bash
hf download google/owlv2-base-patch16-ensemble \
  --local-dir "$DATA_DIR/owlv2/owlv2-base-patch16-ensemble"
```

### TriHelper

```bash
hf download Qwen/Qwen-VL-Chat-Int4 \
  --local-dir agents/baseline_trihelper/TriHelper/pretrained_models/Qwen-VL
```

### OpenFMNav

OpenFMNav requires `OPENAI_API_KEY` to be set when launched.

```bash
mkdir -p agents/baseline_openfmnav/OpenFMNav/Grounded_SAM
wget -nc -P agents/baseline_openfmnav/OpenFMNav/Grounded_SAM \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -nc -P agents/baseline_openfmnav/OpenFMNav/Grounded_SAM \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### BeliefMapNav

BeliefMapNav requires `OPENAI_API_KEY` to be set when launched.
