# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=rt-ovn-agent-base:latest
FROM ${BASE_IMAGE}

USER root
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
ARG MAMBA_DOCKERFILE_ACTIVATE=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GAMAP_ROOT=/opt/gamap \
    HF_HOME=/opt/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/.cache/huggingface \
    TORCH_HOME=/opt/.cache/torch \
    FVCORE_CACHE=/opt/.cache/fvcore \
    LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

WORKDIR /opt/gamap

# Agent-side system libraries for GAMap semantic mapping, Detectron2 and lavis.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget ca-certificates build-essential g++ ninja-build \
      libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1 libxml2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade "pip<25" "wheel<0.45" "setuptools<60.0"

# GAMap runs on torch 2.1 (CUDA >12); see agents/baseline_gamap/GAMap/README.md.
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Detectron2 has no wheels -> build from source. setup.py imports torch, so
# torch must be present and build isolation off. No nvcc in the base image, so
# the C++ extension builds CPU-only; GPU ops come from torchvision at runtime.
RUN pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# py3.9-compatible versions of GAMap's requirements (upstream pins are py3.7-era).
RUN pip install \
      "scikit-fmm" \
      "scikit-learn==0.24.2" \
      "scikit-image==0.19.3" \
      "numpy==1.26.4" \
      "Pillow==9.5.0" \
      "ifcfg" \
      "transformers==4.23.1" \
      "tensorboard"

# lavis (BLIP/CLIP) --no-deps so it can't upgrade torch, then its import-chain deps.
RUN pip install --no-deps salesforce-lavis
RUN pip install \
      "numpy==1.26.4" \
      "pandas" \
      "einops" \
      "contexttimer" \
      "sentencepiece" \
      "spacy<3.8" \
      "pycocoevalcap" \
      "opencv-python==4.5.5.64" \
      "timm" \
      "omegaconf" \
      "iopath" \
      "open_clip_torch" \
      "ftfy" \
      "fairscale" \
      "decord" \
      "webdataset" \
      "imageio[ffmpeg]"

# Cache Detectron2's COCO Mask R-CNN weights (independent of mounted data).
RUN mkdir -p "${FVCORE_CACHE}" && \
    python -c "from detectron2.utils.file_io import PathManager; print(PathManager.get_local_path('detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'))" && \
    chmod -R a+rwX /opt/.cache

# Cache lavis CLIP ViT-B-32 (GAMap's frontier-scoring model).
RUN mkdir -p "${TORCH_HOME}" && \
    python -c "from lavis.models import load_model_and_preprocess as L; L('clip_feature_extractor', model_type='ViT-B-32', is_eval=True, device='cpu'); print('clip ViT-B-32 cached')" && \
    chmod -R a+rwX /opt/.cache

COPY agents/baseline_gamap/agent_node.py /opt/rt_ovn/agent_node.py

ENV PYTHONPATH=/opt/rt_ovn:/opt/rt_ovn/utils:/opt/gamap:${PYTHONPATH}

CMD ["/bin/bash"]
