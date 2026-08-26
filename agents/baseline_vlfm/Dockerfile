# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=rt-ovn-env:latest
FROM ${BASE_IMAGE}

USER root
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/.cache/huggingface \
    XDG_CACHE_HOME=/opt/.cache \
    TORCH_HOME=/opt/.cache/torch \
    LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /opt

# Inherit the exact Habitat 0.2.4/Habitat-Baselines stack used by the Thor
# environment image.  The agent does not run a simulator, but VLFM's released
# policy wrapper and PointNav checkpoint both depend on those APIs.
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl tmux pkg-config \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.10's cuBLAS wheel fails even for a standalone Linear layer on
# Jetson Thor.  2.11 is the first CUDA 13 wheel validated on this device.
# This is the only model-runtime version substitution from upstream VLFM.
RUN pip install --upgrade \
      torch==2.11.0+cu130 torchvision==0.26.0+cu130 \
      --index-url https://download.pytorch.org/whl/cu130

COPY agents/utils/ros_helpers.py    /opt/rt_ovn/utils/ros_helpers.py
COPY agents/utils/robot_api.py      /opt/rt_ovn/utils/robot_api.py
COPY agents/utils/video_recorder.py /opt/rt_ovn/utils/video_recorder.py
COPY agents/utils/base_agent.py     /opt/rt_ovn/utils/base_agent.py
COPY agents/utils/cyclonedds.xml    /opt/rt_ovn/utils/cyclonedds.xml
COPY agents/utils/entrypoint.sh      /opt/entrypoint.sh
RUN chmod 755 /opt/entrypoint.sh

# README install order: GroundingDINO + LAVIS, then the VLFM package.  Habitat
# itself is inherited from the base image so the agent uses the same build as
# the simulator and can load the released PointNav checkpoint unchanged.

# LAVIS declares `decord` for video datasets, but decord publishes no ARM64
# wheel and VLFM uses only its image BLIP-2 path.  Install LAVIS without the
# blanket extras and provide the image-model runtime dependencies explicitly.
RUN pip install "spacy<3.8" && \
    git clone https://github.com/IDEA-Research/GroundingDINO.git /opt/GroundingDINO && \
    cd /opt/GroundingDINO && \
    git checkout eeba084341aaa454ce13cb32fa7fd9282fc73a67 && \
    sed -i \
      -e 's/value\.type()\.is_cuda()/value.is_cuda()/g' \
      -e 's/value\.type()/value.scalar_type()/g' \
      groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu && \
    TORCH_CUDA_ARCH_LIST=11.0 pip install . && \
    pip install --no-deps salesforce-lavis==1.0.2 && \
    pip install \
      contexttimer einops fairscale==0.4.4 ftfy iopath omegaconf \
      pandas plotly pycocotools python-magic scikit-image sentencepiece \
      timm==0.4.12 transformers==4.26.0 webdataset

COPY agents/baseline_vlfm/vlfm/README.md /opt/vlfm/README.md
COPY agents/baseline_vlfm/vlfm/pyproject.toml /opt/vlfm/pyproject.toml
COPY agents/baseline_vlfm/vlfm/setup.py /opt/vlfm/setup.py
COPY agents/baseline_vlfm/vlfm/config /opt/vlfm/config
COPY agents/baseline_vlfm/vlfm/scripts /opt/vlfm/scripts
COPY agents/baseline_vlfm/vlfm/vlfm /opt/vlfm/vlfm

ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
RUN pip install -e /opt/vlfm

# imageio-ffmpeg is needed by agents/utils/video_recorder.py. Force the
# bundled binary fetch at build time so a missing/corrupt download fails
# the build instead of the eval.
RUN pip install "imageio>=2.31" "imageio-ffmpeg>=0.5" && \
    python -c "import imageio_ffmpeg; print('imageio-ffmpeg binary at:', imageio_ffmpeg.get_ffmpeg_exe())"

# YOLOv7 is imported from its source tree by vlfm.vlm.yolov7.
RUN git clone https://github.com/WongKinYiu/yolov7.git /opt/yolov7 && \
    git -C /opt/yolov7 checkout a207844b1ce82d204ab36d87d496728d3d2348e7 && \
    ln -s /opt/yolov7 /opt/vlfm/yolov7

# Cache model assets that are independent of mounted baseline data.
RUN mkdir -p "${HF_HOME}" "${TORCH_HOME}/hub/checkpoints" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='bert-base-uncased', revision='86b5e0934494bd15c9632b12f734a8a67f723594', cache_dir='${HF_HOME}'); print('bert-base-uncased downloaded')"

RUN wget -q -O "${TORCH_HOME}/hub/checkpoints/eva_vit_g.pth" \
      https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth && \
    wget -q -O "${TORCH_HOME}/hub/checkpoints/blip2_pretrained.pth" \
      https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth

# LAVIS imports decord at package import time even for image-only models.
# Decord has no ARM64 wheel; this shim keeps image paths usable and raises a
# clear error if an unsupported video path is ever requested.
COPY agents/baseline_vlfm/compat/decord /opt/compat/decord
RUN pip install ipython pycocoevalcap && \
    PYTHONPATH=/opt/compat:/opt/vlfm python -c \
      "import habitat, habitat_sim; from lavis.models import load_model_and_preprocess; from vlfm.policy.habitat_policies import HabitatITMPolicyV2; print('VLFM Habitat policy imports OK')"

COPY agents/baseline_vlfm/agent_node.py /opt/rt_ovn/agent_node.py

ENV PYTHONPATH=/opt/compat:/opt/rt_ovn:/opt/rt_ovn/utils:/opt/vlfm:/opt/GroundingDINO:/opt/yolov7:${PYTHONPATH}
WORKDIR /opt/vlfm

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["/bin/bash"]
