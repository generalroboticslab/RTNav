# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=rt-ovn-agent-base:latest
FROM ${BASE_IMAGE}

ARG MAMBA_DOCKERFILE_ACTIVATE=1

USER root
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/.cache/huggingface \
    XDG_CACHE_HOME=/opt/.cache \
    TORCH_HOME=/opt/.cache/torch \
    LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /opt

# VLFM runs VLM servers in the agent container and uses EGL/OpenGL helpers.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl ca-certificates tmux ffmpeg build-essential pkg-config ninja-build \
      libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
      libglvnd0 libglvnd-dev libgl1 libglx0 libegl1 libegl-dev \
      libopengl0 libgles2 patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Agent-side numerical stack. Python 3.9 comes from the shared agent base.
RUN micromamba install -y -n base --channel-priority strict \
      -c conda-forge \
      numpy=1.26.4 \
      libstdcxx-ng libgcc-ng \
    && micromamba clean --all --yes

RUN pip install --upgrade "pip<25" "setuptools<70" "wheel<0.45" && \
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

RUN TORCH_DIR="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')" && \
    find "${TORCH_DIR}" -type f -name '*.so' \
      -exec patchelf --clear-execstack {} \; || true

# RUN micromamba install -y -n base -c nvidia/label/cuda-12.1.1 \
#       cuda-toolkit \
#     && micromamba clean --all --yes

# README install order: GroundingDINO + LAVIS, then the VLFM package with
# simulation extras. The spacy pin keeps salesforce-lavis resolvable on py39.
# RUN pip install "spacy<3.8"

RUN pip install "spacy<3.8" && \
    pip install \
      git+https://github.com/IDEA-Research/GroundingDINO.git@eeba084341aaa454ce13cb32fa7fd9282fc73a67 \
      salesforce-lavis==1.0.2

COPY agents/baseline_vlfm/vlfm/README.md /opt/vlfm/README.md
COPY agents/baseline_vlfm/vlfm/pyproject.toml /opt/vlfm/pyproject.toml
COPY agents/baseline_vlfm/vlfm/setup.py /opt/vlfm/setup.py
COPY agents/baseline_vlfm/vlfm/config /opt/vlfm/config
COPY agents/baseline_vlfm/vlfm/scripts /opt/vlfm/scripts
COPY agents/baseline_vlfm/vlfm/vlfm /opt/vlfm/vlfm

ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
RUN pip install -e "/opt/vlfm[habitat]"
RUN pip install transformers==4.49.0

# imageio-ffmpeg is needed by agents/utils/video_recorder.py. Force the
# bundled binary fetch at build time so a missing/corrupt download fails
# the build instead of the eval.
RUN pip install "imageio>=2.31" "imageio-ffmpeg>=0.5" && \
    python -c "import imageio_ffmpeg; print('imageio-ffmpeg binary at:', imageio_ffmpeg.get_ffmpeg_exe())"

# YOLOv7 is imported from its source tree by vlfm.vlm.yolov7.
RUN git clone --depth 1 https://github.com/WongKinYiu/yolov7.git /opt/yolov7 && \
    ln -s /opt/yolov7 /opt/vlfm/yolov7

# Cache model assets that are independent of mounted baseline data.
RUN mkdir -p "${HF_HOME}" "${TORCH_HOME}/hub/checkpoints" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='bert-base-uncased', cache_dir='${HF_HOME}'); print('bert-base-uncased downloaded')"

RUN wget -q -O "${TORCH_HOME}/hub/checkpoints/eva_vit_g.pth" \
      https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth && \
    wget -q -O "${TORCH_HOME}/hub/checkpoints/blip2_pretrained.pth" \
      https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth

COPY agents/baseline_vlfm/agent_node.py /opt/rt_ovn/agent_node.py

ENV PYTHONPATH=/opt/rt_ovn:/opt/rt_ovn/utils:/opt/vlfm:/opt/GroundingDINO:/opt/yolov7:${PYTHONPATH}
WORKDIR /opt/vlfm

CMD ["/bin/bash"]
