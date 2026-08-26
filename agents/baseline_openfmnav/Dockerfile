# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=rt-ovn-agent-base:latest
FROM ${BASE_IMAGE}

USER root
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
ARG MAMBA_DOCKERFILE_ACTIVATE=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/.cache/huggingface \
    XDG_CACHE_HOME=/opt/.cache \
    TORCH_HOME=/opt/.cache/torch \
    LD_LIBRARY_PATH=/opt/conda/lib/python3.9/site-packages/torch/lib:/opt/conda/lib:${LD_LIBRARY_PATH}

WORKDIR /opt/openfmnav

# Agent-side system libraries for Grounded-SAM and CUDA extension builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates build-essential g++ ninja-build patchelf \
      libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN micromamba install -y -n base --channel-priority strict -c aihabitat -c conda-forge \
      numpy=1.26.4 libstdcxx-ng libgcc-ng && micromamba clean --all --yes

# OpenFMNav agent ML stack. Habitat-Sim/Lab stay in the env image.
RUN pip install --upgrade "pip<25" "setuptools<70" "wheel<0.45" && \
    pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121

RUN TORCH_DIR="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')" && \
    find "${TORCH_DIR}" -type f -name '*.so' -exec patchelf --clear-execstack {} \; || true

RUN micromamba install -y -n base \
      -c nvidia/label/cuda-12.1.1 -c conda-forge \
      cuda-nvcc cuda-cudart-dev cuda-libraries-dev \
    && micromamba clean --all --yes

ENV CUDA_HOME=/opt/conda AM_I_DOCKER=1 BUILD_WITH_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

RUN git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git /opt/Grounded-Segment-Anything && \
    cd /opt/Grounded-Segment-Anything && \
    python -m pip install -e segment_anything && \
    pip install --no-build-isolation -e GroundingDINO

RUN pip install "opencv-python==4.11.0.86" "matplotlib==3.9.4" "imageio-ffmpeg==0.6.0" "numpy==1.26.4" \
    "transformers==4.45.2" "scikit-fmm==2025.1.29" "scikit-learn==1.6.1" "scikit-image==0.21.0" "ifcfg==0.24" "openai==0.28.1" "retry" 


COPY agents/baseline_openfmnav/agent_node.py /opt/rt_ovn/agent_node.py
COPY agents/utils/video_recorder.py /opt/rt_ovn/utils/video_recorder.py

ENV OPENFMNAV_ROOT=/opt/openfmnav \
    PYTHONPATH=/opt/rt_ovn:/opt/rt_ovn/utils:/opt/openfmnav:/opt/GroundingDINO:/opt/openfmnav_deps/Grounded-Segment-Anything/segment_anything:${PYTHONPATH}

RUN mkdir -p /opt/.cache/huggingface /opt/.cache/torch

CMD ["/bin/bash"]
