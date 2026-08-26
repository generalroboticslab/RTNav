# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=rt-ovn-agent-base:latest
FROM ${BASE_IMAGE}

USER root
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
ARG MAMBA_DOCKERFILE_ACTIVATE=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    L3MVN_ROOT=/opt/l3mvn \
    HF_HOME=/opt/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/.cache/huggingface \
    TORCH_HOME=/opt/.cache/torch \
    FVCORE_CACHE=/opt/.cache/fvcore \
    LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

WORKDIR /opt/l3mvn

# Agent-side system libraries for L3MVN semantic mapping and Detectron2.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget ca-certificates build-essential g++ ninja-build \
      libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1 libxml2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.5'

RUN pip install --upgrade "pip<25" "wheel<0.45" "setuptools<60.0"

# upgraded some dependencies due to pinned python==3.9
RUN pip install \
      "scikit-fmm==2023.4.2" \
      "scikit-learn==0.24"\
      "scikit-image==0.18.0" \
      "ifcfg==0.24" \
      "transformers==4.23.1" \
      "tensorboard" \
      "algo==1.3.1"
RUN pip install \
      "pillow==9.2.0 " \
      "imageio==2.31.2" \
      "imageio-ffmpeg==0.5.1" \
      "huggingface_hub==0.16.4" \
      "matplotlib==3.5.3" \
      "opencv-python==4.13.0.92" 

RUN pip install "numpy==1.21.5" 

# Cache model assets that are independent of mounted baseline data.
RUN mkdir -p "${HF_HOME}" "${TORCH_HOME}" && \
    python -c "from transformers import RobertaTokenizer, RobertaForMaskedLM; RobertaTokenizer.from_pretrained('roberta-large'); RobertaForMaskedLM.from_pretrained('roberta-large'); print('roberta-large cached')"

# Cache Detectron2's COCO Mask R-CNN
RUN mkdir -p "${FVCORE_CACHE}" && \
    python -c "from detectron2.utils.file_io import PathManager; print(PathManager.get_local_path('detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'))" && \
    chmod -R a+rwX /opt/.cache

COPY agents/baseline_l3mvn/agent_node.py /opt/rt_ovn/agent_node.py

ENV PYTHONPATH=/opt/rt_ovn:/opt/rt_ovn/utils:/opt/l3mvn:${PYTHONPATH}

CMD ["/bin/bash"]
