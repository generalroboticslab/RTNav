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
    TORCH_HOME=/opt/.cache/torch \
    FVCORE_CACHE=/opt/.cache/fvcore \
    LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

WORKDIR /opt/trihelper

RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget ca-certificates build-essential g++ ninja-build \
      libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1 libxml2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade "pip<25" "wheel<0.45" "setuptools<70"

RUN pip install \
      torch==1.13.0+cu116 \
      torchvision==0.14.0+cu116 \
      --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip install "git+https://github.com/facebookresearch/detectron2.git@v0.6"

RUN pip install \
      "numpy==1.23.5" \
      "scipy==1.10.1" \
      "scikit-fmm==2024.5.29" \
      "scikit-learn==0.24.2" \
      "scikit-image==0.16.2" \
      "opencv-python==4.5.5.64" \
      "ifcfg==0.24" \
      "transformers==4.32.0" \
      "huggingface_hub==0.26.5" \
      "accelerate==1.0.1" \
      "auto-gptq==0.5.1" \
      "peft==0.13.2" \
      "optimum==1.23.3" \
      "tiktoken==0.7.0" \
      "einops==0.8.1" \
      "transformers_stream_generator==0.0.5" \
      "pillow==9.4.0" \
      "matplotlib==3.7.5" \
      "tensorboard" \
      "algo==1.3.1"

# Cache the two non-Qwen model assets. Qwen-VL-Chat-Int4 is mounted from the
# host Hugging Face cache at runtime because its quantized shards are large.
RUN mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${FVCORE_CACHE}" && \
    python -c "from transformers import RobertaTokenizer, RobertaForMaskedLM; RobertaTokenizer.from_pretrained('roberta-large'); RobertaForMaskedLM.from_pretrained('roberta-large'); print('roberta-large cached')" && \
    python -c "from detectron2.utils.file_io import PathManager; print(PathManager.get_local_path('detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'))" && \
    chmod -R a+rwX /opt/.cache

COPY agents/baseline_trihelper/TriHelper/constants.py /opt/trihelper/constants.py
COPY agents/baseline_trihelper/TriHelper/model.py /opt/trihelper/model.py
COPY agents/baseline_trihelper/TriHelper/SimSun.ttf /opt/trihelper/SimSun.ttf
COPY agents/baseline_trihelper/TriHelper/RedNet/RedNet_model.py /opt/trihelper/RedNet/RedNet_model.py
COPY agents/baseline_trihelper/TriHelper/RedNet/utils.py /opt/trihelper/RedNet/utils.py
COPY agents/baseline_trihelper/TriHelper/agents/utils /opt/trihelper/agents/utils
COPY agents/baseline_trihelper/TriHelper/configs /opt/trihelper/configs
COPY agents/baseline_trihelper/TriHelper/envs/utils /opt/trihelper/envs/utils
COPY agents/baseline_trihelper/TriHelper/utils /opt/trihelper/utils
COPY agents/baseline_trihelper/TriHelper/pretrained_models/Qwen-VL /opt/trihelper/pretrained_models/Qwen-VL
COPY agents/baseline_trihelper/agent_node.py /opt/rt_ovn/agent_node.py

ENV PYTHONPATH=/opt/rt_ovn:/opt/rt_ovn/utils:/opt/trihelper:${PYTHONPATH}

CMD ["/bin/bash"]
