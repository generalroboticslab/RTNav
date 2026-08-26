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

# BeliefMapNav is a VLFM fork: it runs the same VLM servers in the agent
# container and needs the same EGL/OpenGL helpers.
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
# habitat-sim comes first, before any pip step: vlfm/policy/habitat_policies.py
# imports habitat, whose task registry imports magnum from habitat-sim, and
# running the conda solver after pip lets it downgrade numpy/python underneath
# the pip packages (readme_rtovn_authors.md step A). numpy stays at 1.26.4 —
# GroundingDINO pulls numpy 2.x, which opencv-python-headless==4.5.5.64 below
# cannot use.
RUN micromamba install -y -n base \
      -c aihabitat -c conda-forge \
      habitat-sim=0.2.5 headless \
      numpy=1.26.4 \
      libstdcxx-ng libgcc-ng \
    && micromamba clean --all --yes && \
    python -c "import habitat_sim, magnum; print('habitat_sim', habitat_sim.__version__)"

# nvcc for GroundingDINO's CUDA op. Without it the extension silently fails to
# build, and inference dies with "NameError: name '_C' is not defined" because
# ms_deform_attn.py only falls back to the pure-PyTorch path for CPU tensors.
# Kept before every pip step so the conda solver cannot move numpy/python.
RUN micromamba install -y -n base -c nvidia/label/cuda-12.1.1 \
      cuda-toolkit \
    && micromamba clean --all --yes && \
    nvcc --version | tail -2
# CUDA_HOME alone is not enough: GroundingDINO's setup.py gates the CUDA build on
#   CUDA_HOME is not None and (torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in os.environ)
# and `docker build` has no GPU, so torch.cuda.is_available() is False. Setting
# the arch list cross-compiles the op instead. 8.9 is the L40S target from
# readme_rtovn_authors.md; 8.0/8.6 cover A100/A6000 hosts.
ENV CUDA_HOME=/opt/conda \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

RUN pip install --upgrade "pip<25" "setuptools<70" "wheel<0.45" && \
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

RUN TORCH_DIR="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')" && \
    find "${TORCH_DIR}" -type f -name '*.so' \
      -exec patchelf --clear-execstack {} \; || true

# GroundingDINO's setup.py imports torch at module top level, so build isolation
# has to be off. The spacy pin keeps salesforce-lavis resolvable on py39
# (unpinned spacy pulls thinc>=8.3.12, which is 3.10+ only).
RUN pip install "spacy==3.8.3" "thinc==8.3.4" && \
    pip install --no-build-isolation \
      git+https://github.com/IDEA-Research/GroundingDINO.git@eeba084341aaa454ce13cb32fa7fd9282fc73a67 && \
    python -c "import torch; from groundingdino import _C; print('groundingdino CUDA op built:', _C.__file__)" && \
    pip install --no-deps salesforce-lavis==1.0.2

# lavis's import-chain dependencies, installed separately because the line above
# is --no-deps. decord and webdataset are required even though BeliefMapNav
# never imports them itself: lavis/datasets/data_utils.py imports decord at
# package import time, so `import lavis` fails without it.
RUN pip install \
      "transformers==4.26.1" \
      "tokenizers==0.13.3" \
      "timm==0.4.12" \
      "einops" \
      "contexttimer" \
      "sentencepiece" \
      "pycocoevalcap" \
      "omegaconf" \
      "iopath" \
      "ftfy" \
      "fairscale==0.4.4" \
      "opendatasets" \
      "plotly" \
      "pandas" \
      "open_clip_torch" \
      "decord==0.6.0" \
      "webdataset==0.1.40"

# BeliefMapNav-specific stack. The OpenFusion SLAM module in
# vlfm/mapping/openfusion/ needs open3d + cupy; ultralytics provides the SAM 2.1
# model it wraps. clip / mobile_sam / frontier_exploration / depth_camera_filtering
# are the same git pins the upstream requirements.txt uses.
RUN pip install \
      "cupy-cuda12x==13.4.0" \
      "open3d==0.19.0" \
      "alphashape==1.3.1" \
      "shapely==2.0.7" \
      "ultralytics==8.3.93" \
      "torchshow==0.5.1" \
      "supervision==0.25.1" \
      "scikit-image==0.24.0" \
      "scikit-learn==1.6.1" \
      "openai==1.66.3" \
      "hydra-core==1.3.2" \
      "seaborn" \
      "rtree" \
      "trimesh"

RUN pip install \
      "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1" \
      "git+https://github.com/ChaoningZhang/MobileSAM.git@34bbbfdface3c18e5221aa7de6032d7220c6c6a1" \
      "git+https://github.com/naokiyokoyama/frontier_exploration.git@8b4cb9329f67d150feb1b5f88c8f6ac581b95698" \
      "git+https://github.com/naokiyokoyama/depth_camera_filtering@39d6e2f391c8b2198a67ad96f94bf6da0acd48a0"

# imageio-ffmpeg is needed by agents/utils/video_recorder.py. Use the ffmpeg
# installed above instead of relying on wheel-specific bundled binaries.
ENV IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg
RUN pip install "imageio>=2.31" "imageio-ffmpeg>=0.5" && \
    python -c "import imageio_ffmpeg; print('imageio-ffmpeg binary at:', imageio_ffmpeg.get_ffmpeg_exe())"

# vlfm/policy/habitat_policies.py — the policy the ROS node loads — imports
# habitat and habitat_baselines. BeliefMapNav needs the 0.2.5 line, which is not
# on PyPI, so install from the tag like readme_rtovn_authors.md step E. The
# env_utils.py patch from that step is not applied here: it only makes the
# official runner resumable, and the ROS env container owns episode iteration.
RUN git clone --depth 1 --branch v0.2.5 \
      https://github.com/facebookresearch/habitat-lab.git /opt/habitat-lab && \
    pip install -e /opt/habitat-lab/habitat-lab && \
    pip install -e /opt/habitat-lab/habitat-baselines

# Must be last. Two things get undone by the layers above:
#   1. GroundingDINO's install pulls numpy 2.x (readme step F), and
#      opencv 4.5.5.64 is built against the numpy 1.x ABI — importing it
#      otherwise fails with "numpy.core.multiarray failed to import".
#   2. Several packages depend on opencv-python and pull 4.11.x back in. Both
#      distributions unpack into the same site-packages/cv2/, and the mix leaves
#      an orphan cv2/typing/ that fails with "module 'cv2.dnn' has no attribute
#      'DictValue'" (readme step F2). lavis pins 4.5.5.64 exactly.
# Kept in one layer so the numpy pin and the opencv install cannot drift apart.
# pillow is pinned here too: it must match the official env exactly. The CLIP
# pyramid resizes 21 crops through PIL before encoding, so a resampling change
# between releases perturbs every embedding and the belief map diverges from the
# official run on frame one (measured: emb_sum -176.72477 with 10.4.0 vs
# -176.76311 with 10.3.0).
RUN pip install "numpy==1.26.4" "pillow==10.4.0" && \
    python -c "import PIL; assert PIL.__version__ == '10.4.0', PIL.__version__; print('pillow', PIL.__version__)" && \
    { pip uninstall -y opencv-python opencv-python-headless || true; } && \
    rm -rf "$(python -c 'import site; print(site.getsitepackages()[0])')/cv2" && \
    pip install --no-deps opencv-python-headless==4.5.5.64 && \
    python -c "import cv2, numpy; print('cv2', cv2.__version__, '| numpy', numpy.__version__)"

# YOLOv7 is imported from its source tree by vlfm.vlm.yolov7.
RUN git clone --depth 1 https://github.com/WongKinYiu/yolov7.git /opt/yolov7

# vlfm/vlm/grounding_dino.py defaults to a cloned GroundingDINO/ tree that a pip
# install does not create, so it reads GROUNDING_DINO_CONFIG instead
# (rt_ovn author addition). Assert the baked path matches the installed package
# so a layout change fails the build rather than the eval.
ENV GROUNDING_DINO_CONFIG=/opt/conda/lib/python3.9/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py
RUN python -c "import os, groundingdino; \
      p = os.path.join(os.path.dirname(groundingdino.__file__), 'config', 'GroundingDINO_SwinT_OGC.py'); \
      assert p == os.environ['GROUNDING_DINO_CONFIG'], (p, os.environ['GROUNDING_DINO_CONFIG']); \
      assert os.path.isfile(p), p; print('GROUNDING_DINO_CONFIG ok:', p)"

# Cache model assets that are independent of mounted baseline data:
# bert-base-uncased (GroundingDINO's text encoder), CLIP ViT-B-32 (the OpenFusion
# feature extractor), and SAM 2.1-b (used by vlfm/mapping/openfusion/slam.py).
RUN mkdir -p "${HF_HOME}" "${TORCH_HOME}/hub/checkpoints" /opt/beliefmapnav_assets && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='bert-base-uncased', cache_dir='${HF_HOME}'); print('bert-base-uncased downloaded')" && \
    python -c "from lavis.models import load_model_and_preprocess as L; L('clip_feature_extractor', model_type='ViT-B-32', is_eval=True, device='cpu'); print('clip ViT-B-32 cached')" && \
    wget -q -O /opt/beliefmapnav_assets/sam2.1_b.pt \
      https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt && \
    chmod -R a+rwX /opt/.cache /opt/beliefmapnav_assets

ENV SAM2_CHECKPOINT=/opt/beliefmapnav_assets/sam2.1_b.pt

# BeliefMapNav is not pip-installable (setup.py is a bare setup() with no
# metadata), so the mounted source tree goes on PYTHONPATH instead. The tree is
# mounted at /opt/beliefmapnav by agents/launch_parallel.sh.
ENV PYTHONPATH=/opt/rt_ovn:/opt/rt_ovn/utils:/opt/beliefmapnav:/opt/yolov7:${PYTHONPATH}
WORKDIR /opt/beliefmapnav

CMD ["/bin/bash"]
