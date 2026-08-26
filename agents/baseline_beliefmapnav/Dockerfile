# syntax=docker/dockerfile:1.7

# BeliefMapNav is a VLFM fork.  Reuse the Thor-tested Habitat 0.2.4,
# PyTorch 2.11/cu130, GroundingDINO, LAVIS, and VLFM stack instead of trying to
# install the x86-only Habitat 0.2.5 Conda package on ARM64.
ARG BASE_IMAGE=vlfm-agent:latest

FROM ${BASE_IMAGE} AS open3d-builder

USER root
SHELL ["/bin/bash", "-c"]

ARG OPEN3D_VERSION=0.19.0
ARG CUDA_ARCHITECTURES=110

# PyPI has no ARM64 Open3D 0.19 wheel.  BeliefMapNav uses Open3D tensors on
# CUDA, so build the wheel for Thor (SM 11.0) rather than falling back to the
# CPU-only ARM64 0.18 wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake git ninja-build patch pkg-config \
      libopenblas-dev liblapacke-dev libglu1-mesa-dev libglew-dev \
      libwayland-dev wayland-protocols libxkbcommon-dev libcurl4-openssl-dev \
      xorg-dev \
    && rm -rf /var/lib/apt/lists/*

COPY agents/baseline_beliefmapnav/open3d-cuda13.patch /tmp/open3d-cuda13.patch
COPY agents/baseline_beliefmapnav/stdgpu-cuda13.patch /tmp/stdgpu-cuda13.patch

RUN git clone --depth 1 --branch "v${OPEN3D_VERSION}" \
      https://github.com/isl-org/Open3D.git /tmp/Open3D && \
    patch -d /tmp/Open3D -p1 -i /tmp/open3d-cuda13.patch && \
    cmake -S /tmp/Open3D -B /tmp/open3d-build -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
      -DBUILD_CUDA_MODULE=ON \
      -DBUILD_GUI=OFF \
      -DBUILD_WEBRTC=OFF \
      -DBUILD_JUPYTER_EXTENSION=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_UNIT_TESTS=OFF \
      -DBUILD_BENCHMARKS=OFF \
      -DBUILD_LIBREALSENSE=OFF \
      -DBUILD_PYTORCH_OPS=OFF \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUNDLE_OPEN3D_ML=OFF \
      -DBUILD_ISPC_MODULE=OFF \
      -DUSE_SYSTEM_BLAS=ON \
      -DUSE_SYSTEM_CURL=ON \
      -DUSE_SYSTEM_GLFW=OFF \
      -DUSE_SYSTEM_GLEW=ON \
      -DGLIBCXX_USE_CXX11_ABI=ON && \
    cmake --build /tmp/open3d-build --target pip-package --parallel 4

FROM ${BASE_IMAGE}

USER root
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    GROUNDING_DINO_CONFIG=/opt/conda/lib/python3.11/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    SAM2_CHECKPOINT=/opt/beliefmapnav_assets/sam2.1_b.pt

WORKDIR /opt

RUN apt-get update && apt-get install -y --no-install-recommends \
      libopenblas0 liblapacke libglew2.2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=open3d-builder /tmp/open3d-build/lib/python_package/pip_package /tmp/open3d-wheels
RUN pip uninstall -y open3d && \
    pip install /tmp/open3d-wheels/open3d-*.whl && \
    rm -r /tmp/open3d-wheels && \
    python -c "from pathlib import Path; p = next(Path('/opt/conda/lib/python3.11/site-packages/open3d').glob('_build_config.py')); assert '\"BUILD_CUDA_MODULE\" : True' in p.read_text(); print('Open3D CUDA wheel installed')"

# Only the packages BeliefMapNav adds to VLFM.  Restore VLFM's tested
# NumPy/OpenCV ABI after ultralytics has resolved its dependencies.
RUN pip install \
      "numpy==1.26.4" \
      "scipy==1.13.1" \
      "cupy-cuda13x==13.6.0" \
      "alphashape==1.3.1" \
      "shapely==2.0.7" \
      "ultralytics==8.3.93" \
      "torchshow==0.5.1" \
      "supervision==0.25.1" \
      "scikit-image==0.24.0" \
      "scikit-learn==1.6.1" \
      "openai==1.66.3" \
      "seaborn" \
      "rtree" \
      "trimesh" && \
    pip install --force-reinstall --no-deps \
      "numpy==1.26.4" "pillow==10.4.0" "opencv-python==4.5.5.64"

# BeliefMapNav was evaluated with these revisions, which differ from VLFM's.
RUN pip install --force-reinstall --no-deps \
      "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1" \
      "git+https://github.com/ChaoningZhang/MobileSAM.git@34bbbfdface3c18e5221aa7de6032d7220c6c6a1" \
      "git+https://github.com/naokiyokoyama/frontier_exploration.git@8b4cb9329f67d150feb1b5f88c8f6ac581b95698"

# Rebuild the interface because BeliefMapNav consumes the newer gps_3d fields;
# the parent VLFM image may have been built before that message was added.
COPY agents/utils/ros_msgs /opt/ros2_ws/src/ros_msgs
RUN cd /opt/ros2_ws && colcon build --packages-select ros_msgs --cmake-clean-cache

COPY agents/utils/ros_helpers.py    /opt/rt_ovn/utils/ros_helpers.py
COPY agents/utils/robot_api.py      /opt/rt_ovn/utils/robot_api.py
COPY agents/utils/video_recorder.py /opt/rt_ovn/utils/video_recorder.py
COPY agents/utils/base_agent.py     /opt/rt_ovn/utils/base_agent.py
COPY agents/utils/cyclonedds.xml    /opt/rt_ovn/utils/cyclonedds.xml
COPY agents/utils/entrypoint.sh      /opt/entrypoint.sh
COPY agents/baseline_beliefmapnav/agent_node.py /opt/rt_ovn/agent_node.py
RUN chmod 755 /opt/entrypoint.sh

# Cache the OpenFusion feature extractor and SAM model.  BERT and the VLFM
# model assets are already cached by the parent image.
RUN mkdir -p "${TORCH_HOME}/hub/checkpoints" /opt/beliefmapnav_assets && \
    python -c "from lavis.models import load_model_and_preprocess as L; L('clip_feature_extractor', model_type='ViT-B-32', is_eval=True, device='cpu'); print('CLIP ViT-B-32 cached')" && \
    wget -q -O "${SAM2_CHECKPOINT}" \
      https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt && \
    chmod -R a+rwX /opt/.cache /opt/beliefmapnav_assets

# Do not inherit /opt/GroundingDINO here: that checkout shadows the installed
# ARM64 CUDA extension with an unbuilt source package.
ENV PYTHONPATH=/opt/compat:/opt/rt_ovn:/opt/rt_ovn/utils:/opt/beliefmapnav:/opt/yolov7

RUN python -c "import os, torch, cv2, numpy; from groundingdino import _C; assert os.path.isfile(os.environ['GROUNDING_DINO_CONFIG']); print('BeliefMapNav build-time imports OK')"

WORKDIR /opt/beliefmapnav

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["/bin/bash"]
