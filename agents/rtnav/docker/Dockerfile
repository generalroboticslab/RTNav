# syntax=docker/dockerfile:1.7
# RTNav agent. CUDA 13-compatible torch/vLLM stack, including Jetson Thor.

FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------- System deps ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential python3 python3-pip python3-dev cmake git gosu \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
        libglvnd0 libglx0 libegl1 libopengl0 \
        tesseract-ocr libtesseract-dev libopencv-dev ffmpeg \
        python3-colcon-common-extensions ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && pip3 install --upgrade pip

WORKDIR /opt/rt_ovn

# ---------- ROS2 ros_msgs build ----------
COPY agents/utils/ros_msgs /opt/ros2_ws/src/ros_msgs
RUN bash -c "source /opt/ros/humble/setup.bash && \
        cd /opt/ros2_ws && \
        colcon build --packages-select ros_msgs && \
        echo 'source /opt/ros2_ws/install/setup.bash' >> /etc/bash.bashrc"

# ---------- CUDA / torch / vllm ----------
# The cu130 wheels require a host NVIDIA driver compatible with CUDA 13.
# Keep the vLLM pin synchronized with rtnav/requirements.txt.

RUN pip3 install --ignore-installed blinker && \
    pip3 install \
        torch==2.11.0+cu130 torchvision==0.26.0+cu130 \
        --index-url https://download.pytorch.org/whl/cu130 && \
    pip3 install vllm==0.23.0 \
        --extra-index-url https://download.pytorch.org/whl/cu130

# ---------- Project requirements ----------
COPY agents/rtnav/rtnav/requirements.txt /opt/rt_ovn/requirements.txt
RUN grep -v '^vllm' /opt/rt_ovn/requirements.txt > /opt/rt_ovn/requirements.no-vllm.txt && \
    pip3 install \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r /opt/rt_ovn/requirements.no-vllm.txt && \
    python -c "import imageio_ffmpeg; print('imageio-ffmpeg binary at:', imageio_ffmpeg.get_ffmpeg_exe())"

# ---------- Utils ----------
COPY agents/utils/ros_helpers.py    /opt/rt_ovn/utils/ros_helpers.py
COPY agents/utils/robot_api.py      /opt/rt_ovn/utils/robot_api.py
COPY agents/utils/video_recorder.py /opt/rt_ovn/utils/video_recorder.py
COPY agents/utils/base_agent.py     /opt/rt_ovn/utils/base_agent.py
COPY agents/utils/cyclonedds.xml    /opt/rt_ovn/utils/cyclonedds.xml

# ---------- C++ acceleration modules ----------
COPY agents/rtnav/rtnav/modules/scenegraph/cpp_accel              /opt/sg_accel_src
COPY agents/rtnav/rtnav/modules/perception/cpp_accel              /opt/perception_accel_src
RUN pip3 install --no-build-isolation \
        /opt/sg_accel_src \
        /opt/perception_accel_src && \
    rm -rf /opt/sg_accel_src /opt/perception_accel_src

# ---------- MobileSAM ----------
# Checkpoint fetched separately into $DATA_DIR (see docs/baselines.md).
RUN python -m pip install --no-cache-dir \
      "mobile_sam @ git+https://github.com/ChaoningZhang/MobileSAM.git"

# Do not let pip's cross-install resolver warning turn into a broken runtime.
# PyTorch's official AArch64 CUDA wheel labels cuSPARSELt as "manylinux2014_sbsa";
# pip does not recognize that platform tag even though the library loads on Thor.
RUN pip3 check 2>&1 | tee /tmp/pip-check.txt; \
    sed -i '/^nvidia-cusparselt-cu13 0\.8\.0 is not supported on this platform$/d' /tmp/pip-check.txt; \
    test ! -s /tmp/pip-check.txt; \
    rm /tmp/pip-check.txt

# ---------- Entrypoint ----------
COPY agents/utils/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENV PYTORCH_ALLOC_CONF=expandable_segments:True \
    TRITON_PTXAS_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/bin/ptxas \
    TRITON_PTXAS_BLACKWELL_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/bin/ptxas \
    FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    RTNAV_ROOT=/opt/rtnav \
    RT_OVN_ROOT=/opt/rt_ovn \
    PYTHONPATH=/opt/rt_ovn/agents:/opt/rt_ovn:/opt/rt_ovn/utils:/opt/rtnav

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["/bin/bash", "--login"]
