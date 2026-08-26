#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

# Ensure you have 'export VLFM_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)

export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
export CLASSES_PATH=${CLASSES_PATH:-vlfm/vlm/classes.txt}

resolve_detector_mode() {
  local mode="${VLFM_DETECTOR_MODE:-auto}"
  mode="$(echo "${mode}" | tr '[:upper:]-' '[:lower:]_')"
  case "${mode}" in
    ""|auto|benchmark)
      local benchmark="${BENCHMARK:-${VLFM_CONFIG:-ovon}}"
      benchmark="$(echo "${benchmark}" | tr '[:upper:]' '[:lower:]')"
      if [[ "${benchmark}" == "hm3d" ]]; then
        echo "yolo_gdino"
      else
        echo "owlv2"
      fi
      ;;
    hm3d|yolo|yolo_gdino|yolo_groundingdino|groundingdino|gdino)
      echo "yolo_gdino"
      ;;
    ovon|owl|owlv2)
      echo "owlv2"
      ;;
    *)
      echo "Unsupported VLFM_DETECTOR_MODE=${VLFM_DETECTOR_MODE}" >&2
      return 2
      ;;
  esac
}

if ! VLFM_DETECTOR_MODE="$(resolve_detector_mode)"; then
  exit 2
fi
export VLFM_DETECTOR_MODE

# Find a free port by binding to port 0 and reading back the assigned port
free_port() {
  ${VLFM_PYTHON} -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-$(free_port)}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-$(free_port)}
export SAM_PORT=${SAM_PORT:-$(free_port)}
export YOLOV7_PORT=${YOLOV7_PORT:-$(free_port)}
if [[ "${VLFM_DETECTOR_MODE}" == "yolo_gdino" ]]; then
  DETECTOR_MODULE="vlfm.vlm.grounding_dino"
  DETECTOR_NAME="GroundingDINO"
else
  DETECTOR_MODULE="vlfm.vlm.owlv2"
  DETECTOR_NAME="OWLv2"
fi

session_name=vlm_servers_${RANDOM}

# Write allocated ports to an env file so the client can source them
PORT_ENV_FILE="vlm_ports_${session_name}.env"
cat > "${PORT_ENV_FILE}" <<EOF
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT}
export BLIP2ITM_PORT=${BLIP2ITM_PORT}
export SAM_PORT=${SAM_PORT}
export YOLOV7_PORT=${YOLOV7_PORT}
export VLFM_DETECTOR_MODE=${VLFM_DETECTOR_MODE}
EOF

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window vertically
tmux split-window -v -t ${session_name}:0

# Split both panes horizontally
tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.2

# Run commands in each pane
tmux send-keys -t ${session_name}:0.0 "${VLFM_PYTHON} -m ${DETECTOR_MODULE} --port ${GROUNDING_DINO_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 "${VLFM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
tmux send-keys -t ${session_name}:0.2 "${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m
if [[ "${VLFM_DETECTOR_MODE}" == "yolo_gdino" ]]; then
  tmux send-keys -t ${session_name}:0.3 "${VLFM_PYTHON} -m vlfm.vlm.yolov7 --port ${YOLOV7_PORT}" C-m
else
  tmux send-keys -t ${session_name}:0.3 "echo 'YOLOv7 disabled for VLFM_DETECTOR_MODE=${VLFM_DETECTOR_MODE}'" C-m
fi

# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Detector mode: ${VLFM_DETECTOR_MODE} (${DETECTOR_NAME})"
echo "Allocated ports: GROUNDING_DINO=${GROUNDING_DINO_PORT} BLIP2ITM=${BLIP2ITM_PORT} SAM=${SAM_PORT} YOLOV7=${YOLOV7_PORT}"
echo "Port env file written to: ${PORT_ENV_FILE}"
echo "Before running the client, source the ports:  source ${PORT_ENV_FILE}"
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"
