#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_parallel.sh — Parallel ObjectNav evaluation across N GPUs via Docker
#
# Generates docker-compose.parallel.yml with N (env, agent) container pairs.
# Each pair gets its own GPU, ROS2 domain, episode slice, and log directory.
#
# Supported baselines: vlfm, l3mvn, trihelper, openfmnav, rtnav, gamap, beliefmapnav.
#
# Usage:
#   bash launch_parallel.sh --baseline vlfm --episodes 3000 --workers 8 --gpus 0,1,2,3,4,5,6,7
#   bash launch_parallel.sh --baseline vlfm --benchmark hm3d_v1 --episodes 2000 --workers 8 --gpus 0,1,2,3,4,5,6,7
#   bash launch_parallel.sh --baseline l3mvn --benchmark hm3d_v2 --episodes 1000 --workers 8 --gpus 0,1,2,3,4,5,6,7
#   bash launch_parallel.sh --baseline trihelper --benchmark hm3d_v1 --episodes 2000 --workers 8 --gpus 0,1,2,3,4,5,6,7
#   OPENAI_API_KEY=... bash launch_parallel.sh --baseline openfmnav --benchmark hm3d_v2 --episodes 1000 --workers 8 --gpus 0,1,2,3,4,5,6,7
#   OPENAI_API_KEY=... bash launch_parallel.sh --baseline openfmnav --benchmark ovon --episodes 3000 --workers 8 --gpus 0,1,2,3,4,5,6,7
#   bash launch_parallel.sh --baseline rtnav --benchmark ovon --episodes 100 --workers 4 --gpus 0,1,2,3
#   bash launch_parallel.sh --baseline rtnav --llm gemma4 --benchmark hm3d_v2 --episodes 100 --workers 4 --gpus 0,1,2,3
#   bash launch_parallel.sh --baseline rtnav --llm qwen3.5-4b --benchmark hm3d_v2 --episodes 100 --workers 4 --gpus 0,1,2,3
#
# Environment variables can also be used:
#   BASELINE=vlfm TOTAL_EPISODES=3000 NUM_WORKERS=8 GPU_LIST=0,1,2,3,4,5,6,7 bash launch_parallel.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────────────
BASELINE="${BASELINE:-vlfm}"
TOTAL_EPISODES="${TOTAL_EPISODES:-3000}"
START_EPISODE="${START_EPISODE:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
BENCHMARK="${BENCHMARK:-ovon}"
VLFM_DETECTOR_MODE="${VLFM_DETECTOR_MODE:-auto}"
MODE_SOURCE="${MODE:+env}"
MODE="${MODE:-sync}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
TRACE="${TRACE:-0}"
TRACE_DIR="${TRACE_DIR:-}"
VIDEO_MODE="${VIDEO:-${video:-none}}"
LAUNCH_NOW="${LAUNCH_NOW:-prompt}"
RTNAV_LLM="${RTNAV_LLM:-qwen3.5}"
OPENFMNAV_DISABLE_LEGEND=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline)   BASELINE="$2"; shift 2;;
        --episodes)   TOTAL_EPISODES="$2"; shift 2;;
        --start-episode) START_EPISODE="$2"; shift 2;;
        --workers)    NUM_WORKERS="$2"; shift 2;;
        --gpus)       GPU_LIST="$2"; shift 2;;
        --benchmark)  BENCHMARK="$2"; shift 2;;
        --llm)        RTNAV_LLM="$2"; shift 2;;
        --vlfm-detector-mode) VLFM_DETECTOR_MODE="$2"; shift 2;;
        --mode)       MODE="$2"; MODE_SOURCE="arg"; shift 2;;
        --trace)      TRACE=1; shift;;
        --trace-dir)  TRACE=1; TRACE_DIR="$2"; shift 2;;
        --video)
            if [[ $# -lt 2 || "${2:-}" == --* ]]; then
                echo "ERROR: --video requires one of: none, display, save" >&2
                exit 1
            fi
            VIDEO_MODE="$2"; shift 2;;
        --video=*)    VIDEO_MODE="${1#*=}"; shift;;
        --launch)     LAUNCH_NOW="y"; shift;;
        --no-launch)  LAUNCH_NOW="n"; shift;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

_resolve_vlfm_detector_mode() {
  local mode="$1"
  mode="$(echo "${mode}" | tr '[:upper:]-' '[:lower:]_')"
  case "$mode" in
    ""|auto|benchmark)
      if [[ "$BENCHMARK" == hm3d_v* ]]; then
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
      echo "ERROR: VLFM_DETECTOR_MODE must be auto, owlv2, or yolo_gdino (got ${mode})" >&2
      return 2
      ;;
  esac
}

if [[ "$BASELINE" == "vlfm" ]]; then
  VLFM_DETECTOR_MODE="$(_resolve_vlfm_detector_mode "$VLFM_DETECTOR_MODE")"
fi

VIDEO_MODE="$(echo "${VIDEO_MODE}" | tr '[:upper:]' '[:lower:]')"
case "$VIDEO_MODE" in
  none|display|save) ;;
  *)
    echo "ERROR: --video must be one of: none, display, save (got ${VIDEO_MODE})" >&2
    exit 1
    ;;
esac

case "$BASELINE" in
  vlfm|l3mvn|trihelper|openfmnav|rtnav|gamap|beliefmapnav) ;;
  *)
    echo "ERROR: Unsupported baseline '$BASELINE'. Supported: vlfm, l3mvn, trihelper, openfmnav, rtnav, gamap, beliefmapnav." >&2
    exit 1
    ;;
esac

if [[ "$BASELINE" == "trihelper" && "$BENCHMARK" == "ovon" ]]; then
  echo "ERROR: trihelper supports the six-category HM3D ObjectNav benchmark only." >&2
  exit 1
fi

if [[ "$BASELINE" == "beliefmapnav" && "$BENCHMARK" == "ovon" ]]; then
  echo "ERROR: beliefmapnav currently supports HM3D only." >&2
  exit 1
fi

if [[ "$BASELINE" == "rtnav" && -z "$MODE_SOURCE" ]]; then
  MODE="async"
fi

if [[ "$BASELINE" == "rtnav" && "$MODE" != "async" ]]; then
  echo "ERROR: rtnav currently supports async mode only." >&2
  exit 1
fi

case "$BENCHMARK" in
  hm3d_v1|hm3d_v2|ovon) ;;
  hm3d)
    echo "ERROR: benchmark 'hm3d' is ambiguous; use hm3d_v1 or hm3d_v2." >&2
    exit 1
    ;;
  *)
    echo "ERROR: Unsupported benchmark '$BENCHMARK'. Supported: hm3d_v1, hm3d_v2, ovon." >&2
    exit 1
    ;;
esac

RTNAV_LLM="$(echo "${RTNAV_LLM}" | tr '[:upper:]' '[:lower:]')"
case "$RTNAV_LLM" in
  qwen|qwen3.5) RTNAV_LLM="qwen3.5" ;;
  qwen3.5-4b) RTNAV_LLM="qwen3.5-4b" ;;
  gemma|gemma4) RTNAV_LLM="gemma4" ;;
  *)
    echo "ERROR: --llm must be one of: qwen3.5, qwen3.5-4b, gemma4 (got ${RTNAV_LLM})" >&2
    exit 1
    ;;
esac

AGENT_BENCHMARK="$BENCHMARK"
if [[ "$BASELINE" != "rtnav" && "$BENCHMARK" == hm3d_v* ]]; then
  AGENT_BENCHMARK="hm3d"
fi

IFS=',' read -ra GPUS <<< "$GPU_LIST"
if (( ${#GPUS[@]} < NUM_WORKERS )); then
    echo "ERROR: GPU_LIST has ${#GPUS[@]} GPUs but NUM_WORKERS=$NUM_WORKERS" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENTS_DIR="$SCRIPT_DIR"
RT_OVN_DIR="$(dirname "$AGENTS_DIR")"
DATA_DIR="${DATA_DIR:-}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the dataset root from docs/datasets.md}"
HOST_QWEN_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}"
if [[ "$BASELINE" == "rtnav" ]]; then
  BASELINE_DIR="${AGENTS_DIR}/rtnav"
else
  BASELINE_DIR="${AGENTS_DIR}/baseline_${BASELINE}"
fi
_require_model_file() {
  if [[ ! -s "${DATA_DIR}/$1" ]]; then
    echo "ERROR: Missing model: ${DATA_DIR}/$1" >&2
    echo "Download shared checkpoints as described in docs/baselines.md." >&2
    exit 1
  fi
}

case "$BASELINE" in
  vlfm)
    _require_model_file pointnav_weights.pth
    ;;
  l3mvn|gamap)
    _require_model_file rednet_semmap_mp3d_40.pth
    ;;
  trihelper)
    _require_model_file rednet_semmap_mp3d_40.pth
    if [[ ! -d "${HOST_QWEN_CACHE}/models--Qwen--Qwen-VL-Chat-Int4" ]]; then
      echo "ERROR: Qwen/Qwen-VL-Chat-Int4 is not present in ${HOST_QWEN_CACHE}." >&2
      exit 1
    fi
    ;;
  beliefmapnav)
    _require_model_file pointnav_weights.pth
    _require_model_file rednet_semmap_mp3d_40.pth
    ;;
  rtnav)
    _require_model_file pointnav_weights.pth
    ;;
esac
if [[ -n "$TRACE_DIR" ]]; then
  LOG_DIR="$TRACE_DIR"
else
  LOG_DIR="${LOG_DIR:-${BASELINE_DIR}/logs_parallel}"
fi
COMPOSE_FILE="${BASELINE_DIR}/docker-compose.parallel.yml"
MERGE_SCRIPT="${AGENTS_DIR}/utils/merge_stats.py"

ENV_TRACE_DEFAULT=""
AGENT_TRACE_DEFAULT=""
if [[ "$TRACE" == "1" ]]; then
  ENV_TRACE_DEFAULT="/opt/worker_logs/env_trace.json"
  AGENT_TRACE_DEFAULT="/opt/worker_logs/agent_trace.json"
fi
AGENT_VIDEO_DIR_DEFAULT=""
if [[ "$VIDEO_MODE" == "save" ]]; then
  AGENT_VIDEO_DIR_DEFAULT="/opt/worker_logs/videos"
fi
DISABLE_TOP_DOWN_MAP_DEFAULT="1"

BASE_COUNT=$(( TOTAL_EPISODES / NUM_WORKERS ))
REMAINDER=$(( TOTAL_EPISODES % NUM_WORKERS ))
BASE_DOMAIN_ID="${BASE_DOMAIN_ID:-50}"
BASE_PORT=12200  # VLM server ports (VLFM only)

echo "═══════════════════════════════════════════════════════════════════"
echo " Parallel ${BASELINE^^} ${BENCHMARK^^} Evaluation (Docker)"
echo "   Baseline       : $BASELINE"
echo "   Benchmark      : $BENCHMARK"
echo "   Mode           : $MODE"
if [[ "$BASELINE" == "vlfm" ]]; then
  echo "   VLFM detector  : $VLFM_DETECTOR_MODE"
fi
if [[ "$BASELINE" == "rtnav" ]]; then
  echo "   LLM            : $RTNAV_LLM"
fi
echo "   Total episodes : $TOTAL_EPISODES"
echo "   Workers        : $NUM_WORKERS"
echo "   GPUs           : $GPU_LIST"
echo "   Data dir       : $DATA_DIR"
echo "   Log root       : $LOG_DIR"
echo "   Trace          : $TRACE"
echo "   Video          : $VIDEO_MODE"
echo "   Env video      : 0"
echo "   Agent video    : $VIDEO_MODE"
echo "   Disable Legend : $OPENFMNAV_DISABLE_LEGEND"
echo "═══════════════════════════════════════════════════════════════════"

mkdir -p "$LOG_DIR"
if [[ "$BASELINE" == "l3mvn" ]]; then
  mkdir -p "${BASELINE_DIR}/L3MVN/data"
fi
if [[ "$BASELINE" == "gamap" ]]; then
  mkdir -p "${BASELINE_DIR}/GAMap/data"
fi
[[ "$BASELINE" != "trihelper" ]] || mkdir -p "${BASELINE_DIR}/TriHelper/data"
if [[ "$BASELINE" == "beliefmapnav" ]]; then
  for weight in mobile_sam.pt groundingdino_swint_ogc.pth yolov7-e6e.pt; do
    if [[ ! -s "${DATA_DIR}/${weight}" ]]; then
      echo "ERROR: beliefmapnav weight is missing: ${DATA_DIR}/${weight}" >&2
      exit 1
    fi
  done
  if [[ "${BMN_DUMMY_LLM:-0}" != "1" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: beliefmapnav runs GPT-4o by default; set OPENAI_API_KEY or BMN_DUMMY_LLM=1." >&2
    exit 1
  fi
fi

# ── Image names ──────────────────────────────────────────────────────────────
case "$BASELINE" in
  vlfm) ENV_IMAGE="rt-ovn-env:latest"; AGENT_IMAGE="vlfm-agent:latest";;
  l3mvn) ENV_IMAGE="rt-ovn-env:latest"; AGENT_IMAGE="l3mvn-agent:latest";;
  trihelper) ENV_IMAGE="rt-ovn-env:latest"; AGENT_IMAGE="trihelper-agent:latest";;
  openfmnav) ENV_IMAGE="rt-ovn-env:latest"; AGENT_IMAGE="openfmnav-agent:latest";;
  rtnav) ENV_IMAGE="rt-ovn-env:latest"; AGENT_IMAGE="rt-ovn-agent:latest";;
  gamap) ENV_IMAGE="rt-ovn-env:latest"; AGENT_IMAGE="gamap-agent:latest";;
  beliefmapnav) ENV_IMAGE="rt-ovn-env:latest"; AGENT_IMAGE="beliefmapnav-agent:latest";;
esac
[[ "$BASELINE" != "rtnav" || -s "${DATA_DIR}/pointnav_weights.pth.flat.pth" ]] || docker run --rm --user "${HOST_UID}:${HOST_GID}" -v "${DATA_DIR}:/data" -v "${AGENTS_DIR}/baseline_vlfm/vlfm:/opt/vlfm:ro" "$ENV_IMAGE" python -c 'import os, torch; p="/data/pointnav_weights.pth"; o=p+".flat.pth"; t=o+".tmp"; torch.save(torch.load(p, map_location="cpu")["state_dict"], t); os.replace(t, o)'

# ── Baseline-specific mount paths ───────────────────────────────────────────
_env_working_dir() {
    case "$BASELINE" in
    vlfm) echo "/opt/vlfm";;
    l3mvn) echo "/opt/l3mvn";;
    trihelper) echo "/opt/trihelper";;
    openfmnav) echo "/opt/openfmnav";;
    rtnav) echo "/opt/rt_ovn";;
    gamap) echo "/opt/gamap";;
    beliefmapnav) echo "/opt/rt_ovn";;
    esac
}

_env_code_volumes() {
    case "$BASELINE" in
    vlfm)
      echo "      - ${BASELINE_DIR}/vlfm:/opt/vlfm"
      echo "      - ${RT_OVN_DIR}/envs/ovon:/opt/ovon"
      ;;
    l3mvn)
      echo "      - ${BASELINE_DIR}/L3MVN:/opt/l3mvn:ro"
      ;;
    trihelper)
      echo "      - ${BASELINE_DIR}/TriHelper:/opt/trihelper:ro"
      ;;
    gamap)
      echo "      - ${BASELINE_DIR}/GAMap:/opt/gamap:ro"
      ;;
    openfmnav)
      echo "      - ${BASELINE_DIR}/OpenFMNav:/opt/openfmnav"
      ;;
    rtnav)
      echo "      - ${RT_OVN_DIR}:/opt/rt_ovn"
      echo "      - ${RT_OVN_DIR}/envs/ovon:/opt/ovon"
      echo "      - ${DATA_DIR}:${DATA_DIR}"
      echo "      - ${DATA_DIR}:/opt/vlfm/data:ro"
      ;;
    esac
}

_env_data_volume() {
    echo "${DATA_DIR}:/opt/rt_ovn/data:ro"
}

_env_shim_volumes() {
    case "$BASELINE" in
    rtnav)
      ;;
    *)
      echo "      - ${RT_OVN_DIR}/envs/env_node.py:/opt/rt_ovn/env_node.py:ro"
      echo "      - ${RT_OVN_DIR}/envs/habitat_env.py:/opt/rt_ovn/habitat_env.py:ro"
      echo "      - ${RT_OVN_DIR}/envs/async_ui.py:/opt/rt_ovn/async_ui.py:ro"
      ;;
    esac
}

_env_node_script() {
    case "$BASELINE" in
    rtnav) echo "/opt/rt_ovn/envs/env_node.py";;
    *) echo "/opt/rt_ovn/env_node.py";;
    esac
}

_env_cd_cmd() {
    case "$BASELINE" in
    vlfm) echo "cd /opt/vlfm/vlfm";;
    l3mvn) echo "cd /opt/l3mvn";;
    trihelper) echo "cd /opt/trihelper";;
    openfmnav) echo "cd /opt/openfmnav";;
    rtnav) echo "cd /opt/rt_ovn";;
    gamap) echo "cd /opt/gamap";;
    beliefmapnav) echo "cd /opt/rt_ovn";;
    esac
}

_host_user_path() {
    case "$BASELINE" in
    vlfm) echo "/opt/vlfm";;
    l3mvn) echo "/opt/l3mvn";;
    trihelper) echo "/opt/trihelper";;
    openfmnav) echo "/opt/openfmnav";;
    rtnav) echo "/opt/rt_ovn";;
    gamap) echo "/opt/gamap";;
    beliefmapnav) echo "/opt/beliefmapnav";;
    esac
}

_host_chown_paths() {
    case "$BASELINE" in
    vlfm) echo "/opt/worker_logs:/opt/vlfm/lockfiles:/opt/.cache";;
    rtnav) echo "/opt/worker_logs:/opt/.cache:/tmp/.cache";;
    *) echo "/opt/worker_logs:/opt/.cache";;
    esac
}

_env_hydra_config_path() {
    case "$BASELINE" in
    rtnav) echo "/opt/rt_ovn/agents/rtnav/config";;
    *) echo "/opt/rt_ovn/env_config";;
    esac
}

_env_hydra_config_name() {
    case "$BASELINE" in
    rtnav) echo "\${HYDRA_CONFIG_NAME:-experiments/rt_ovn_objectnav_${BENCHMARK}}";;
    *) echo "\${HYDRA_CONFIG_NAME:-${BENCHMARK}}";;
    esac
}


_agent_extra_volumes() {
  case "$BASELINE" in
    vlfm)
      echo "      - ${BASELINE_DIR}/vlfm:/opt/vlfm"
      echo "      - ${DATA_DIR}:/opt/vlfm/data"
      ;;
    l3mvn)
      echo "      - ${BASELINE_DIR}/L3MVN:/opt/l3mvn:ro"
      echo "      - ${DATA_DIR}:/opt/l3mvn/data:ro"
      ;;
    trihelper)
      echo "      - ${BASELINE_DIR}/TriHelper:/opt/trihelper:ro"
      echo "      - ${DATA_DIR}:/opt/trihelper/data:ro"
      echo "      - ${HOST_QWEN_CACHE}:/.cache/huggingface/hub"
      ;;
    gamap)
      echo "      - ${BASELINE_DIR}/GAMap:/opt/gamap:ro"
      echo "      - ${DATA_DIR}:/opt/gamap/data:ro"
      ;;
    beliefmapnav)
      echo "      - ${BASELINE_DIR}/BeliefMapNav:/opt/beliefmapnav:ro"
      echo "      - ${DATA_DIR}:/opt/beliefmapnav/data:ro"
      echo "      - ${WORKER_LOG}/beliefmapnav_outputs:/opt/beliefmapnav_outputs"
      ;;
    openfmnav)
      echo "      - ${BASELINE_DIR}/OpenFMNav:/opt/openfmnav"
      echo "      - ${DATA_DIR}:/opt/openfmnav/data"
      echo "      - ${DATA_DIR}:/opt/vlfm/data"
      ;;
    rtnav)
      echo "      - ${RT_OVN_DIR}:/opt/rt_ovn"
      echo "      - ${BASELINE_DIR}:/opt/rtnav"
      echo "      - ${DATA_DIR}:${DATA_DIR}"
      echo "      - ${DATA_DIR}:/opt/rt_ovn/data"
      echo "      - ${DATA_DIR}:/opt/vlfm/data:ro"
      echo "      - ${WORKER_LOG}/cache/vllm:/home/roboticslab/.cache/vllm"
      ;;
  esac
}

_agent_command() {
    case "$BASELINE" in
    vlfm)
      cat <<'AGENTCMD'
        set -e
        cleanup_vlm_servers() {
          PIDS=$$(jobs -p)
          kill $$PIDS 2>/dev/null || true
          wait $$PIDS 2>/dev/null || true
        }
        wait_for_port() {
          NAME=$$1
          PORT=$$2
          TRIES=0
          until curl -s -o /dev/null http://localhost:$$PORT/ 2>/dev/null; do
            if [ $$TRIES -ge 600 ]; then
              echo ERROR: $$NAME did not become ready on port $$PORT >&2
              return 1
            fi
            sleep 1
            TRIES=$$(( TRIES + 1 ))
          done
        }
        trap cleanup_vlm_servers EXIT
        source /opt/ros2_ws/install/setup.bash 2>/dev/null || true
        cd /opt/vlfm
        DETECTOR_MODE=$${VLFM_DETECTOR_MODE:-owlv2}
        export VLFM_DETECTOR_MODE=$$DETECTOR_MODE
        if [ $$DETECTOR_MODE = yolo_gdino ]; then
          python -m vlfm.vlm.grounding_dino --port $${GROUNDING_DINO_PORT} &
        else
          python -m vlfm.vlm.owlv2 --port $${GROUNDING_DINO_PORT} &
        fi
        python -m vlfm.vlm.blip2itm      --port $${BLIP2ITM_PORT}       &
        python -m vlfm.vlm.sam            --port $${SAM_PORT}            &
        if [ $$DETECTOR_MODE = yolo_gdino ]; then
          mkdir -p /tmp/rt_ovn/vlfm_yolov7
          ln -sfn /opt/vlfm/data /tmp/rt_ovn/vlfm_yolov7/data
          ln -sfn /opt/vlfm/yolov7 /tmp/rt_ovn/vlfm_yolov7/yolov7
          (cd /tmp/rt_ovn/vlfm_yolov7 && python -m vlfm.vlm.yolov7 --port $${YOLOV7_PORT}) &
        fi
        echo 'Waiting for VLM servers …'
        wait_for_port detector $${GROUNDING_DINO_PORT}
        wait_for_port blip2itm $${BLIP2ITM_PORT}
        wait_for_port sam $${SAM_PORT}
        if [ $$DETECTOR_MODE = yolo_gdino ]; then
          wait_for_port yolov7 $${YOLOV7_PORT}
        fi
        python -m vlfm.utils.generate_dummy_policy 2>/dev/null || true
        python /opt/rt_ovn/agent_node.py
AGENTCMD
      ;;
    beliefmapnav)
      cat <<'AGENTCMD'
        set -e
        cleanup_vlm_servers() {
          PIDS=$$(jobs -p)
          kill $$PIDS 2>/dev/null || true
          wait $$PIDS 2>/dev/null || true
        }
        wait_for_port() {
          NAME=$$1
          PORT=$$2
          TRIES=0
          until curl -s -o /dev/null http://localhost:$$PORT/ 2>/dev/null; do
            if [ $$TRIES -ge 600 ]; then
              echo ERROR: $$NAME did not become ready on port $$PORT >&2
              return 1
            fi
            sleep 1
            TRIES=$$(( TRIES + 1 ))
          done
        }
        trap cleanup_vlm_servers EXIT
        source /opt/ros2_ws/install/setup.bash 2>/dev/null || true
        cd /opt/beliefmapnav
        python -m vlfm.vlm.grounding_dino --port $${GROUNDING_DINO_PORT} &
        python -m vlfm.vlm.sam            --port $${SAM_PORT}            &
        mkdir -p /tmp/rt_ovn/bmn_yolov7
        ln -sfn /opt/beliefmapnav/data /tmp/rt_ovn/bmn_yolov7/data
        ln -sfn /opt/yolov7 /tmp/rt_ovn/bmn_yolov7/yolov7
        (cd /tmp/rt_ovn/bmn_yolov7 && python -m vlfm.vlm.yolov7 --port $${YOLOV7_PORT}) &
        echo 'Waiting for VLM servers …'
        wait_for_port grounding_dino $${GROUNDING_DINO_PORT}
        wait_for_port sam $${SAM_PORT}
        wait_for_port yolov7 $${YOLOV7_PORT}
        python /opt/rt_ovn/agent_node.py
AGENTCMD
      ;;
    l3mvn)
      cat <<'AGENTCMD'
        source /opt/ros2_ws/install/setup.bash 2>/dev/null || true &&
        cd /opt/l3mvn &&
        mkdir -p /opt/.cache /opt/.cache/huggingface /opt/.cache/torch /opt/.cache/fvcore &&
        python /opt/rt_ovn/agent_node.py
AGENTCMD
      ;;
    trihelper)
      cat <<'AGENTCMD'
        source /opt/ros2_ws/install/setup.bash 2>/dev/null || true &&
        cd /opt/trihelper &&
        mkdir -p /opt/.cache /opt/.cache/huggingface /opt/.cache/torch /opt/.cache/fvcore /opt/worker_logs/trihelper_tmp &&
        python /opt/rt_ovn/agent_node.py
AGENTCMD
      ;;
    gamap)
      cat <<'AGENTCMD'
        source /opt/ros2_ws/install/setup.bash 2>/dev/null || true &&
        cd /opt/gamap &&
        mkdir -p /opt/.cache /opt/.cache/huggingface /opt/.cache/torch /opt/.cache/fvcore &&
        python /opt/rt_ovn/agent_node.py
AGENTCMD
      ;;
    openfmnav)
      cat <<'AGENTCMD'
        source /opt/ros2_ws/install/setup.bash 2>/dev/null || true &&
        cd /opt/openfmnav &&
        mkdir -p /opt/.cache /opt/.cache/huggingface /opt/.cache/torch &&
        python /opt/rt_ovn/agent_node.py
AGENTCMD
      ;;
    rtnav)
      cat <<'AGENTCMD'
        source /opt/ros2_ws/install/setup.bash 2>/dev/null || true &&
        cd /opt/rt_ovn &&
        _pyver=$$(python3 -c 'import sys; print(f\"{sys.version_info.major}{sys.version_info.minor}\")') &&
        mkdir -p /opt/rt_ovn/.locks &&
        (
          flock 9;
          for d in agents/rtnav/rtnav/modules/perception/cpp_accel agents/rtnav/rtnav/modules/scenegraph/cpp_accel; do
            _so=$$(ls $$d/*.cpython-$${_pyver}-*.so 2>/dev/null | head -1);
            _stale=0;
            if [ -n \"$$_so\" ]; then
              for src in $$d/*.cpp $$d/*.h $$d/*.hpp $$d/build.sh; do
                [ -f \"$$src\" ] && [ \"$$src\" -nt \"$$_so\" ] && _stale=1;
              done;
            fi;
            if [ -z \"$$_so\" ] || [ \"$$_stale\" = 1 ] || ldd \"$$_so\" 2>&1 | grep -q 'not found'; then
              echo \"[cpp_accel] (re)building $$d (so=$$_so stale=$$_stale)\";
              if ! bash $$d/build.sh; then
                echo \"[cpp_accel] FATAL: build failed in $$d (so=$$_so stale=$$_stale pyver=$$_pyver)\" >&2;
                exit 1;
              fi;
            fi;
          done
        ) 9>/opt/rt_ovn/.locks/rtnav_cpp_accel.lock &&
        exec python /opt/rtnav/agent_node.py $${RTNAV_ARGS:-}
AGENTCMD
      ;;
    esac
}

# ── Generate docker-compose.parallel.yml ─────────────────────────────────────

cat > "$COMPOSE_FILE" <<HEADER
# AUTO-GENERATED by launch_parallel.sh — do not edit manually.
# Baseline: $BASELINE | Benchmark: $BENCHMARK | Workers: $NUM_WORKERS

x-common: &common
  shm_size: "64g"
  network_mode: host
  environment: &common-env
    NVIDIA_VISIBLE_DEVICES: all
    NVIDIA_DRIVER_CAPABILITIES: all
    HOST_UID: ${HOST_UID}
    HOST_GID: ${HOST_GID}
    HOST_USER_PATH: $(_host_user_path)
    HOST_CHOWN_PATHS: $(_host_chown_paths)
    HOME: /tmp
    MPLCONFIGDIR: /tmp/matplotlib
    RMW_IMPLEMENTATION: rmw_cyclonedds_cpp
    CYCLONEDDS_URI: file:///opt/rt_ovn/utils/cyclonedds.xml
    MODE: ${MODE}
    BENCHMARK: ${BENCHMARK}
    BASELINE: ${BASELINE}
    VLFM_MODE: ${MODE}
    VLFM_CONFIG: ${BENCHMARK}
    VLFM_DETECTOR_MODE: ${VLFM_DETECTOR_MODE}
    OMP_NUM_THREADS: "${OMP_NUM_THREADS:-1}"
    MKL_NUM_THREADS: "${MKL_NUM_THREADS:-1}"
    OPENBLAS_NUM_THREADS: "${OPENBLAS_NUM_THREADS:-1}"
    NUMEXPR_NUM_THREADS: "${NUMEXPR_NUM_THREADS:-1}"
    KMP_DUPLICATE_LIB_OK: "${KMP_DUPLICATE_LIB_OK:-TRUE}"

services:
HEADER

OFFSET="$START_EPISODE"
for (( i=0; i<NUM_WORKERS; i++ )); do
    GPU="${GPUS[$i]}"
    DOMAIN_ID=$(( BASE_DOMAIN_ID + i ))
    EP_COUNT=$(( BASE_COUNT + (i < REMAINDER ? 1 : 0) ))
    WORKER_LOG="${LOG_DIR}/subset_${i}"
    mkdir -p "${WORKER_LOG}"
    if [[ "$BASELINE" == "rtnav" ]]; then
      mkdir -p "${WORKER_LOG}/cache/vllm" "${WORKER_LOG}/cache/xdg" \
               "${WORKER_LOG}/cache/torch" "${WORKER_LOG}/cache/torchinductor" \
               "${WORKER_LOG}/cache/flashinfer"
    fi
    if [[ "$BASELINE" == "beliefmapnav" ]]; then
      mkdir -p "${WORKER_LOG}/beliefmapnav_outputs"
    fi
    if [[ "$VIDEO_MODE" == "save" ]]; then
      mkdir -p "${WORKER_LOG}/videos"
    fi

    echo "── Subset $i: GPU=$GPU  DOMAIN=$DOMAIN_ID  eps=[${OFFSET}..$(( OFFSET + EP_COUNT - 1 ))]"

    # ── env container ────────────────────────────────────────────────────
    cat >> "$COMPOSE_FILE" <<EOF

  # ── Subset $i  (GPU $GPU, domain $DOMAIN_ID, episodes $OFFSET..$(( OFFSET + EP_COUNT - 1 ))) ──
  env_${i}:
    <<: *common
    image: ${ENV_IMAGE}
    container_name: ${BASELINE}_env_${i}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${GPU}"]
              capabilities: [gpu]
    environment:
      <<: *common-env
      OPENCV_FOR_THREADS_NUM: "${OPENCV_FOR_THREADS_NUM:-1}"
      NVIDIA_VISIBLE_DEVICES: "${GPU}"   # override common-env "all": isolate this worker to its GPU
      CUDA_VISIBLE_DEVICES: "0"          # the single visible GPU is index 0 inside the container
      ROS_DOMAIN_ID: "${DOMAIN_ID}"
      EPISODE_OFFSET: "${OFFSET}"
      EPISODE_COUNT: "${EP_COUNT}"
      ENV_STATS_PATH: /opt/worker_logs/stats_${i}.json
      ENV_TRACE_PATH: ${ENV_TRACE_PATH:-${ENV_TRACE_DEFAULT}}
      VIDEO_DIR: ""
      HYDRA_CONFIG_PATH: $(_env_hydra_config_path)
      HYDRA_CONFIG_NAME: "$(_env_hydra_config_name)"
      HYDRA_RUN_DIR: /opt/worker_logs/hydra
      EVAL_SPLIT: "${EVAL_SPLIT:-}"
      DISABLE_TOP_DOWN_MAP: "${DISABLE_TOP_DOWN_MAP_DEFAULT}"
    volumes:
$(_env_code_volumes $i)
      - $(_env_data_volume)
      - ${WORKER_LOG}:/opt/worker_logs
      - ${AGENTS_DIR}/utils:/opt/rt_ovn/utils:ro
$(_env_shim_volumes)
      - ${RT_OVN_DIR}/envs/config:/opt/rt_ovn/env_config:ro
    working_dir: $(_env_working_dir)
    command: >
      bash -c "
        source /opt/ros2_ws/install/setup.bash 2>/dev/null || true &&
        $(_env_cd_cmd) &&
        python $(_env_node_script)
      "
EOF

    # ── agent container ──────────────────────────────────────────────────
    AGENT_ENV_EXTRA=""
    case "$BASELINE" in
      vlfm)
        GD_PORT=$(( BASE_PORT + i * 10 + 0 ))
        BLIP_PORT=$(( BASE_PORT + i * 10 + 1 ))
        SAM_PORT=$(( BASE_PORT + i * 10 + 2 ))
        YOLO_PORT=$(( BASE_PORT + i * 10 + 3 ))
        AGENT_ENV_EXTRA="
      GROUNDING_DINO_PORT: \"${GD_PORT}\"
      BLIP2ITM_PORT: \"${BLIP_PORT}\"
      SAM_PORT: \"${SAM_PORT}\"
      YOLOV7_PORT: \"${YOLO_PORT}\"
      POINTNAV_WEIGHTS: data/pointnav_weights.pth
      OVON_CLIP_CACHE: /opt/vlfm/data/hm3d_ovon/text_embeddings/siglip.pkl
      VLFM_AGENT_TRACE_PATH: \${VLFM_AGENT_TRACE_PATH:-${AGENT_TRACE_DEFAULT}}
      VLFM_DUMMY_POLICY_PATH: /tmp/rt_ovn/dummy_policy.pth"
        ;;
      l3mvn)
        AGENT_ENV_EXTRA="
      L3MVN_ROOT: /opt/l3mvn
      L3MVN_REDNET_PATH: data/rednet_semmap_mp3d_40.pth
      L3MVN_AGENT_TRACE_PATH: \${L3MVN_AGENT_TRACE_PATH:-${AGENT_TRACE_DEFAULT}}
      XDG_CACHE_HOME: /opt/.cache
      HF_HOME: /opt/.cache/huggingface
      TRANSFORMERS_CACHE: /opt/.cache/huggingface
      TORCH_HOME: /opt/.cache/torch
      FVCORE_CACHE: /opt/.cache/fvcore
      L3MVN_DUMP_LOCATION: /opt/worker_logs/l3mvn_tmp
      L3MVN_PRINT_IMAGES: \${L3MVN_PRINT_IMAGES:-0}
      L3MVN_EXP_NAME: subset_${i}"
        ;;
      trihelper)
        AGENT_ENV_EXTRA="
      AGENT_TRACE_PATH: \${AGENT_TRACE_PATH:-${AGENT_TRACE_DEFAULT}}
      XDG_CACHE_HOME: /opt/.cache
      HF_HOME: /opt/.cache/huggingface
      TRANSFORMERS_CACHE: /opt/.cache/huggingface
      TORCH_HOME: /opt/.cache/torch
      FVCORE_CACHE: /opt/.cache/fvcore"
        ;;
      gamap)
        AGENT_ENV_EXTRA="
      GAMAP_ROOT: /opt/gamap
      GAMAP_REDNET_PATH: data/rednet_semmap_mp3d_40.pth
      GAMAP_AGENT_TRACE_PATH: \${GAMAP_AGENT_TRACE_PATH:-${AGENT_TRACE_DEFAULT}}
      GAMAP_TASK_CONFIG: \${GAMAP_TASK_CONFIG:-tasks/objectnav_hm3d.yaml}
      GAMAP_VERSION: v1
      GAMAP_BLIP_GPU_ID: \"0\"
      XDG_CACHE_HOME: /opt/.cache
      HF_HOME: /opt/.cache/huggingface
      TRANSFORMERS_CACHE: /opt/.cache/huggingface
      TORCH_HOME: /opt/.cache/torch
      FVCORE_CACHE: /opt/.cache/fvcore
      GAMAP_DUMP_LOCATION: /opt/worker_logs/gamap_tmp
      GAMAP_PRINT_IMAGES: \${GAMAP_PRINT_IMAGES:-0}
      GAMAP_EXP_NAME: subset_${i}"
        ;;
      beliefmapnav)
        GD_PORT=$(( BASE_PORT + i * 10 + 0 ))
        SAM_PORT=$(( BASE_PORT + i * 10 + 2 ))
        YOLO_PORT=$(( BASE_PORT + i * 10 + 3 ))
        AGENT_ENV_EXTRA="
      GROUNDING_DINO_PORT: \"${GD_PORT}\"
      SAM_PORT: \"${SAM_PORT}\"
      YOLOV7_PORT: \"${YOLO_PORT}\"
      POINTNAV_WEIGHTS: data/pointnav_weights.pth
      REDNET_WEIGHTS: data/rednet_semmap_mp3d_40.pth
      BMN_OUTPUTS_DIR: /opt/beliefmapnav_outputs
      VLM_LOCKFILES_DIR: /tmp/vlfm_lockfiles
      BMN_AGENT_TRACE_PATH: \${BMN_AGENT_TRACE_PATH:-${AGENT_TRACE_DEFAULT}}
      BMN_DUMMY_LLM: \${BMN_DUMMY_LLM:-0}
      BMN_DETERMINISTIC: \${BMN_DETERMINISTIC:-1}
      CUBLAS_WORKSPACE_CONFIG: \${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
      OPENAI_API_KEY: \${OPENAI_API_KEY:-}
      OPENAI_BASE_URL: \${OPENAI_BASE_URL:-}
      gpu_id: \"0\"
      gpu_num: \"0\"
      result_path: subset_${i}
      XDG_CACHE_HOME: /opt/.cache
      HF_HOME: /opt/.cache/huggingface
      TRANSFORMERS_CACHE: /opt/.cache/huggingface
      TORCH_HOME: /opt/.cache/torch"
        ;;
      openfmnav)
        AGENT_ENV_EXTRA="
      OPENFMNAV_ROOT: /opt/openfmnav
      OPENFMNAV_TASK_CONFIG: \${OPENFMNAV_TASK_CONFIG:-tasks/objectnav_hm3d.yaml}
      OPENFMNAV_VERSION: v1
      OPENFMNAV_DUMP_LOCATION: /opt/worker_logs/openfmnav_dump
      OPENFMNAV_EXP_NAME: subset_${i}
      OPENFMNAV_AGENT_TRACE_PATH: \${OPENFMNAV_AGENT_TRACE_PATH:-${AGENT_TRACE_DEFAULT}}
      OPENFMNAV_NUM_LOCAL_STEPS: \${OPENFMNAV_NUM_LOCAL_STEPS:-20}
      OPENFMNAV_PROMPT_TYPE: \${OPENFMNAV_PROMPT_TYPE:-scoring}
      OPENFMNAV_TEXT_THRESHOLD: \${OPENFMNAV_TEXT_THRESHOLD:-0.55}
      OPENFMNAV_BOUNDARY_COEFF: \${OPENFMNAV_BOUNDARY_COEFF:-12}
      OPENFMNAV_TAG_FREQ: \${OPENFMNAV_TAG_FREQ:-100}
      OPENFMNAV_PRINT_IMAGES: \${OPENFMNAV_PRINT_IMAGES:-0}
      OPENFMNAV_DUMMY_LLM: \${OPENFMNAV_DUMMY_LLM:-}
      OPENAI_API_KEY: \"\${OPENAI_API_KEY:-}\"
      XDG_CACHE_HOME: /opt/.cache
      HF_HOME: /opt/.cache/huggingface
      TRANSFORMERS_CACHE: /opt/.cache/huggingface
      TORCH_HOME: /opt/.cache/torch"
        ;;
      rtnav)
        VLLM_PORT=$(( BASE_PORT + i * 100 ))
        AGENT_ENV_EXTRA="
      RTNAV_ROOT: /opt/rtnav
      RT_OVN_ROOT: /opt/rt_ovn
      RUN_DIR: /opt/worker_logs
      VLLM_PORT: \"${VLLM_PORT}\"
      RTNAV_LLM: \"${RTNAV_LLM}\"
      VLLM_USE_FLASHINFER_SAMPLER: \"0\"
      FLASHINFER_WORKSPACE_BASE: /opt/worker_logs/cache/flashinfer
      VLLM_CACHE_ROOT: /opt/worker_logs/cache/vllm
      XDG_CACHE_HOME: /opt/worker_logs/cache/xdg
      USER: \"rt_ovn\"
      LOGNAME: \"rt_ovn\"
      TORCHINDUCTOR_CACHE_DIR: /opt/worker_logs/cache/torchinductor
      POINTNAV_CKPT: /opt/rt_ovn/data/pointnav_weights.pth
      MOBILE_SAM_CHECKPOINT: /opt/rt_ovn/data/mobile_sam.pt
      RTNAV_ARGS: \"\${RTNAV_ARGS:-}\""
        ;;
    esac

    cat >> "$COMPOSE_FILE" <<EOF

  agent_${i}:
    <<: *common
    image: ${AGENT_IMAGE}
    container_name: ${BASELINE}_agent_${i}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${GPU}"]
              capabilities: [gpu]
    environment:
      <<: *common-env
      BENCHMARK: ${AGENT_BENCHMARK}
      VLFM_CONFIG: ${AGENT_BENCHMARK}
      NVIDIA_VISIBLE_DEVICES: "${GPU}"   # override common-env "all": isolate this worker to its GPU
      CUDA_VISIBLE_DEVICES: "0"          # the single visible GPU is index 0 inside the container
      ROS_DOMAIN_ID: "${DOMAIN_ID}"
      VIDEO_DIR: "${AGENT_VIDEO_DIR_DEFAULT}"
      AGENT_VIDEO_MODE: "${VIDEO_MODE}"
$(if [[ "$VIDEO_MODE" == "display" ]]; then
  echo "      DISPLAY: \"${DISPLAY:-}\""
  echo "      QT_X11_NO_MITSHM: \"1\""
fi)${AGENT_ENV_EXTRA}
    volumes:
$(_agent_extra_volumes)
      - ${WORKER_LOG}:/opt/worker_logs
      - ${AGENTS_DIR}/utils:/opt/rt_ovn/utils:ro
$(if [[ "$BASELINE" != "rtnav" ]]; then
  echo "      - ${BASELINE_DIR}/agent_node.py:/opt/rt_ovn/agent_node.py:ro"
fi)
$(if [[ "$VIDEO_MODE" == "display" ]]; then
  echo "      - /tmp/.X11-unix:/tmp/.X11-unix"
fi)
    depends_on:
      - env_${i}
    command: >
      bash -c "
$(_agent_command $i)
      "
EOF

    OFFSET=$(( OFFSET + EP_COUNT ))
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " Generated: $COMPOSE_FILE"
echo ""
echo " Start:  docker compose -f $COMPOSE_FILE up"
echo " Stop:   docker compose -f $COMPOSE_FILE down"
echo " Merge:  python3 $MERGE_SCRIPT --input-dir $LOG_DIR  --num-workers $NUM_WORKERS"
echo "═══════════════════════════════════════════════════════════════════"

if [[ "$LAUNCH_NOW" == "prompt" ]]; then
    read -rp "Launch now? [Y/n] " answer
    answer="${answer:-Y}"
else
    answer="$LAUNCH_NOW"
fi
if [[ "$answer" =~ ^[Yy] ]]; then
    echo "Starting $NUM_WORKERS worker pairs …"
    docker compose -f "$COMPOSE_FILE" up -d
    docker compose -f "$COMPOSE_FILE" logs -f --no-color &
    LOGS_PID=$!
    cleanup_launched_containers() {
        kill "$LOGS_PID" 2>/dev/null || true
        docker compose -f "$COMPOSE_FILE" down
    }
    trap cleanup_launched_containers EXIT INT TERM
    EXIT_CODE=0
    for (( i=0; i<NUM_WORKERS; i++ )); do
        AGENT_EXIT_CODE="$(docker wait "${BASELINE}_agent_${i}")"
        if [[ "$AGENT_EXIT_CODE" != "0" ]]; then
            EXIT_CODE="$AGENT_EXIT_CODE"
        fi
    done
    docker compose -f "$COMPOSE_FILE" down
    kill "$LOGS_PID" 2>/dev/null || true
    wait "$LOGS_PID" 2>/dev/null || true
    trap - EXIT INT TERM
    if [[ "$EXIT_CODE" != "0" ]]; then
        echo "One or more agent containers failed (exit code $EXIT_CODE)." >&2
        exit "$EXIT_CODE"
    fi
    echo ""
    echo "All containers exited. Merging stats …"
    python3 "$MERGE_SCRIPT" \
        --input-dir "$LOG_DIR" \
        --num-workers "$NUM_WORKERS"
    echo "Results: $LOG_DIR/live.json"
fi
