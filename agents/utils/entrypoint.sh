#!/usr/bin/env bash
set -eo pipefail

_infer_host_user() {
    local path
    IFS=: read -ra paths <<< "${HOST_USER_PATHS:-${HOST_USER_PATH:-$PWD}}"
    for path in "${paths[@]}"; do
        if [ -e "$path" ]; then
            HOST_UID="${HOST_UID:-$(stat -c '%u' "$path")}"
            HOST_GID="${HOST_GID:-$(stat -c '%g' "$path")}"
            return
        fi
    done
}

_prepare_writable_paths() {
    local path
    IFS=: read -ra paths <<< "${HOST_CHOWN_PATHS:-}"
    for path in "${paths[@]}"; do
        [ -n "$path" ] || continue
        mkdir -p "$path" 2>/dev/null || true
        chown -R "${HOST_UID}:${HOST_GID}" "$path" 2>/dev/null || true
    done
}

if [ "$(id -u)" = "0" ] && [ "${RT_OVN_RUN_AS_HOST_USER:-1}" != "0" ]; then
    _infer_host_user
    if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ] && [ "$HOST_UID" != "0" ]; then
        export HOST_UID HOST_GID
        _prepare_writable_paths
        exec gosu "${HOST_UID}:${HOST_GID}" "$0" "$@"
    fi
fi

if [ -z "${HOME:-}" ] || [ "$HOME" = "/" ]; then
    export HOME=/tmp
fi
export USER="${USER:-rt_ovn}"
export LOGNAME="${LOGNAME:-$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torchinductor}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/matplotlib}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$HOME/.ros/log}"
mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$ROS_LOG_DIR" 2>/dev/null || true

# Source ROS2 workspace if available
if [ -f /opt/ros2_ws/install/setup.bash ]; then
    set +u; source /opt/ros2_ws/install/setup.bash; set -u
fi

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Create symlinks for GroundingDINO/yolov7 if available (VLFM only)
for pkg in GroundingDINO yolov7; do
    if [ -d "/opt/$pkg" ] && [ -d /opt/vlfm ] && [ ! -e "/opt/vlfm/$pkg" ]; then
        ln -s "/opt/$pkg" "/opt/vlfm/$pkg" 2>/dev/null || true
    fi
done

exec "$@"
