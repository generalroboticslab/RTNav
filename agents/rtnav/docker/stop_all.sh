#!/usr/bin/env bash
# Stop RTNav compose stacks and containers left behind by Ctrl-C.
#
# Usage:
#   ./agents/rtnav/docker/stop_all.sh         # stop+remove containers, keep volumes
#   ./agents/rtnav/docker/stop_all.sh --hard  # also drop volumes + named networks
#
# Idempotent: safe to run when nothing is up.
set -euo pipefail

DOCKER_DIR="$(dirname "$(realpath "$0")")"
METHOD_DIR="$(dirname "$DOCKER_DIR")"

HARD=0
if [[ "${1:-}" == "--hard" ]]; then
    HARD=1
fi

DOWN_FLAGS="--remove-orphans"
if [[ $HARD -eq 1 ]]; then
    DOWN_FLAGS="$DOWN_FLAGS --volumes"
fi

# 1) Compose stacks. `down` is a no-op if nothing is up.
for f in "$METHOD_DIR"/docker-compose*.yml; do
    [[ -f "$f" ]] || continue
    echo "=== compose down: $f ==="
    docker compose -f "$f" down $DOWN_FLAGS || true
    echo ""
done

# 2) Sweep up RTNav containers not attached to the current Compose project.
echo "=== sweeping leftover rtnav containers ==="
ALL=$(docker ps -aq --format '{{.ID}} {{.Names}}' 2>/dev/null \
    | grep -i 'rtnav' | awk '{print $1}' | sort -u || true)

if [[ -n "$ALL" ]]; then
    echo "stopping: $ALL"
    docker stop $ALL >/dev/null 2>&1 || true
    docker rm   $ALL >/dev/null 2>&1 || true
else
    echo "no leftovers found"
fi
echo ""

# 3) Confirm.
echo "=== remaining rtnav containers ==="
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' \
    | grep -i 'rtnav' || echo "  none"
