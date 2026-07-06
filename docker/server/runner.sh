#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Single control script for the containerized self-hosted GPU runner.
#
#   ./runner.sh run <TOKEN> [NAME]   first launch; NAME defaults to rust-gpu-runner
#   ./runner.sh start|stop|restart   control the existing container (no token needed)
#   ./runner.sh logs                 follow the runner log
#   ./runner.sh status               show container state + nvidia-smi / nvcc
#   ./runner.sh rm                   stop and delete the container
#   ./runner.sh build                (re)build the image
#
# Only `run` needs a token; restarts reuse the persisted registration.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-ultralytics-gpu-runner:cuda12.8}"
NAME="${RUNNER_CONTAINER:-rust-gpu-runner}"
URL="${RUNNER_URL:-https://github.com/ultralytics/inference}"
LABELS="${RUNNER_LABELS:-rust-gpu-runner}"

build() { docker build -t "$IMAGE" "$SCRIPT_DIR"; }

run() {
  local token="${1:?usage: runner.sh run <TOKEN> [NAME]}"
  local display="${2:-$NAME}"
  docker image inspect "$IMAGE" > /dev/null 2>&1 || build
  docker rm -f "$NAME" > /dev/null 2>&1 || true
  docker run -d --restart unless-stopped --gpus all \
    --name "$NAME" \
    -e RUNNER_URL="$URL" \
    -e RUNNER_TOKEN="$token" \
    -e RUNNER_NAME="$display" \
    -e RUNNER_LABELS="$LABELS" \
    "$IMAGE"
  echo "started '$NAME' (label: $LABELS) -> $URL"
}

case "${1:-}" in
  build) build ;;
  run)
    shift
    run "$@"
    ;;
  start) docker start "$NAME" ;;
  stop) docker stop "$NAME" ;;
  restart) docker restart "$NAME" ;;
  logs) docker logs -f "$NAME" ;;
  status)
    docker ps -a --filter "name=$NAME" --format 'table {{.Names}}\t{{.Status}}'
    docker exec "$NAME" bash -lc 'nvidia-smi; nvcc --version' 2> /dev/null \
      || echo "(container not running; GPU/toolkit check skipped)"
    ;;
  rm) docker rm -f "$NAME" ;;
  *)
    echo "usage: $0 {run <TOKEN> [NAME]|start|stop|restart|logs|status|rm|build}" >&2
    exit 1
    ;;
esac
