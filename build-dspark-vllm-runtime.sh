#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

DSPARK_VLLM_IMAGE="${DSPARK_VLLM_IMAGE:-dspark-vllm-gb10:v0.27.1}"
WORKER_BUILD="${WORKER_BUILD:-1}"

build_local() {
  FINAL_IMAGE="$DSPARK_VLLM_IMAGE" "$SCRIPT_DIR/runtime/scripts/build-image.sh"
}

build_worker() {
  local checkout="${WORKER_CHECKOUT:-${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}}"
  : "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"
  ssh "$WORKER_HOST" "mkdir -p '$checkout'"
  rsync -az --delete \
    --exclude='.git/' \
    --exclude='.build/' \
    --exclude='runtime/.build/' \
    "$SCRIPT_DIR/" "$WORKER_HOST:$checkout/"
  ssh "$WORKER_HOST" \
    "cd '$checkout' && FINAL_IMAGE='$DSPARK_VLLM_IMAGE' runtime/scripts/build-image.sh"
}

build_local
if [ "$WORKER_BUILD" = "1" ]; then
  build_worker
fi
