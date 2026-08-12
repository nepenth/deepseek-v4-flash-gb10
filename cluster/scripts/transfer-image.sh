#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

IMAGE="${1:-dspark-vllm-gb10:v0.27.1-gb10-rc3}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ -f "$ROOT/cluster/environments/vllm-switch.env" ]] && source "$ROOT/cluster/environments/vllm-switch.env"
WORKER_SSH="${WORKER_SSH:-${VLLM_SWITCH_WORKER_SSH:-}}"
[[ -n "$WORKER_SSH" ]] || { echo "WORKER_SSH is required." >&2; exit 1; }

[[ $# -le 1 ]] || { echo "Usage: $0 [IMAGE]" >&2; exit 2; }
head_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')" || {
  echo "Missing local image: $IMAGE" >&2
  exit 1
}

echo "Streaming $IMAGE ($head_id) to $WORKER_SSH"
docker save "$IMAGE" | ssh -o BatchMode=yes "$WORKER_SSH" docker load
worker_id="$(ssh -o BatchMode=yes "$WORKER_SSH" \
  "docker image inspect '$IMAGE' --format '{{.Id}}'")"
[[ "$head_id" == "$worker_id" ]] || {
  echo "Image ID mismatch: head=$head_id worker=$worker_id" >&2
  exit 1
}
echo "Image IDs match: $head_id"
