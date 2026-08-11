#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ -f "$ROOT/cluster/environments/vllm-switch.env" ]] && source "$ROOT/cluster/environments/vllm-switch.env"
WORKER_SSH="${WORKER_SSH:-${VLLM_SWITCH_WORKER_SSH:-}}"
[[ -n "$WORKER_SSH" ]] || { echo "WORKER_SSH is required." >&2; exit 1; }
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/.private/cluster-runs}"
LABEL=state

if [[ "${1:-}" == "--label" ]]; then
  [[ $# -eq 2 && "$2" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Usage: $0 [--label LABEL]" >&2; exit 2; }
  LABEL="$2"
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--label LABEL]" >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUTPUT_ROOT/$STAMP-$LABEL"
mkdir -p "$OUT"

"$ROOT/cluster/scripts/cluster-preflight.sh" >"$OUT/manifest.txt" 2>&1 || true
systemctl cat vllm-cluster.service >"$OUT/vllm-cluster.unit.txt" 2>&1 || true
systemctl status vllm-cluster.service --no-pager >"$OUT/vllm-cluster.status.txt" 2>&1 || true
journalctl -u vllm-cluster.service --since '-45 min' --no-pager >"$OUT/head-systemd.log" 2>&1 || true
docker ps -a --no-trunc >"$OUT/head-containers.txt" 2>&1 || true
docker logs --since 45m vllm_ds4_0731 >"$OUT/head-vllm.log" 2>&1 || \
  docker logs --since 45m dsv4-v0271-canary-vllm-dspark-1 >"$OUT/head-vllm.log" 2>&1 || true
ssh -o BatchMode=yes "$WORKER_SSH" "docker ps -a --no-trunc" >"$OUT/worker-containers.txt" 2>&1 || true
ssh -o BatchMode=yes "$WORKER_SSH" "docker logs --since 45m vllm_ds4_0731" >"$OUT/worker-vllm.log" 2>&1 || \
  ssh -o BatchMode=yes "$WORKER_SSH" "docker logs --since 45m dsv4-v0271-canary-vllm-dspark-1" >"$OUT/worker-vllm.log" 2>&1 || true
curl -fsS --max-time 10 http://127.0.0.1:8888/v1/models >"$OUT/models.json" 2>"$OUT/models.stderr" || true
curl -fsS --max-time 15 http://127.0.0.1:8888/metrics >"$OUT/metrics.txt" 2>"$OUT/metrics.stderr" || true

if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$ROOT" rev-parse HEAD >"$OUT/source-commit.txt"
  git -C "$ROOT" status --short >"$OUT/source-status.txt"
  sha256sum "$ROOT"/runtime/patches/vllm/*.patch >"$OUT/patch-sha256.txt"
elif [[ -s "$ROOT/.source-commit" ]]; then
  cp "$ROOT/.source-commit" "$OUT/source-commit.txt"
  echo "git-archive deployment" >"$OUT/source-status.txt"
  sha256sum "$ROOT"/runtime/patches/vllm/*.patch >"$OUT/patch-sha256.txt"
fi

echo "$OUT"
