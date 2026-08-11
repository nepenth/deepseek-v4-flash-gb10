#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ -f "$ROOT/cluster/environments/vllm-switch.env" ]] && source "$ROOT/cluster/environments/vllm-switch.env"
WORKER_SSH="${WORKER_SSH:-${VLLM_SWITCH_WORKER_SSH:-}}"
MODEL_PATH="${MODEL_PATH:-/opt/models/deepseek-ai--DeepSeek-V4-Flash-0731}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/.private/checkpoint-hashes}"

[[ "${1:-}" == "--full" && $# -eq 1 ]] || {
  echo "Usage: $0 --full" >&2
  echo "Computes all checkpoint hashes on both nodes and refuses to run while serving." >&2
  exit 2
}
[[ -n "$WORKER_SSH" ]] || { echo "WORKER_SSH is required." >&2; exit 1; }
if systemctl is-active --quiet vllm-cluster.service; then
  echo "Refusing the 155+ GiB hash pass while vllm-cluster.service is active." >&2
  exit 1
fi
[[ -d "$MODEL_PATH" ]] || { echo "Missing head checkpoint: $MODEL_PATH" >&2; exit 1; }
ssh -o BatchMode=yes "$WORKER_SSH" "test -d '$MODEL_PATH'" || {
  echo "Missing worker checkpoint: $MODEL_PATH" >&2
  exit 1
}

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUTPUT_ROOT/$STAMP"
mkdir -p "$OUT"
hash_command="cd '$MODEL_PATH' && find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum"
echo "Hashing checkpoint on both ranks. Output: $OUT"
bash -lc "$hash_command" >"$OUT/head.sha256" &
head_pid=$!
ssh -o BatchMode=yes "$WORKER_SSH" "$hash_command" >"$OUT/worker.sha256" &
worker_pid=$!
wait "$head_pid"
wait "$worker_pid"
diff -u "$OUT/head.sha256" "$OUT/worker.sha256" >"$OUT/diff.txt" || {
  echo "Checkpoint hashes differ; see $OUT/diff.txt" >&2
  exit 1
}
echo "Checkpoint hashes match: $(wc -l <"$OUT/head.sha256") files"
