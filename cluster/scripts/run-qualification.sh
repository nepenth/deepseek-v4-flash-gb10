#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8888/v1}"
MODEL=""
LABEL=""
MODE=quick

usage() {
  echo "Usage: $0 --run --label LABEL --model SERVED_MODEL [--mode quick|full]" >&2
  exit 2
}

[[ "${1:-}" == "--run" ]] || usage
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) [[ $# -ge 2 ]] || usage; LABEL="$2"; shift 2 ;;
    --model) [[ $# -ge 2 ]] || usage; MODEL="$2"; shift 2 ;;
    --mode) [[ $# -ge 2 ]] || usage; MODE="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$LABEL" =~ ^[A-Za-z0-9._-]+$ && -n "$MODEL" ]] || usage
[[ "$MODE" == quick || "$MODE" == full ]] || usage

curl -fsS --max-time 10 "$BASE_URL/models" | grep -Fq "$MODEL" || {
  echo "Served model is not ready: $MODEL" >&2
  exit 1
}

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUTPUT_ROOT:-$ROOT/.private/qualification}/$STAMP-$LABEL-$MODE"
mkdir -p "$OUT"
capture_post() {
  "$ROOT/cluster/scripts/capture-cluster-state.sh" --label "$LABEL-post" >"$OUT/post-capture-path.txt" 2>&1 || true
}
trap capture_post EXIT
"$ROOT/cluster/scripts/capture-cluster-state.sh" --label "$LABEL-pre" >"$OUT/pre-capture-path.txt"
curl -fsS "$BASE_URL/models" >"$OUT/models.json"

if [[ "$MODE" == full ]]; then
  ladder="8192,32768,131072,300000,380000"
  soak_minutes=60
  soak_concurrency=3
  bench_lengths="256,8192,32768,131072,300000"
  bench_concurrency="1,2,4,6"
  retrieval_lengths="8192,32768,131072,300000,380000"
else
  ladder="8192,32768"
  soak_minutes=12
  soak_concurrency=2
  bench_lengths="256,8192,32768"
  bench_concurrency="1,2"
  retrieval_lengths="8192,32768"
fi

BASE_URL="$BASE_URL" MODEL="$MODEL" CONCURRENCY=6 \
  "$ROOT/smoke-deepseek-v4-flash-dspark.sh" | tee "$OUT/concurrency-smoke.log"
python3 "$ROOT/scripts/api-contract-qualification.py" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --output "$OUT/api-contract.json" | tee "$OUT/api-contract.log"
python3 "$ROOT/scripts/stability-quick.py" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --skip-vision \
  --ladder "$ladder" \
  --soak-minutes "$soak_minutes" \
  --soak-concurrency "$soak_concurrency" \
  --output "$OUT/stability.json" | tee "$OUT/stability.log"
python3 "$ROOT/scripts/benchmark-0731.py" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --prompt-lengths "$bench_lengths" \
  --concurrency "$bench_concurrency" \
  --output "$OUT/benchmark.json" | tee "$OUT/benchmark.log"
python3 "$ROOT/scripts/retrieval-qualification.py" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --lengths "$retrieval_lengths" \
  --output "$OUT/retrieval.json" | tee "$OUT/retrieval.log"

echo "$OUT"
