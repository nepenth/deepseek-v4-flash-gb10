#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ -f "$ROOT/cluster/environments/vllm-switch.env" ]] && source "$ROOT/cluster/environments/vllm-switch.env"
WORKER_SSH="${WORKER_SSH:-${VLLM_SWITCH_WORKER_SSH:-}}"
[[ -n "$WORKER_SSH" ]] || { echo "WORKER_SSH is required." >&2; exit 1; }
IMAGE="${1:-dspark-vllm-gb10:v0.27.1-gb10-rc2}"
MODEL_PATH="${MODEL_PATH:-/opt/models/deepseek-ai--DeepSeek-V4-Flash-0731}"

node_report() {
  local label="$1" mode="$2" command
  command=$(cat <<EOF
set -u
echo 'node=$label'
printf 'hostname='; hostname
printf 'arch='; uname -m
printf 'kernel='; uname -r
printf 'os='; . /etc/os-release; echo "\$PRETTY_NAME"
printf 'docker='; docker version --format '{{.Server.Version}}' 2>/dev/null || echo missing
printf 'compose='; docker compose version --short 2>/dev/null || echo missing
printf 'driver='; nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo missing
printf 'gpu='; nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used,temperature.gpu,clocks.current.sm --format=csv,noheader 2>/dev/null || echo missing
free -h
df -h /
ip -br link show enp1s0f0np0 2>/dev/null || true
ibstat rocep1s0f0 2>/dev/null | sed -n '1,24p' || true
test -d '$MODEL_PATH' && echo 'model_path=present' || echo 'model_path=MISSING'
find '$MODEL_PATH' -maxdepth 1 -type f -name '*.safetensors' -printf '%f %s\n' 2>/dev/null | sort | sha256sum | sed 's/^/model_stat_manifest=/'
docker image inspect '$IMAGE' --format 'candidate_image={{.Id}} arch={{.Architecture}} created={{.Created}}' 2>/dev/null || echo 'candidate_image=MISSING'
docker ps --format 'container={{.Names}} image={{.Image}} status={{.Status}}' | grep -Ei 'vllm|deepseek|dspark' || true
EOF
)
  if [[ "$mode" == local ]]; then
    bash -c "$command"
  else
    ssh -o BatchMode=yes "$WORKER_SSH" "bash -s" <<<"$command"
  fi
}

echo "captured_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "candidate_image=$IMAGE"
echo "model_path=$MODEL_PATH"
echo
node_report spark-1 local
echo
node_report spark-2 remote
echo
echo "service_state=$(systemctl is-active vllm-cluster.service 2>/dev/null || true)"
systemctl show vllm-cluster.service \
  -p FragmentPath -p DropInPaths -p ExecStartPre -p ExecStart -p ExecStop 2>/dev/null || true
echo
echo "switch_path=$(command -v vllm-switch 2>/dev/null || echo missing)"
if command -v vllm-switch >/dev/null 2>&1; then
  sha256sum "$(command -v vllm-switch)" 2>/dev/null || true
fi
