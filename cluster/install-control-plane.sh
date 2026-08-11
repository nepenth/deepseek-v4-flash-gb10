#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRIVATE_SWITCH_CONFIG="$ROOT/cluster/environments/vllm-switch.env"
[[ -f "$PRIVATE_SWITCH_CONFIG" ]] || { echo "Missing private switch config: $PRIVATE_SWITCH_CONFIG" >&2; exit 1; }
# shellcheck disable=SC1090
source "$PRIVATE_SWITCH_CONFIG"
MODELS_DIR="${MODELS_DIR:-$HOME/vllm-models}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="$MODELS_DIR/backups/control-plane-$STAMP"

[[ "${1:-}" == "--install" && $# -eq 1 ]] || {
  cat <<EOF
Usage: $0 --install

Installs the repository-owned vllm-switch implementation on the head node.
It does not stop, start, reload, or otherwise modify vllm-cluster.service.
Run only from the deployed repository on spark-1.
EOF
  exit 2
}

[[ -z "${VLLM_SWITCH_EXPECTED_HEAD_HOST:-}" || "$(hostname)" == "$VLLM_SWITCH_EXPECTED_HEAD_HOST" || "${ALLOW_OTHER_HOST:-0}" == "1" ]] || {
  echo "Refusing to install on unexpected host $(hostname)." >&2
  exit 1
}

mkdir -p "$BACKUP_DIR"
for path in \
  "$MODELS_DIR/vllm-switch" \
  "$MODELS_DIR/vllm-profile-runner" \
  "$MODELS_DIR/deepseek-v4-flash-0731-dspark.conf" \
  "$MODELS_DIR/deepseek-v4-flash-0731-v0271-canary.conf"; do
  [[ -e "$path" ]] && cp -a "$path" "$BACKUP_DIR/"
done
if [[ -e /usr/local/bin/vllm-switch || -L /usr/local/bin/vllm-switch ]]; then
  cp -a --dereference /usr/local/bin/vllm-switch "$BACKUP_DIR/vllm-switch.path-copy"
fi

install -m 0755 "$ROOT/cluster/vllm-switch" "$MODELS_DIR/vllm-switch"
install -m 0755 "$ROOT/cluster/vllm-profile-runner" "$MODELS_DIR/vllm-profile-runner"
install -m 0600 "$PRIVATE_SWITCH_CONFIG" "$MODELS_DIR/vllm-switch.env"
install -m 0644 "$ROOT/cluster/profiles/deepseek-v4-flash-0731-dspark.conf" \
  "$MODELS_DIR/deepseek-v4-flash-0731-dspark.conf"
install -m 0644 "$ROOT/cluster/profiles/deepseek-v4-flash-0731-v0271-canary.conf" \
  "$MODELS_DIR/deepseek-v4-flash-0731-v0271-canary.conf"
sudo ln -sfn "$MODELS_DIR/vllm-switch" /usr/local/bin/vllm-switch

echo "Installed control plane. Backup: $BACKUP_DIR"
echo "No service was changed. During the maintenance window, first run:"
echo "  vllm-switch adopt deepseek-v4-flash-0731-dspark"
echo "  vllm-switch validate deepseek-v4-flash-0731-v0271-canary"
