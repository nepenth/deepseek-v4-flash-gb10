#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRIVATE_DIR="${PRIVATE_DIR:-$ROOT/.private}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-$PRIVATE_DIR/deploy.env}"
[[ -f "$DEPLOY_CONFIG" ]] && source "$DEPLOY_CONFIG"
HEAD_SSH="${HEAD_SSH:-CHANGEME}"
WORKER_SSH="${WORKER_SSH:-CHANGEME}"
REMOTE_DIR="${REMOTE_DIR:-CHANGEME}"

[[ "${1:-}" == "--apply" && $# -eq 1 ]] || {
  cat <<EOF
Dry run only. No remote files were changed.

Apply with:
  $0 --apply

Targets:
  head:   $HEAD_SSH:$REMOTE_DIR
  worker: $WORKER_SSH:$REMOTE_DIR

This copies the current working tree without .git or build/cache artifacts. It
also installs the two ignored cluster configuration inputs from .private on
the head. It does not install the control plane, build an image, or touch the
live service.
EOF
  exit 0
}

[[ "$HEAD_SSH" != CHANGEME && "$WORKER_SSH" != CHANGEME && "$REMOTE_DIR" != CHANGEME ]] || {
  echo "Set HEAD_SSH, WORKER_SSH, and REMOTE_DIR in $DEPLOY_CONFIG." >&2
  exit 1
}

for file in \
  "$PRIVATE_DIR/deepseek-v4-flash-0731-v0271-canary.env" \
  "$PRIVATE_DIR/vllm-switch.env"; do
  [[ -f "$file" ]] || { echo "Missing private deployment input: $file" >&2; exit 1; }
done

deploy_one() {
  local target="$1" source_commit="$2"
  echo "Deploying repository to $target:$REMOTE_DIR"
  git -C "$ROOT" archive --format=tar HEAD | ssh -F /dev/null -o BatchMode=yes "$target" \
      "mkdir -p '$REMOTE_DIR' && tar -C '$REMOTE_DIR' --no-overwrite-dir -xf -"
  ssh -F /dev/null -o BatchMode=yes "$target" \
    "printf '%s\n' '$source_commit' > '$REMOTE_DIR/.source-commit'"
}

git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet || {
  echo "Commit the intended deployment tree before using --apply." >&2
  exit 1
}

SOURCE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
deploy_one "$HEAD_SSH" "$SOURCE_COMMIT"
deploy_one "$WORKER_SSH" "$SOURCE_COMMIT"

scp -F /dev/null "$PRIVATE_DIR/deepseek-v4-flash-0731-v0271-canary.env" \
  "$HEAD_SSH:$REMOTE_DIR/cluster/environments/deepseek-v4-flash-0731-v0271-canary.env"
scp -F /dev/null "$PRIVATE_DIR/vllm-switch.env" \
  "$HEAD_SSH:$REMOTE_DIR/cluster/environments/vllm-switch.env"

local_manifest="$(
  cd "$ROOT"
  sha256sum \
    cluster/vllm-switch \
    cluster/vllm-profile-runner \
    docker-compose.dspark.yml \
    start-deepseek-v4-flash-dspark.sh \
    stop-deepseek-v4-flash-dspark.sh
)"
for target in "$HEAD_SSH" "$WORKER_SSH"; do
  remote_manifest="$(ssh -F /dev/null -o BatchMode=yes "$target" \
    "cd '$REMOTE_DIR' && sha256sum cluster/vllm-switch cluster/vllm-profile-runner docker-compose.dspark.yml start-deepseek-v4-flash-dspark.sh stop-deepseek-v4-flash-dspark.sh")"
  [[ "$remote_manifest" == "$local_manifest" ]] || {
    echo "Deployment manifest mismatch on $target" >&2
    diff -u <(printf '%s\n' "$local_manifest") <(printf '%s\n' "$remote_manifest") || true
    exit 1
  }
  remote_commit="$(ssh -F /dev/null "$target" "cat '$REMOTE_DIR/.source-commit'")"
  [[ "$remote_commit" == "$SOURCE_COMMIT" ]] || { echo "Source commit marker mismatch on $target" >&2; exit 1; }
done
private_sha="$(sha256sum "$PRIVATE_DIR/deepseek-v4-flash-0731-v0271-canary.env" | awk '{print $1}')"
remote_private_sha="$(ssh -F /dev/null "$HEAD_SSH" \
  "sha256sum '$REMOTE_DIR/cluster/environments/deepseek-v4-flash-0731-v0271-canary.env' | cut -d' ' -f1")"
[[ "$private_sha" == "$remote_private_sha" ]] || { echo "Private canary environment hash mismatch on head" >&2; exit 1; }
echo "Both node deployment manifests match. The live service was not changed."
