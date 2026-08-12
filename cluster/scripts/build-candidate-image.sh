#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${CANDIDATE_IMAGE:-dspark-vllm-gb10:v0.27.1-gb10-rc6}"
MIN_FREE_GIB="${MIN_FREE_GIB:-120}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BUILD_RECORD="${BUILD_RECORD:-$ROOT/.private/builds/$STAMP}"

source_commit() {
  if git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1; then
    git -C "$ROOT" rev-parse HEAD
  elif [[ -s "$ROOT/.source-commit" ]]; then
    cat "$ROOT/.source-commit"
  else
    echo unknown
  fi
}

[[ "${1:-}" == "--build" && $# -eq 1 ]] || {
  echo "Usage: $0 --build" >&2
  echo "Refuses to build while vllm-cluster.service or a vLLM container is active." >&2
  exit 2
}

mkdir -p "$BUILD_RECORD"
exec > >(tee "$BUILD_RECORD/build.log") 2>&1

if systemctl is-active --quiet vllm-cluster.service; then
  echo "Refusing to build while vllm-cluster.service is active." >&2
  exit 1
fi
if docker ps --format '{{.Names}}' | grep -Ei 'vllm|dspark|deepseek' >/dev/null; then
  echo "Refusing to build while a matching inference container is active." >&2
  docker ps --format '  {{.Names}} {{.Image}} {{.Status}}' | grep -Ei 'vllm|dspark|deepseek' >&2 || true
  exit 1
fi

free_gib="$(df -BG --output=avail "$ROOT" | tail -n 1 | tr -dc '0-9')"
[[ -n "$free_gib" && "$free_gib" -ge "$MIN_FREE_GIB" ]] || {
  echo "Need at least ${MIN_FREE_GIB} GiB free for the ARM64 source build; found ${free_gib:-unknown}." >&2
  exit 1
}

echo "Building $IMAGE from $(source_commit)"
source_commit >"$BUILD_RECORD/source-commit.txt"
if git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1; then
  git -C "$ROOT" status --short >"$BUILD_RECORD/source-status.txt"
else
  echo "clean git-archive deployment; see .source-commit" >"$BUILD_RECORD/source-status.txt"
fi
sha256sum "$ROOT"/runtime/patches/vllm/*.patch >"$BUILD_RECORD/patch-sha256.txt"
sha256sum "$ROOT"/runtime/patches/flashinfer/*.patch >>"$BUILD_RECORD/patch-sha256.txt"
cp "$ROOT/runtime/upstream.lock" "$BUILD_RECORD/upstream.lock"
FINAL_IMAGE="$IMAGE" "$ROOT/runtime/scripts/build-image.sh"
docker image inspect "$IMAGE" >"$BUILD_RECORD/image-inspect.json"
docker image inspect "$IMAGE" --format 'image={{.RepoTags}} id={{.Id}} arch={{.Architecture}} created={{.Created}}'
docker run --rm -i --entrypoint python3 "$IMAGE" - <<'PY' | tee "$BUILD_RECORD/python-packages.txt"
import importlib.metadata as md
for package in ("vllm", "torch", "flashinfer-python", "nvidia-cutlass-dsl", "quack-kernels"):
    try:
        print(f"{package}={md.version(package)}")
    except md.PackageNotFoundError:
        print(f"{package}=MISSING")
PY
docker run --rm --entrypoint python3 "$IMAGE" -m pip list --format=json >"$BUILD_RECORD/pip-list.json"
if command -v syft >/dev/null 2>&1; then
  syft "$IMAGE" -o spdx-json="$BUILD_RECORD/sbom.spdx.json"
else
  echo "syft is unavailable; pip-list.json and image-inspect.json are the package manifest." \
    >"$BUILD_RECORD/sbom-unavailable.txt"
fi
echo "Build record: $BUILD_RECORD"
