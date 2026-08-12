#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$root/upstream.lock"

build_root="${BUILD_ROOT:-$root/.build}"
source_dir="${VLLM_SOURCE_DIR:-$build_root/vllm}"
docker_cmd="${DOCKER:-docker}"
base_image="${BASE_IMAGE:-dspark-vllm-gb10:vllm-base-$VLLM_TAG}"
final_image="${FINAL_IMAGE:-dspark-vllm-gb10:$VLLM_VERSION}"

VLLM_SOURCE_DIR="$source_dir" "$root/scripts/prepare-source.sh"

upstream_dockerfile="${VLLM_DOCKERFILE:-$source_dir/docker/Dockerfile}"
if [[ ! -f "$upstream_dockerfile" ]]; then
  echo "vLLM Dockerfile not found: $upstream_dockerfile" >&2
  exit 1
fi

"$docker_cmd" build \
  --file "$upstream_dockerfile" \
  --target vllm-openai \
  --build-arg "torch_cuda_arch_list=$TORCH_CUDA_ARCH_LIST" \
  --build-arg "max_jobs=$BUILD_MAX_JOBS" \
  --build-arg "nvcc_threads=$BUILD_NVCC_THREADS" \
  --build-arg "VLLM_BUILD_COMMIT=$VLLM_COMMIT" \
  --build-arg "VLLM_IMAGE_TAG=$final_image" \
  --tag "$base_image" \
  "$source_dir"

"$docker_cmd" build \
  --file "$root/docker/Dockerfile.runtime" \
  --build-arg "VLLM_BASE=$base_image" \
  --build-arg "VLLM_COMMIT=$VLLM_COMMIT" \
  --build-arg "VLLM_VERSION=$VLLM_VERSION" \
  --build-arg "FLASHINFER_VERSION=$FLASHINFER_VERSION" \
  --build-arg "FLASHINFER_DSV4_SM120_COMMIT=$FLASHINFER_DSV4_SM120_COMMIT" \
  --build-arg "FLASHINFER_DSV4_SM120_PATCH_SHA256=$FLASHINFER_DSV4_SM120_PATCH_SHA256" \
  --tag "$final_image" \
  "$root"

"$docker_cmd" run --rm --entrypoint python3 "$final_image" -c \
  'from importlib.metadata import version; from pathlib import Path; from flashinfer.mla._sparse_mla_sm120 import _DECODE_DSV4_DISPATCH; import flashinfer_jit_cache, torch, vllm; assert (32, 192) in _DECODE_DSV4_DISPATCH; assert (32, 256) in _DECODE_DSV4_DISPATCH; assert not (Path(flashinfer_jit_cache.get_jit_cache_dir()) / "sparse_mla_sm120" / "sparse_mla_sm120.so").exists(); print({"vllm": vllm.__version__, "torch": torch.__version__, "flashinfer": version("flashinfer-python"), "dsv4_sm120_topk": [192, 256]})'

echo "Built $final_image from vLLM $VLLM_COMMIT for SM121"
