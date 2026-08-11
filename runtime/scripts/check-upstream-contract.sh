#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$root/upstream.lock"

source_dir="${1:-${VLLM_SOURCE_DIR:-$root/.build/vllm}}"
phase="${2:-patched}"

fail() {
  echo "contract failure: $*" >&2
  exit 1
}

[[ -d "$source_dir/.git" ]] || fail "not a git checkout: $source_dir"
actual_commit="$(git -C "$source_dir" rev-parse HEAD)"
[[ "$actual_commit" == "$VLLM_COMMIT" ]] || \
  fail "expected vLLM $VLLM_COMMIT, found $actual_commit"

require_text() {
  local pattern="$1" file="$2"
  grep -Fq -- "$pattern" "$source_dir/$file" || \
    fail "$file is missing required text: $pattern"
}

require_absent() {
  local pattern="$1" file="$2"
  if grep -Fq -- "$pattern" "$source_dir/$file"; then
    fail "$file unexpectedly contains: $pattern"
  fi
}

require_text "torch==$TORCH_VERSION" requirements/cuda.txt
require_text "flashinfer-python==$FLASHINFER_VERSION" requirements/cuda.txt
require_text "nvidia-cutlass-dsl[cu13]==$CUTLASS_DSL_VERSION" requirements/cuda.txt
require_text "apache-tvm-ffi==$TVM_FFI_VERSION" requirements/cuda.txt
require_text '12.0a;12.1a' CMakeLists.txt
require_text 'class FlashInferMLASparseSM120Impl' \
  vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
require_text 'requires the packed fp8_ds_mla' \
  vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
require_text '"flashinfer_b12x": NvFp4MoeBackend.FLASHINFER_B12X' \
  vllm/model_executor/layers/fused_moe/oracle/nvfp4.py
require_text 'class DSparkSpeculator' \
  vllm/v1/worker/gpu/spec_decode/dspark/speculator.py

# Generic nvfp4 exists for other attention backends. The DSv4 SM120 sparse
# backend must not be mislabeled as supporting it until a packed writer and
# matching reader have both landed.
require_absent 'nvfp4_ds_mla' \
  vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py

if [[ "$phase" == "patched" ]]; then
  require_text 'REASONING_EFFORT_PROMPTS' vllm/tokenizers/deepseek_v4_encoding.py
  require_text 'VLLM_ALLOW_SPEC_DEC_SAME_STEP_PREFIX_HIT' vllm/envs.py
  require_text '_ghost_block_guard_enabled' \
    vllm/v1/core/single_type_kv_cache_manager.py
  git -C "$source_dir" diff --check
fi

echo "vLLM $phase contract OK: $actual_commit"
