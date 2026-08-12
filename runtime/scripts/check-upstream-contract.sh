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
if [[ "$phase" == "patched" ]]; then
  require_text "nvidia-cutlass-dsl[cu13]==$CUTLASS_DSL_VERSION" requirements/cuda.txt
  require_text "quack-kernels==$QUACK_VERSION" requirements/cuda.txt
else
  require_text 'nvidia-cutlass-dsl[cu13]==4.6.0' requirements/cuda.txt
  require_text 'quack-kernels==0.6.1' requirements/cuda.txt
fi
require_text "apache-tvm-ffi==$TVM_FFI_VERSION" requirements/cuda.txt
require_text '12.0a;12.1a' CMakeLists.txt
require_text 'class FlashInferMLASparseSM120Impl' \
  vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
require_text 'requires the packed fp8_ds_mla' \
  vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
require_text '"deep_gemm": [Mxfp4MoeBackend.DEEPGEMM_MXFP4]' \
  vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
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
  require_text "$DEEPGEMM_COMMIT" cmake/external_projects/deepgemm.cmake
  require_text "$DEEPGEMM_COMMIT" tools/install_deepgemm.sh
  require_text 'active_topk_width = self.c128a_max_compressed' \
    vllm/models/deepseek_v4/sparse_mla.py
  require_text 'local_max != local_max' \
    vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
  require_text 'seg_block_strides_ptr' vllm/v1/worker/utils.py
  require_text '_token_to_req_indices_cache = None' \
    vllm/v1/spec_decode/llm_base_proposer.py
  require_text 'vocab_size - 1' vllm/v1/worker/gpu/sample/gumbel.py
  require_text '_uses_mhc_tilelang' \
    vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py
  require_text 'GLOBAL_TOPK_MASK_MAX_BYTES = 128 * 1024 * 1024' \
    vllm/model_executor/layers/attention/sparse_mla_attention.py
  require_text 'class ContextChunk' \
    vllm/model_executor/layers/attention/mla_attention.py
  require_text 'input_budget = self.scheduler_config.max_num_batched_tokens' \
    vllm/v1/core/sched/scheduler.py
  require_text '"thinking" not in chat_kwargs' vllm/parser/deepseek_v4.py
  require_text 'GIT_SUBMODULES csrc/cutlass' \
    cmake/external_projects/vllm_flash_attn.cmake
  require_text 'PATCH_COMMAND git apply --unidiff-zero --whitespace=nowarn' \
    cmake/external_projects/vllm_flash_attn.cmake
  require_text 'target architectures do not include Hopper SM90' \
    cmake/external_projects/patches/flash-attn-skip-fa3-without-sm90.patch
  require_text 'add_custom_target(_vllm_fa3_C)' \
    cmake/external_projects/patches/flash-attn-skip-fa3-without-sm90.patch
  require_text 'GIT_SHALLOW TRUE' \
    cmake/external_projects/triton_kernels.cmake
  require_text 'VLLM_ALLOW_SPEC_DEC_SAME_STEP_PREFIX_HIT' vllm/envs.py
  require_text '_ghost_block_guard_enabled' \
    vllm/v1/core/single_type_kv_cache_manager.py
  require_text '_SparseMLAPagedAttentionRunner' \
    vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
  require_text '_reserve_sm120_decode_workspace' \
    vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
  require_text 'mid_out=mid_out' \
    vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
  git -C "$source_dir" diff --check
fi

echo "vLLM $phase contract OK: $actual_commit"
