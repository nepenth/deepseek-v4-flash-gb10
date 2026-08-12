# Upstream audit: vLLM 0.27.1, DeepSeek V4, and GB10

Audit date: 2026-08-11. This is the evidence record for the runtime patch
selection, not a claim of two-node hardware validation.

## Baseline

- vLLM 0.27.1 (`6e448d0e`) is the latest published release at audit time.
- CUDA 13.0.3, PyTorch 2.13.0, and FlashInfer 0.6.16.post3 remain owned by
  that release. The candidate does not independently upgrade FlashInfer across
  its ABI; it applies the exact, checksummed SM120 DSV4 source delta described
  in the post-audit correction below.
- CUTLASS DSL 4.6.2 and QuACK 0.6.4 move as the paired versions merged in
  vLLM PR #51566.
- The official checkpoint is pinned to revision
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`, matching NVIDIA's dual-Spark
  experimental recipe.

## Required corrections

The released DeepGEMM pin cannot transform the checkpoint's UE8M0 scale
layout on SM120/SM121. Patch 0009 takes the combined SM120+SITU DeepGEMM pin
from PR #50796. Its parent was exercised with the full checkpoint on SM120;
the exact child adds the CUDA FP8 header compatibility change. The exact child
still needs a GB10 image build here.

Patch 0010 fixes C128A decode metadata whose row stride changed between CUDA
graph capture and a runtime batch. The failure affects rows after row zero at
long context and concurrency. PR #51318 reports an SM121 reproduction plus a
4x GB10 A/B with zero failures after the fix.

Patches 0011-0017 cover NaN/token bounds, physical packed-cache zeroing,
mixed-batch drafter metadata invalidation, CUDA mHC warmup, DSpark token
initialization, and speculative warmup block accounting. These address
observed corruption, crash, or cold-JIT modes rather than tuning preferences.

## Selected performance work

Patches 0018-0021 are bounded, upstream-merged changes: sparse masked-MHA
workspace fallback, sparse index remap, per-request MLA context chunks, and
adaptive speculative input budgeting. Each preserves a fallback or the same
math and has upstream correctness coverage. The adaptive-budget benchmark was
on Kimi K3 DSpark, so its DeepSeek V4 gain remains a GB10 measurement item.
PR #51733 is not carried separately: it reverts a workspace-size requirement
introduced by later PR #50484, which is absent from the v0.27.1 base used here.

## Deliberately deferred

- MRV2 fused multi-step CUDA graphs (PR #46849): upstream DeepSeek V4 results
  showed no systematic device-bound gain, while the metadata surface is large.
- Long-context cache-gather PR #51739: merged and correct, but its FP8
  upconvert gain was modest and measured on GB300; it adds a large CUDA delta.
- TokenSpeed non-causal DSpark PR #50911: validated on B200/Kimi K3, not GB10
  DeepSeek V4, and may change backend selection.
- A true packed `nvfp4_ds_mla`: no upstream DeepSeek V4 writer/reader exists.
  The 352/368-byte research records are based on 512+64 geometry, not DeepSeek
  V4's 448-dimension NoPE plus 64-dimension RoPE geometry.
- Broad FlashInfer post-release upgrades: vLLM main still owns 0.6.16.post3,
  so changing the package alone would trade a tested dependency graph for an
  unverified one.

## Post-audit correction: DSpark SM120 DSV4 top-k 256

The first two-node candidate runs exposed an SM120 sparse-MLA dispatch gap
that was not resolved by the original patch series. DeepSeek V4's DSpark K=5
non-causal decode has 133 active SWA entries, rounded by vLLM to a 256-wide
index tensor. FlashInfer `0.6.16.post3` supplies direct DSV4 SM120 decode only
for top-k 128, 512, and 1024, so the 256 shape falls into the prefill-only
orchestrator and fails for the 30-token mHC warmup batch.

FlashInfer PR #4380, merged as
`24d7dfb2639083c5a4d418881099421fc800b7bb`, adds compiled DSV4 192/256 decode
and prefill dispatch plus a fail-loud Python guard. The `v0.6.17` tag does not
contain this commit. The candidate therefore retains `flashinfer-python`
`0.6.16.post3`, overlays just that upstream diff, deletes only the stale
`sparse_mla_sm120` AOT module, and JIT-builds the patched module in a separate
mounted workspace. This avoids a broad dependency replacement while using the
newest upstream implementation of the required kernel.

## Primary references

- <https://github.com/vllm-project/vllm/releases/tag/v0.27.1>
- <https://github.com/vllm-project/vllm/issues/51758>
- <https://github.com/vllm-project/vllm/pull/50796>
- <https://github.com/vllm-project/vllm/pull/51318>
- <https://github.com/vllm-project/vllm/pull/41834>
- <https://github.com/vllm-project/vllm/issues/51790>
- <https://github.com/vllm-project/vllm/pull/51802>
- <https://github.com/vllm-project/vllm/pull/50613>
- <https://github.com/vllm-project/vllm/pull/50365>
- <https://github.com/vllm-project/vllm/pull/51725>
- <https://github.com/vllm-project/vllm/pull/51566>
- <https://github.com/flashinfer-ai/flashinfer/pull/4380>
- <https://github.com/NVIDIA/NemoClaw/blob/main/managed-inference/recipes/vllm.deepseek-v4-flash-0731.spark-dual.v1.yaml>
- <https://github.com/lrozewicz/vLLM-Moet-GB10>

## Remaining proof

The patch series applies cleanly and its source/static contracts are tested on
the development host. This host is x86_64/SM89, so it cannot validate the
ARM64 image, SM121 code generation, TP=2 communication, full-checkpoint load,
or the acceptance/latency/quality matrix. Those are release gates, not items
that source inspection can prove.
