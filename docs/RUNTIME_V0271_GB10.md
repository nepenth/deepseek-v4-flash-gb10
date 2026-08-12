# vLLM 0.27.1 GB10 runtime

## Inputs

The runtime is built from the exact commit in `runtime/upstream.lock`:

```text
vLLM v0.27.1
commit 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
CUDA 13.0.3
PyTorch 2.13.0
FlashInfer 0.6.16.post3
CUTLASS DSL 4.6.2 / QuACK 0.6.4 (merged upstream post-release patch)
DeepGEMM 2fd67329ec2942f65ba35d561256ab6ed3b903cb
TORCH_CUDA_ARCH_LIST=12.1a
```

The upstream Dockerfile builds the `vllm-openai` target. The downstream layer
adds labels and licenses only; it does not replace vLLM's dependency pins.

## Patch policy

`runtime/patches/vllm/series` is ordered and tied to the exact base commit.
`runtime/scripts/prepare-source.sh` starts from a clean checkout, verifies the
upstream contract, checks every patch before applying it, and verifies the
patched source again.

The current series contains:

- upstream DeepSeek V4 0731 reasoning-effort mappings;
- upstream tokenizer vocabulary-size crash fix;
- upstream DSpark warmup fix when no sparse index buffer exists;
- upstream narrower DeepSeek V4 eager CUDA graph region;
- the maintained port of vLLM PR #42359's same-step prefix-cache guard,
  extended to all DeepSeek V4 cache groups and enabled for this recipe.
- vLLM PR #50796's DeepGEMM SM120/SITU pin, required to load the official
  UE8M0 checkpoint on SM121;
- the capture-stable C128A row-stride fix from vLLM PR #51318;
- merged post-release packed-KV zeroing, rejection-sampler, DSpark warmup,
  sparse-MLA workspace/index-remap, per-request chunked-context, adaptive
  scheduling, dependency-pin, and DeepSeek V4 parser fixes;
- focused mixed-batch drafter metadata and token-bound guards from the
  hardware-validated jasl SM121 preview branch.
- a direct FlashInfer SM120 sparse-MLA runner path with graph-stable split-K
  scratch for the <=64-token DeepSeek V4 decode and prefill dispatch.

The FlashInfer 0.6.16 DSV4 small-decode specialization has a hard 64-token KV
page requirement. The GB10 canary therefore fixes `KV_BLOCK_SIZE=64` on both
ranks. Although the upstream vLLM SM120 backend advertises 256-token pages,
those pages fall into FlashInfer's generic path and reject a 30-token mHC
warmup batch. The profile-level override prevents a stale private environment
from silently restoring the incompatible 256-token geometry.

The broad v0.25 overlay under `runtime/overlay/` remains for history and
attribution. It is not copied into v0.27.1.

The selection evidence, excluded candidates, and remaining hardware gates are
recorded in `docs/UPSTREAM_AUDIT_2026-08-11.md`.

## Build

On one node:

```bash
WORKER_BUILD=0 ./build-dspark-vllm-runtime.sh
```

On the configured head node, build the same pinned image on both nodes:

```bash
./build-dspark-vllm-runtime.sh
```

The resulting default image is `dspark-vllm-gb10:v0.27.1`.

For a source-only contract test:

```bash
runtime/scripts/prepare-source.sh
```

## Serve contract

The default target is the pinned
`deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint, TP=2 over two GB10 nodes,
DSpark with five speculative tokens, `deep_gemm` MXFP4 MoE, and
`fp8_ds_mla` KV.

`flashinfer_b12x` is retained for the older production image, but the pinned
vLLM source line rejects it for DeepSeek V4 MXFP4 experts. The canary must set
`MOE_BACKEND=deep_gemm`; the image builds the matching SM120 DeepGEMM path.

The patch series adds `VLLM_ALLOW_SPEC_DEC_SAME_STEP_PREFIX_HIT`:

- `2`: guard all cache groups; recipe default;
- `1`: upstream PR #42359 semantics, gated to EAGLE-marked groups;
- `0`: disable for controlled diagnosis only.

The image has been built once on GB10, but the complete two-node qualification
suite remains required before publication.
The source and static contracts do not replace
the long-context, acceptance, and throughput gates in
`docs/NVFP4_DS_MLA.md`.
