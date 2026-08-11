# NVFP4 DS-MLA status and implementation contract

## Decision

The vLLM 0.27.1 production lane uses `--kv-cache-dtype fp8_ds_mla`.

The previous runtime accepted `nvfp4_ds_mla`, but its implementation routed
that name to the same FP8 DS-MLA record and FP8 attention kernel. It preserved
the desired long-context behavior, but it was not a packed 4-bit KV cache.
This repo accepts that spelling only as a launcher compatibility alias and
normalizes it to `fp8_ds_mla`.

vLLM 0.27.1 also accepts a generic `nvfp4` cache dtype for other attention
paths. That does not make it compatible with DeepSeek V4 sparse MLA on SM121.
The native `FlashInferMLASparseSM120Impl` explicitly requires
`fp8_ds_mla`, and no upstream writer/reader pair exists for packed NVFP4
DeepSeek V4 sparse MLA at this tag.

## Current production path

vLLM 0.27.1 already contains the components that the older overlay supplied:

- DeepSeek V4 CUDA model and sparse-MLA integration;
- the SM120/SM121 FlashInfer sparse-MLA backend;
- DSpark model-runner-v2 speculative decoding;
- `flashinfer_b12x` NVFP4 MoE weight execution;
- ARM64 and SM121 source-build support.

NVFP4 MoE **weights** and NVFP4 MLA **KV cache** are separate features. This
runtime uses native NVFP4 weight kernels where vLLM selects them, while the KV
cache remains the supported FP8 DS-MLA layout.

## True packed NVFP4 research

Two downstream implementations establish the viable architecture:

1. `kacper-daftcode/vLLM-Moet` implements a 352-byte record for GLM-5.2:
   256 bytes of packed E2M1 latent data, 32 E4M3 group scales, and 64 E4M3
   RoPE bytes. It patches FlashInfer's SM120 sparse reader to expand the packed
   latent before the existing FP8 MMA path. Its published validation is on
   RTX PRO 6000/SM120 and GLM-5.2, not GB10 or DeepSeek V4.
2. `local-inference-lab/b12x` carries the newer 368-byte reader/writer design:
   256 packed latent bytes, 32 group scales, a 4-byte per-token RoPE scale,
   12 bytes of scale/padding space, and 64 FP8 RoPE bytes. It also supports an
   inline per-token second-level latent scale, avoiding a static calibration
   file for small-magnitude tokens.

The research commits are pinned in `runtime/upstream.lock`. They are references,
not build inputs, because neither implementation is a reviewed vLLM 0.27.1
DeepSeek V4 integration.

## Landing criteria

A true `nvfp4_ds_mla` lane must land as one atomic feature. Adding only a dtype
enum or mapping it to vLLM's generic `KVQuantMode.NVFP4` is invalid.

Required implementation pieces:

- one versioned record definition and one source of truth for every offset;
- a CUDA or CuTeDSL writer for the 512-dimension latent and 64-dimension RoPE;
- matching SM121 sparse decode and prefill readers;
- matching sliding-window cache handling for DeepSeek V4's hybrid layers;
- DSpark non-causal draft-block attention support;
- CUDA graph capture and padded `slot_mapping` behavior;
- cache sizing based on the packed record, including alignment and page size;
- explicit backend rejection on unsupported GPUs, models, and block sizes.

Required GB10/TP=2 validation:

- writer dequantization parity against a BF16 reference, including zeros,
  subnormals, outliers, and padded slots;
- sparse decode, chunked prefill, SWA, compressed C128, and DSpark tests;
- deterministic short prompts and arithmetic/tool-call suites;
- 8K, 32K, 128K, and near-1M multi-needle retrieval;
- concurrent identical-prefix requests with the ghost-block guard enabled;
- DSpark acceptance by draft position, not only aggregate throughput;
- KV bytes/token, pool capacity, TTFT, ITL, and decode throughput A/B against
  `fp8_ds_mla` on the same image and checkpoint revision.

Until those gates pass, `fp8_ds_mla` is the supported serving path and packed
NVFP4 KV remains experimental research.
