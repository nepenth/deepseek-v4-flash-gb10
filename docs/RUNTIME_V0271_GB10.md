# vLLM 0.27.1 GB10 runtime

## Inputs

The runtime is built from the exact commit in `runtime/upstream.lock`:

```text
vLLM v0.27.1
commit 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
CUDA 13.0.3
PyTorch 2.13.0
FlashInfer 0.6.16.post3
CUTLASS DSL 4.6.0
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

The broad v0.25 overlay under `runtime/overlay/` remains for history and
attribution. It is not copied into v0.27.1.

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
DSpark with five speculative tokens, `flashinfer_b12x` MoE, and
`fp8_ds_mla` KV.

The patch series adds `VLLM_ALLOW_SPEC_DEC_SAME_STEP_PREFIX_HIT`:

- `2`: guard all cache groups; recipe default;
- `1`: upstream PR #42359 semantics, gated to EAGLE-marked groups;
- `0`: disable for controlled diagnosis only.

The current image and patch series still require two-node DGX Spark hardware
validation before publication. The source and static contracts do not replace
the long-context, acceptance, and throughput gates in
`docs/NVFP4_DS_MLA.md`.
