# 2x DGX Spark execution plan

This plan starts with the audited vLLM 0.27.1 candidate and introduces one
uncertainty at a time. No image becomes the recipe default until it passes the
quality, stability, memory, and performance gates below.

## Phase 0: capture the cluster contract

Before building, record both nodes' hardware and software state:

- DGX OS, kernel, NVIDIA driver, CUDA toolkit, firmware, Docker, and Compose;
- GPU name, compute capability, total/available unified memory, and clocks;
- CX-7 firmware, link rate, MTU, RoCE GID, HCA/interface names, and topology;
- checkpoint file hashes and revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`;
- cold cache versus warmed compile/model caches.

Keep the output with each run. A result without this manifest is not comparable.

## Phase 1: build and boot qualification

Build the same source tree independently on both ARM64 nodes, then compare the
image digest. Preserve complete build logs and verify:

1. DeepGEMM checks out `2fd67329ec2942f65ba35d561256ab6ed3b903cb`.
2. Every CUDA/CuTeDSL/FlashInfer build target includes SM121/`12.1a`.
3. Runtime versions match `runtime/upstream.lock`; no pip resolver replaces a
   pinned package.
4. All 48 checkpoint shards load on TP=2 without scale-layout assertions,
   illegal memory access, mid-request JIT, or unsupported-kernel fallback.
5. The selected paths are DSpark K=5, `fp8_ds_mla`, native SM121 sparse MLA,
   a `KV_BLOCK_SIZE=256` global cache with zero-copy 64-token SWA views for
   FlashInfer small decode, and `deep_gemm` MXFP4 MoE for the v0.27.1-line
   candidate.

Build three images for diagnosis and attribution:

- `boot-minimal`: v0.27.1 plus only the DeepGEMM build fix;
- `candidate`: the complete 27-patch series, including the FlashInfer SM120
  small-batch sparse-MLA dispatch repair;
- `reference`: the previously working Anemll/NVIDIA-derived image, clearly
  labeled as a different vLLM/dependency baseline.

The unpatched v0.27.1 image is expected to fail checkpoint loading and is a
negative control, not a performance baseline.

## Phase 2: correctness and corruption gates

Use deterministic seeds and archive request/response JSON plus both-rank logs.

- Smoke: completion, streaming, low/high/max/no reasoning, DSML tool calls,
  multi-turn tool replay, and structured output.
- Quality: a frozen code/reasoning/tool suite plus GSM8K and a task-oriented
  natural-language set. Random-token prompts are not an acceptance-quality test.
- Context: 8K, 32K, 128K, 300K, 600K, and near 1M multi-needle retrieval.
- Concurrency: 1, 2, 4, and 6 requests, including staggered/ragged arrivals,
  identical prefix hits, cache eviction/reuse, and mixed prefill/decode.
- C128 regression: concurrent long streams with at least two decode rows per
  batch; scan every output for leaked special tokens and multi-script corruption.
- Stability: one-hour soak, repeated load/unload, request cancellation, OOM
  rejection, worker restart, and cache-pressure block zero/reuse.

Instrument non-finite logits/draft probabilities and token-ID bounds in the
diagnostic build. The release gate is zero NaN/Inf events, zero out-of-vocabulary
IDs, zero CUDA errors, zero engine deaths, and no unexplained quality regression.

## Phase 3: performance characterization

Warm every measured shape before recording it. Report median, p95, and p99 for
TTFT and ITL; aggregate and per-user output throughput; prefill throughput;
DSpark acceptance by draft position; KV capacity; peak memory; and compile time.

Run a controlled matrix over:

- DSpark K=1..5, with K=5 as the initial checkpoint-native default;
- concurrency 1/2/4/6 and short/32K/128K/300K/600K contexts;
- `max_num_batched_tokens` 4096/8192/16384 where memory permits;
- `gpu_memory_utilization` 0.80/0.82/0.835 with an explicit safety margin;
- full candidate versus candidate without adaptive budgeting;
- CUDA graphs versus eager only as a diagnostic control.

Use the same prompts, arrival schedule, output lengths, caches, and checkpoint
revision for every A/B. Select defaults from workload-weighted results, not the
best isolated throughput number.

## Phase 4: next implementation candidates

Promote only one candidate at a time behind a build patch or explicit flag:

1. **FlashInfer SM121 fail-loud gate.** Port the maintained branch's cubin,
   symbol, and architecture checks so an incompatible wheel fails at startup
   instead of silently selecting a fallback.
2. **Long-context gather kernel (vLLM PR #51739).** Test the FP8 upconvert path
   on GB10; promote only if end-to-end long-prefill gains justify the CUDA delta.
3. **TokenSpeed non-causal DSpark (PR #50911).** Verify SM121 kernel support,
   sparse-SWA semantics, quality, and acceptance before enabling selection.
4. **Fused MRV2 multi-step graphs (PR #46849).** Revisit only if profiling
   shows the GB10 workload is host-bound; upstream DSV4 results were mixed.
5. **Expert/index bounds.** Port the remaining maintained-branch defensive
   guards if instrumentation observes stale padded IDs, or before enabling EP.
6. **Remove `--trust-remote-code`.** Validate native tokenizer/model loading and
   tool parsing first; removal reduces checkpoint-side code execution risk.

Each item needs a separate commit, feature switch when practical, focused unit
test, candidate image, and candidate-off/candidate-on cluster A/B.

## Phase 5: true DeepSeek V4 packed NVFP4 KV

Treat this as a new backend, not a dtype alias. First freeze a versioned record
for the model's 448-dimension NoPE latent and 64-dimension RoPE data. Determine
encoding, scale hierarchy, offsets, alignment, and bytes/token from measured
error; do not inherit the unrelated 352/368-byte 512+64 layouts.

Implement in this order:

1. CPU reference pack/unpack and error/property tests.
2. SM121 writer with padded-slot and CUDA-graph correctness.
3. Sparse decode readers for normal and compressed C128 pages.
4. Chunked-prefill and sliding-window readers.
5. DSpark non-causal draft attention and cache-zero/reuse integration.
6. Cache sizing, backend rejection, metrics, and an explicit experimental flag.

Compare BF16, `fp8_ds_mla`, and packed NVFP4 on retrieval, perplexity/task
quality, DSpark acceptance by position, bytes/token, maximum context, TTFT,
ITL, and throughput. Promotion requires a meaningful capacity/performance gain
without a material quality loss; otherwise retain FP8 as the production lane.

## Release decision

Publish a candidate only when its image digest, source commit, patch-series
hashes, cluster manifest, full logs, correctness report, and benchmark JSON are
archived together. Tag the first fully passing image as a GB10 release candidate;
do not overwrite the v0.27.1 tag or inherit historical performance claims.
