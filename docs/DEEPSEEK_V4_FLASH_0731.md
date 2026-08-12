# DeepSeek V4 Flash 0731 Serving Contract

## Target Model

- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Pinned checkpoint revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Serving topology: two DGX Spark GB10 nodes, tensor parallelism 2
- Speculation: DSpark K=5
- Serving lane: text only

The model uses DeepSeek V4 sparse MLA and the DSpark speculative decoding
structure. Successful weight loading alone is not a serving qualification.
Validate chat encoding, reasoning controls, tool calls, cancellation recovery,
retrieval, and long-context stability after every runtime change.

## Current Candidate Configuration

The validated vLLM 0.27.1 rc7 candidate uses:

| Setting | Value |
|---|---|
| Max model length | 1,048,576 |
| Max sequences | 6 |
| Max batched tokens | 8,192 |
| GPU memory utilization | 0.84 |
| MoE backend | `deep_gemm` |
| KV cache dtype | `fp8_ds_mla` |
| Global C128 / SWA pages | 256 / 64 |
| Default thinking | off; request-level reasoning controls remain available |

The native sparse-MLA backend requires `fp8_ds_mla` for this model. Do not
substitute generic `nvfp4` or describe the legacy `nvfp4_ds_mla` spelling as a
real packed cache. The current reasoning and implementation boundary is in
[NVFP4_DS_MLA.md](NVFP4_DS_MLA.md).

## Validation Requirements

The required release gates are:

1. Cold-cache full checkpoint load and native sparse DSV4 dispatch on both
   ranks.
2. Completion, streaming, reasoning low/high/max, structured output, tool
   replay, and cancellation recovery.
3. Context ladder, concurrent smoke, soak, and deterministic multi-needle
   retrieval.
4. Fixture-locked A/B with TTFT, end-to-end latency, decode latency, prefill,
   and output throughput.

The current evidence is recorded in
[GB10_DSV4_HANDOFF_2026-08-12.md](GB10_DSV4_HANDOFF_2026-08-12.md). It
includes an identical 900K three-needle retrieval proof for baseline and
candidate and the final long-context A/B results.
