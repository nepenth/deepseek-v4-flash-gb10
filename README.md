# DeepSeek V4 Flash GB10 Runtime

An independently maintained, private-ready serving recipe for
`deepseek-ai/DeepSeek-V4-Flash-0731` on a two-node NVIDIA DGX Spark GB10
cluster. It builds a pinned vLLM runtime, applies focused SM121/DSpark
corrections, provides an operator-safe cluster control plane, and records a
reproducible qualification protocol.

<p align="center">
  <img src="docs/assets/gb10-runtime-architecture.svg" alt="Two-node DGX Spark runtime architecture" width="920">
</p>

## Current Result

The active release candidate is `dspark-vllm-gb10:v0.27.1-gb10-rc7`:

- vLLM `v0.27.1` pinned to commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`
- CUDA 13.0.3, Torch 2.13.0, FlashInfer 0.6.16.post3
- TP=2, DSpark K=5, DeepGEMM MXFP4 experts, `fp8_ds_mla` KV cache
- 1,048,576 max context, 6 sequences, 8,192 batched tokens, 0.84 GPU memory
  utilization
- Native FlashInfer DSV4 top-k 192/256 dispatch for the DSpark sparse-MLA
  shape on GB10
- Focused DeepGEMM router-counter warmup that prevents the prior under-load
  Triton JIT event

On the controlled 1M baseline comparison, rc7 passed identical 900K
three-needle retrieval and improved long-context prefill, TTFT, and
end-to-end latency by approximately 8-11% at 131K and 300K prompts.

<p align="center">
  <img src="docs/assets/long-context-ab.svg" alt="Long-context rc7 candidate improvements" width="920">
</p>

**2026-08-19 (live now):** server default `thinking=max` via the Mia #48 GPU
closer (no omit-field `DEFAULT_THINKING_TOKEN_BUDGET`). `#31` CPU host-scan
stays skipped. `#55` tool-truncation and `#52492` indexer capture guard are
on. Exact 32k×6 think-off on this bounce: **43.92 / 45.39 / 46.79 tok/s**,
1.07× spread, MTP 94.9%. Clients that must not think send
`chat_template_kwargs.thinking=false`. Earlier 2026-08-14 C1 ladder and
correctness stamps stay in
[docs/CAMPAIGN_2026-08-14.md](docs/CAMPAIGN_2026-08-14.md). Machine state:
[`project-status.json`](project-status.json).

The 2026-08-12 build/A/B history remains in
[the GB10 handoff](docs/GB10_DSV4_HANDOFF_2026-08-12.md).

## Design Principles

| Principle | How this repository applies it |
|---|---|
| Reproducibility | Exact sources and dependency/research references are in [`runtime/upstream.lock`](runtime/upstream.lock); the ordered vLLM series is checked before application. |
| Narrow customization | Carry focused hardware/model fixes, not a wholesale unpublished runtime fork. FlashInfer is overlaid only with the exact upstream DSV4 dispatch delta required by DSpark. |
| Operator safety | `vllm-switch` validates both ranks before activation, records the active profile, captures failure state, and rolls back to the previously active comparable profile. |
| Evidence over claims | Cold-cache startup, API contracts, soak, retrieval, and fixture-locked latency-aware A/B are all part of qualification. |
| Privacy by default | Node topology, credentials, raw prompts, logs, artifact paths, and operational history remain under ignored `.private/` storage. |

## Supported Runtime Contract

This is a text-serving lane for DeepSeek V4 Flash 0731 on paired GB10 nodes.
It uses `fp8_ds_mla` for DeepSeek V4 sparse MLA KV cache.

`nvfp4_ds_mla` must not be presented as native packed NVFP4 support. Previous
runtime paths used that spelling as an alias for an FP8-style record and kernel
path. A real DeepSeek V4 packed NVFP4 cache requires a separate 448+64 record,
writer, SM121 sparse readers, DSpark/hybrid-cache support, and quality gates.
See [NVFP4 DS-MLA status](docs/NVFP4_DS_MLA.md).

## Documentation

| Document | Use it for |
|---|---|
| [GB10 handoff](docs/GB10_DSV4_HANDOFF_2026-08-12.md) | Full implementation history, configuration, results, caveats, and next-agent plan. |
| [Runtime contract](docs/RUNTIME_V0271_GB10.md) | Pinned inputs, build rules, patch policy, and serving contract. |
| [Control plane](docs/CLUSTER_CONTROL_PLANE.md) | Deployment boundaries, `vllm-switch`, validation, and rollback behavior. |
| [Upstream audit](docs/UPSTREAM_AUDIT_2026-08-11.md) | Why each upstream/downstream change was selected or deferred. |
| [NVFP4 DS-MLA](docs/NVFP4_DS_MLA.md) | Current FP8 decision and requirements for true packed NVFP4 research. |
| [Documentation guide](docs/README.md) | Current documents versus retained historical references. |
| [Credits](CREDITS.md) | Attribution and license lineage for upstream foundations. |

## Build and Operate

Prepare and verify the exact patched source locally:

```bash
runtime/scripts/prepare-source.sh
```

Build only through the guarded helper. It verifies source inputs and protects
the cluster service lifecycle:

```bash
./build-dspark-vllm-runtime.sh
```

On the configured cluster, stage and install control-plane changes separately,
then validate before any maintenance-window activation:

```bash
./cluster/deploy-to-sparks.sh --apply
./cluster/install-control-plane.sh --install
vllm-switch validate deepseek-v4-flash-0731-v0271-canary
vllm-switch status
```

`deploy-to-sparks.sh --apply`, control-plane installation, validation, and
profile activation have intentionally separate responsibilities. Read the
[control-plane guide](docs/CLUSTER_CONTROL_PLANE.md) before switching a live
model. Keep actual host values in the deployed, ignored environment file; do
not commit them.

## Qualification

The qualification suite checks completion, streaming, reasoning modes,
structured output, tool replay, cancellation recovery, concurrent smoke,
context ladders, soak stability, retrieval, and performance smoke:

```bash
OUTPUT_ROOT="$PWD/.private/qualification" \
  cluster/scripts/run-qualification.sh --run \
  --label rc-next --model deepseek-v4-flash-0731-v0271-canary --mode full
```

For release comparisons, use [`scripts/ab-matrix-0731.py`](scripts/ab-matrix-0731.py)
and [`scripts/compare-ab-matrix.py`](scripts/compare-ab-matrix.py). They lock
prompt fixtures and report TTFT, end-to-end latency, decode latency, prefill,
and output throughput. Do not compare mismatched fixtures, completion limits,
trial counts, temperatures, or prefix-cache behavior.

<p align="center">
  <img src="docs/assets/qualification-gates.svg" alt="GB10 qualification gate sequence" width="920">
</p>

## Repository Layout

```text
runtime/                 Pinned vLLM source preparation, Docker build, patch series
cluster/                 Two-node deployment, profiles, vllm-switch, qualification runner
scripts/                 API, retrieval, benchmark, and fixture-locked A/B tools
docs/                    Runtime contract, audit, handoff, operational documentation
docs/assets/             Local SVG architecture, evidence, and qualification diagrams
```

## Attribution and Licensing

This repository is an independent project, not a GitHub fork. It retains
clear provenance and licenses for vLLM, FlashInfer, DeepGEMM, DSpark, NVIDIA
components, and prior public research. The project-specific credit and license
notes are in [CREDITS.md](CREDITS.md); downstream source files retain their
original license headers.

## Status and Next Work

The live winner is still `dspark-vllm-gb10:v0.27.1-gb10-rc7` at advertised
1M with the 2026-08-14 Arm B knobs. See
[PROJECT-DECISIONS.md](PROJECT-DECISIONS.md) and
[docs/CAMPAIGN_2026-08-14.md](docs/CAMPAIGN_2026-08-14.md).

Next engineering work is intentionally limited: A/B 0029 against this
winner before any rebuild, port Mia #48 only if `thinking=max` is required,
and treat packed NVFP4 DS-MLA as a separate research implementation. Do not
cut `max_model_len` to free RAM; shrink `KV_CACHE_MEMORY`.
