# DGX Spark DeepSeek V4 Flash Handoff

Status date: 2026-08-13

This document is the detailed handoff for the vLLM 0.27.1 GB10/DGX Spark
recipe, runtime patches, cluster control plane, and qualification work. A
future agent should read it before changing the image, cache dtype, patch
series, or live profile.

## Current State

The target is `deepseek-ai/DeepSeek-V4-Flash-0731` on two DGX Spark GB10
nodes, tensor parallelism 2, with DSpark speculative decoding (K=5). The rc7
image remains active. Ops envelope is **400k / seqs 6 / batch 8192 / GMU 0.84**
with an explicit `--kv-cache-memory=28235618304` (26.3 GiB) arena. Served
names are the historical pair so existing clients keep working:
`deepseek-ai/DeepSeek-V4-Flash-0731` and `deepseek-v4-flash-0731`.

| Item | Value |
|---|---|
| Active profile | `deepseek-v4-flash-0731-v0271-canary` |
| Candidate image | `dspark-vllm-gb10:v0.27.1-gb10-rc7` |
| vLLM source | tag `v0.27.1`, commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Image version report | `0.27.2.dev0+g6e448d0ea`; expected package metadata from the pinned source build |
| CUDA / Torch | CUDA 13.0.3 / Torch 2.13.0+cu130 |
| FlashInfer | 0.6.16.post3 plus a narrow upstream SM120 DSV4 dispatch overlay |
| MoE backend | DeepGEMM MXFP4 |
| KV cache dtype | `fp8_ds_mla` |
| Context / sequences / batched tokens | **400,000 / 6 / 8,192** |
| GPU memory utilization | 0.84 (still set; `--kv-cache-memory` overrides the leftover allocator) |
| KV arena | `--kv-cache-memory=28235618304` (26.3 GiB) |
| Live KV after that knob | **1,366,743 tokens · 3.42× @400k** (C3 full-window; not C4) |
| Default arena without the knob | 18.2 GiB / 944,766 tokens · 2.36× @400k |
| Served model names | `deepseek-ai/DeepSeek-V4-Flash-0731` and `deepseek-v4-flash-0731` |
| Global / SWA cache pages | 256 / 64 tokens |
| Rollback target after normal candidate activation | `deepseek-v4-flash-0731-dspark-1m-baseline` |

FlashInfer uses `SM120` in source and patch names for its kernel family. GB10
is SM121. The image is ARM64 and explicitly built for `12.1a` on both nodes.

Exact hosts, addresses, mounts, image digests, raw request artifacts, and
operation logs are intentionally excluded from Git. They are stored in
ignored `.private/` records on the cluster. Do not force-add them. This file
contains the reproducible public contract and sanitized measured results.

## Visual Evidence

The project README presents the current architecture, qualification sequence,
and long-context A/B results as repository-local SVGs. The asset registry in
[assets/README.md](assets/README.md) ties each graphic to its source table or
runtime contract. The performance graphic is a concise view of the long-matrix
results recorded in [Long matrix](#long-matrix); it is not a substitute for the
full raw comparison protocol.

## Important Boundaries

1. This is **not** a real packed NVFP4 DeepSeek V4 MLA KV cache. The
   supported candidate uses `fp8_ds_mla`. The old `nvfp4_ds_mla` spelling was
   a compatibility alias, not a native packed 4-bit sparse MLA implementation.
2. The A/B is controlled at the resource-envelope level but compares complete
   stacks, not one flag: vLLM, DeepGEMM, FlashInfer dispatch, runtime patches,
   and cache dtype differ. It proves the recipe outcome, not causation by an
   individual patch.
3. The candidate passed all completed gates but is not automatically qualified
   for a newer vLLM commit, model revision, cache dtype, or profile change.
4. Candidate and baseline report different KV-pool capacities at startup.
   Both passed a real 900K retrieval request. Treat the capacity difference as
   an investigation item, not a compression claim.

## Source of Truth

| Area | Location | Role |
|---|---|---|
| Locked inputs | [`runtime/upstream.lock`](../runtime/upstream.lock) | vLLM, dependency, architecture, and research pins. |
| Patch preparation | [`runtime/scripts/prepare-source.sh`](../runtime/scripts/prepare-source.sh) | Clean checkout, contract checks, ordered patch application. |
| Patch order | [`runtime/patches/vllm/series`](../runtime/patches/vllm/series) | Only supported order for patches 0001-0028. |
| FlashInfer delta | [`runtime/patches/flashinfer/0001-sm120-dsv4-192-256-topk.patch`](../runtime/patches/flashinfer/0001-sm120-dsv4-192-256-topk.patch) | Exact upstream-derived dispatch overlay. |
| Image build | [`build-dspark-vllm-runtime.sh`](../build-dspark-vllm-runtime.sh) | Guarded ARM64 image build. |
| Candidate env | [`cluster/environments/deepseek-v4-flash-0731-v0271-canary.env.example`](../cluster/environments/deepseek-v4-flash-0731-v0271-canary.env.example) | Tracked redacted runtime configuration. |
| Candidate profile | [`cluster/profiles/deepseek-v4-flash-0731-v0271-canary.conf`](../cluster/profiles/deepseek-v4-flash-0731-v0271-canary.conf) | Scripted candidate lifecycle. |
| Baseline profile | [`cluster/profiles/deepseek-v4-flash-0731-dspark-1m-baseline.conf`](../cluster/profiles/deepseek-v4-flash-0731-dspark-1m-baseline.conf) | Exact old-image 1M A/B control. |
| Control plane | [`cluster/vllm-switch`](../cluster/vllm-switch), [`cluster/vllm-profile-runner`](../cluster/vllm-profile-runner) | Profile state, validation, activation, rollback. |
| Qualification | [`cluster/scripts/run-qualification.sh`](../cluster/scripts/run-qualification.sh) | API, soak, retrieval, and basic performance suite. |
| A/B tools | [`scripts/ab-matrix-0731.py`](../scripts/ab-matrix-0731.py), [`scripts/compare-ab-matrix.py`](../scripts/compare-ab-matrix.py) | Fixture-locked latency and throughput comparison. |

The original upstream selection evidence and deferred alternatives are in
[UPSTREAM_AUDIT_2026-08-11.md](UPSTREAM_AUDIT_2026-08-11.md). The native
NVFP4 DS-MLA boundary and future implementation criteria are in
[NVFP4_DS_MLA.md](NVFP4_DS_MLA.md).

## Architecture Decisions

### Pinned source and dependencies

The source baseline is vLLM `v0.27.1` at the exact locked commit. The image
keeps vLLM's owned CUDA, Torch, Python, and FlashInfer package graph. It uses
the paired CUTLASS DSL 4.6.2 / QuACK 0.6.4 update and a DeepGEMM revision that
restores SM120/SM121 SITU scale handling for the official UE8M0 checkpoint.

Do not replace FlashInfer merely because a newer tag exists. This runtime
requires a particular upstream commit absent from release `v0.6.17`; upgrading
the package broadly would change an unqualified ABI/dependency graph.

### FP8 DS-MLA versus NVFP4 DS-MLA

DeepSeek V4 has a 448-dimension NoPE latent plus 64-dimension RoPE geometry.
The supported record is 584 bytes: 448 FP8 latent bytes, 128 BF16 RoPE bytes,
and 8 bytes of metadata. Generic `nvfp4` cache support is not a DeepSeek V4
sparse-MLA implementation. The SM120 sparse MLA backend requires
`fp8_ds_mla` for this model.

The Moet and b12x projects were studied as packed-cache research references.
Their 352/368-byte records use a different 512+64 MLA geometry. They prove a
packed reader/writer can be viable, but their layout, writer, sparse reader,
page sizing, and quality results cannot be transplanted to DeepSeek V4.

A genuine NVFP4 lane requires a versioned 448+64 record definition; writer;
SM121 sparse decode and prefill readers; C128/SWA and DSpark non-causal
handling; CUDA-graph-safe padding; cache sizing; strict rejection of
unsupported paths; dequantization parity; and end-to-end quality/performance
validation. It is a separate feature project, not a flag change.

### Cache page geometry

Keep global `KV_BLOCK_SIZE=256`: compressed C128 pages need a nonzero storage
block. The independent sliding-window attention cache uses its own native
64-token pages. With DSpark K=5, the non-causal sparse path has 133 active
SWA entries and vLLM rounds the relevant index tensor to 256. That exact shape
exposed the FlashInfer dispatch gap fixed by this recipe.

### MoE and speculation

The candidate selects `deep_gemm` for DeepSeek V4 MXFP4 experts. The old
`flashinfer_b12x` backend name is not accepted by the pinned vLLM MXFP4 oracle
and is not a viable candidate setting. DSpark remains K=5.

## Custom Work

### Native FlashInfer DSV4 top-k 192/256 dispatch

**Problem:** stock FlashInfer 0.6.16.post3 provides direct SM120 DSV4 decode
for top-k 128, 512, and 1024. The DSpark shape requires 256. Falling through
to a prefill-only path for a small decode batch fails during mHC warmup.

**Implementation:** retain FlashInfer 0.6.16.post3 and overlay only the
checksummed upstream FlashInfer PR #4380 commit
`24d7dfb2639083c5a4d418881099421fc800b7bb`. It adds 192/256 DSV4 decode and
prefill instantiations, matching Python dispatch, and a fail-loud unsupported
small-decode guard. The image deletes only the stale `sparse_mla_sm120` AOT
module; the patched module JIT-builds into an isolated mounted workspace.

**Evidence:** cold-workspace rc4 through rc7 booted on both ranks, completed
mHC and DSV4 autotune, and exposed H32/top-k 192/256 dispatch. Qualification
and the 900K proof had no sparse-MLA `num_tokens` error or fallback failure.

### Direct small-row sparse-MLA runner

The public FlashInfer wrapper omits split-K `mid_out`/`mid_lse` scratch needed
by the <=64-row SM120 decode kernel. Patch 0027 constructs the low-level
runner once per attention layer, reserves graph-stable workspace through the
vLLM workspace manager, and calls it directly for normal decode, short
prefill chunks, and mHC warmup. It preserves FP8 DS-MLA cache/index metadata
and makes C128A sparse-index tensors contiguous before use.

Patch 0027 and the FlashInfer 192/256 overlay are a pair: the vLLM patch
provides the routing and scratch buffers; the FlashInfer patch provides the
native kernel shape.

### Routed-expert token counter warmup

The one investigated non-fatal qualification warning was Triton's
`_count_expert_num_tokens`. It belongs to DeepGEMM routed experts, not
FlashInfer sparse MLA, DS-MLA KV, NCCL, or model correctness.

rc5 first added a narrow warmup, but a log call passed a mutable list to
`logger.info_once`; that produced a deterministic startup `TypeError` and the
legacy service was restored. rc6 fixed the logging issue and warmed each
power-of-two flattened router-ID block size. Its first real 32K top-k-6
request still JITed because Triton distinguished an unaligned 512-element
specialization from one carrying a 16-divisible runtime-scalar descriptor.

The final patch 0028:

- discovers loaded `RoutedExperts` and targets only `DeepGemmExperts` /
  `DeepGemmFP4Experts`;
- keeps the real tensor-parallel expert map when it exists;
- covers flattened blocks 8 through 1024;
- calculates aligned variants with `16 / gcd(top_k, 16)`;
- for deployed top-k 6, uses `(1, 2, 3, 6, 8, 11, 16, 22, 24, 43, 48, 86, 88)`
  token counts;
- covers aligned specializations for blocks 64, 128, 256, 512, and 1024;
- adds a focused upstream-tree unit test.

rc7 started from a fresh Triton cache, logged that full ladder on both ranks,
and did not emit the counter warning during qualification, 23 rounds of 2x32K
soak, either A/B matrix, or the 900K proof. vLLM main was inspected and still
did not warm this counter; its generic Triton helper supports the conclusion
that runtime-specialization variants must be modeled explicitly.

### GB10 build-path fixes

The build recipe includes focused changes that do not alter model math:

- CUDA-only FlashAttention submodule selection avoids irrelevant ROCm AITER
  and Composable Kernel downloads.
- FlashAttention-3 is not compiled when no Hopper target is selected, while an
  empty target remains for vLLM's packaging contract.
- The pinned Triton kernels source is shallow-cloned instead of fetching full
  history.
- `MAX_JOBS=16` and `NVCC_THREADS=8` give GB10 a useful CUDA compile pool.

Early builds spent most wall time on unnecessary ROCm source retrieval,
Hopper-only compilation, or a full Triton clone. The final recipe built from
the locked source and retained its Docker layers for controlled fast rebuilds.
Preserve the build cache unless a documented disk-space policy requires its
removal; pruning it forces a long rebuild without improving the image.

## Complete vLLM Patch Series

Every patch below applies cleanly, in the listed order, to the pinned source.
Do not reorder them or apply them to a newer vLLM revision without an upstream
audit and repeat hardware qualification.

| Patch(es) | Purpose | Evidence / provenance |
|---|---|---|
| 0001 | DeepSeek V4 0731 reasoning-effort prompt mappings. | Upstream frontend behavior; reasoning low/high/max API checks passed. |
| 0002 | Fix DeepSeek V4/3.2 tokenizer vocabulary overcount. | Upstream bug fix. |
| 0003 | Allow DSpark warmup without a sparse-index buffer. | Upstream cold-start correctness fix. |
| 0004 | Narrow DeepSeek V4 eager CUDA graph region. | Upstream performance/correctness boundary. |
| 0005-0008 | Guard same-step prefix hits on not-yet-written blocks across all cache groups. | Port of vLLM PR #42359 with DeepSeek V4 SM121 coverage. Recipe sets `VLLM_ALLOW_SPEC_DEC_SAME_STEP_PREFIX_HIT=2`. |
| 0009 | Restore DeepGEMM SM120 SITU support. | vLLM PR #50796 line; official UE8M0 checkpoint loaded on both ranks. |
| 0010 | Make C128A decode top-k row stride capture-stable. | vLLM PR #51318 line; long-context/concurrency correctness fix. |
| 0011 | Make speculative rejection argmax NaN-safe. | Upstream post-release correctness fix. |
| 0012 | Fix packed-KV physical zeroing stride. | Upstream post-release correctness fix. |
| 0013 | Invalidate layout-derived drafter metadata when layout changes. | Focused maintained SM121 preview fix. |
| 0014 | Clamp tile-local argmax IDs to vocabulary bounds. | Focused maintained SM121 preview fix. |
| 0015 | Make DSv4 mHC TileLang warmup execute on CUDA. | Focused SM121 preview fix; passed on hardware. |
| 0016 | Fix DSpark parallel token-ID initialization. | Upstream speculative-decoding correction. |
| 0017 | Reserve MRV2 speculative lookahead blocks in warmup. | Upstream warmup/accounting correction. |
| 0018 | Guard sparse MLA masked-MHA workspace allocation. | Upstream fallback-preserving attention fix. |
| 0019 | Remove sparse-MLA index-remap atomic contention. | Upstream performance change with unchanged math. |
| 0020 | Schedule MLA chunked context per request. | Upstream attention scheduling improvement. |
| 0021 | Use adaptive speculative scheduled-token budget. | Upstream performance work; DSpark acceptance telemetry still remains to be added. |
| 0022 | Pair CUTLASS DSL 4.6.2 and QuACK 0.6.4. | vLLM PR #51566 dependency pair. |
| 0023 | Align DeepSeek V4 parser thinking default. | Upstream parser fix; reasoning modes passed. |
| 0024 | Restrict FlashAttention submodules for CUDA builds. | Downstream GB10 build minimization. |
| 0025 | Skip actual FlashAttention-3 Hopper compile but preserve package target. | Downstream GB10 build repair. |
| 0026 | Shallow-clone Triton kernels source. | Downstream GB10 build repair. |
| 0027 | Direct SM120 DeepSeek V4 sparse-MLA small decode with stable scratch. | Custom patch; paired with FlashInfer overlay and hardware-qualified. |
| 0028 | Warm DeepGEMM routed-expert token counters. | Custom patch with unit test; rc7 removed observed serving-time warning. |

## Control Plane and Cluster Operations

`vllm-switch` is the repository-owned control plane. It supports the historic
generic launcher and the specialized scripted DeepSeek V4 service; the
candidate uses the scripted profile because it needs explicit two-rank start,
tracking, stop, readiness, and failure-capture commands.

Key behavior:

- active profile identity is stored in `.active-profile`, not inferred from a
  brittle systemd command string;
- `list` does not source profiles;
- `status` combines service state with an API probe;
- `render` is non-mutating;
- `validate` is non-mutating and checks both nodes' model paths, mounts, and
  identical image IDs;
- `install-control-plane.sh --install` backs up installed switch/profile files
  but does not stop or reload the active service;
- the managed systemd drop-in clears historical duplicate `ExecStartPre`
  entries before the scripted profile supplies its one pre-start hook;
- failure state is captured before rollback and failed containers are cleaned
  up before a rollback profile starts.

Automatic rollback normally uses the profile recorded as active immediately
before the switch, not merely the candidate profile's static fallback. The A/B
transition was deliberately:

```text
legacy 393K -> Stage-C 1M baseline -> v0.27.1 rc7 candidate
```

That ordering means a failed candidate restores the configuration-equivalent
1M baseline, not a mismatched legacy 393K profile. Maintain this property for
any comparable test.

### Operator commands

Substitute private connection details outside tracked documentation.

```bash
# Stage only: no profile installation and no service change.
./cluster/deploy-to-sparks.sh --apply

# Install changed control-plane files: no service restart/reload.
./cluster/install-control-plane.sh --install

# Preflight before activation: no service change.
vllm-switch validate deepseek-v4-flash-0731-v0271-canary

# Inspect active profile and API state.
vllm-switch status

# Use only in an approved maintenance window.
vllm-switch deepseek-v4-flash-0731-v0271-canary

# Explicit recovery command.
vllm-switch rollback
```

Long builds and model startups must be detached, privately logged operations
that are polled until final readiness or rollback. A short foreground SSH
wrapper can exit before a long switch completes. Never interpret caller
disconnect as service health; check `vllm-switch status`, API readiness, and
the private operation log.

### Private operational reporting

Ignored `.private/CLUSTER_IMPLEMENTATION_LOG.md` is the detailed cluster
chronology. It is synchronized between nodes and is deliberately untracked.
After each mutation append UTC time, authorized intent, affected
files/images/profiles/services, validation result, active/rollback profile,
and private artifact location. Never copy addresses, node IDs, credentials,
or raw prompt data into Git.

## Configurations Tested

### Final controlled A/B

| Field | Stage-C baseline | vLLM 0.27.1 rc7 candidate |
|---|---|---|
| Profile | `deepseek-v4-flash-0731-dspark-1m-baseline` | `deepseek-v4-flash-0731-v0271-canary` |
| Image | `vllm-dspark-runtime:dspark-nvfp4-stage-c` | `dspark-vllm-gb10:v0.27.1-gb10-rc7` |
| KV path | Legacy `nvfp4_ds_mla` compatibility spelling | Supported `fp8_ds_mla` sparse-MLA path |
| Context | 1,048,576 | 1,048,576 |
| Max sequences | 6 | 6 |
| Max batched tokens | 8,192 | 8,192 |
| GPU memory utilization | 0.84 | 0.84 |
| DSpark K | 5 | 5 |
| Tensor parallelism | 2 | 2 |
| Checkpoint | Same DeepSeek V4 Flash 0731 checkpoint | Same checkpoint |

The candidate first booted and was debugged using a stabilization envelope of
393,216 context, six sequences, 4,096 batched tokens, and 0.78 GPU utilization.
It passed final cold-cache router qualification there before the 1M profile was
activated. Do not conflate those earlier results with the final 1M A/B.

### Cold-cache policy

The candidate selects dedicated FlashInfer and Triton cache directories.
Meaningful dispatch/warmup tests use a new cache so old artifacts cannot mask a
missing compile or warmup. These are private deployment locations. Change them
for a new cold-start experiment; do not remove them casually from a live
profile.

## Verification Completed

### Source and image contracts

Before deployment, clean pinned-source preparation passed all 28 patches,
FlashInfer overlay checksum/application, Python compilation, shell syntax,
runtime contract tests, control-plane tests, whitespace checks, and Compose
rendering. The development workstation cannot establish ARM64, SM121, TP=2,
checkpoint-load, CUDA graph, or latency behavior; those gates were then run on
the real two-node GB10 deployment.

Both nodes completed checkpoint manifest/hash verification while inactive,
guarded ARM64 build, image transfer, exact image-ID parity, non-mutating
profile validation, full model load, DSpark drafter init, DeepGEMM FP4
experts, mHC CUDA warmup, native DSV4 autotune, and H32/top-k 192/256 dispatch
confirmation.

### Candidate progression

| Build | Result | Learning |
|---|---|---|
| rc4 | Passed two-rank boot and quick qualification. | The FlashInfer overlay plus direct small-row runner fixes the real DSpark sparse-MLA startup gap. |
| rc5 | Failed before readiness, then legacy restored. | Patch logging bug: mutable list supplied to `logger.info_once`; not a CUDA, FlashInfer, transport, or model defect. |
| rc6 | Passed quick qualification but first real 32K request emitted targeted counter JIT. | Counter block size alone misses Triton 16-divisibility specializations. |
| rc7 | Passed fresh-cache boot, qualification, A/B, and 900K proof. | Warm unaligned and attainable aligned top-k-6 variants. Active candidate. |

At rc7 readiness, three separate speculative helper kernels may JIT:
`_compute_local_logits_stats_kernel`, `_rejection_kernel`, and
`_resample_kernel`. They are non-fatal and are not the routed-expert counter
event. They are an optional next optimization.

### Final rc7 quick qualification

| Gate | Result |
|---|---|
| API readiness/model probe | Passed |
| Six-way concurrent smoke | 6/6 passed |
| Completion and streaming | Passed |
| Reasoning low/high/max | Passed |
| Structured output | Passed |
| Tool call and tool replay | Passed |
| Cancellation recovery | Passed |
| Context ladder | 8K and 32K passed |
| Two-user 32K soak | 23 successful rounds over about 12.4 minutes |
| Deterministic retrieval | 8K and 32K passed |
| Targeted `_count_expert_num_tokens` JIT | Not observed |

## Controlled A/B Method

The benchmark was deliberately extended to include latency, not merely output
throughput. `ab-matrix-0731.py` creates an immutable fixture on the first arm
and verifies prompt hashes when it is reused on the second. Every warmup and
measured request has a unique first cache block, preventing prefix-cache reuse
from contaminating prefill timing. Warmups are excluded from measured output.

Both arms use:

- streaming Chat Completions;
- `thinking: false`;
- temperature 0 and `ignore_eos: true`;
- a fixed 256-token completion limit;
- the same checkpoint and byte-identical prompt fixture.

The comparison tool rejects differences in fixture digest, trial count,
completion limit, or case set. Each case records median/P95 where relevant for
prefill tokens/second, aggregate and per-request output tokens/second, TTFT,
end-to-end elapsed time, and post-first-token decode milliseconds/token.

Positive changes below are favorable candidate movement: throughput is higher
and latency is lower.

| Matrix | Prompt targets | Concurrency | Trials per case | Completion limit |
|---|---:|---:|---:|---:|
| Short | 256, 8,192, 32,768 | 1, 2, 4, 6 | 5 | 256 |
| Long | 131,072, 300,000 | 1, 2 | 3 | 256 |
| Near-1M retrieval | 900,000 target, actual 900,073 | 1 | 1 deterministic proof | 32 |

## A/B Results

### Short matrix

The 256-token workload is mixed and must not be averaged away. The candidate
has a large single-request win, a small regression at concurrency 2, mixed
tail behavior at concurrency 4, and a regression at concurrency 6. Longer
short-context workloads show consistently favorable prefill and end-to-end
behavior at useful concurrency.

| Prompt / concurrency | Candidate change from baseline |
|---|---|
| 256 / 1 | Aggregate output +61.2%, per-request output +66.5%, median decode latency +39.9%, median E2E +38.0%; median TTFT -10.3%. |
| 256 / 2 | Broad small regression: aggregate output -3.2%, median TTFT -5.7%, P95 TTFT -24.6%, median E2E -2.2%. |
| 256 / 4 | Mixed: prefill +14.3%, per-request output +13.6%, median TTFT +12.5%, median E2E +11.0%; P95 E2E -17.9% and P95 decode latency -17.3%. |
| 256 / 6 | Aggregate output -13.0%, per-request output -16.4%, median E2E -18.1%, median decode latency -19.6%; TTFT is roughly flat/slightly favorable. |
| 8,192 / 1 | Prefill +10.2%, aggregate output +4.9%, median TTFT +9.2%, median E2E +4.7%; per-request output and decode latency about -6%. |
| 8,192 / 2 | Prefill +12.5%, aggregate output +9.1%, median TTFT +11.1%, median E2E +7.6%; P95 decode latency +39.6%. |
| 8,192 / 4 | Prefill +12.1%, aggregate output +11.9%, per-request output +28.9%, median TTFT +11.0%, median E2E +8.4%, P95 E2E +36.4%. |
| 8,192 / 6 | Prefill +12.1%, aggregate output +34.9%, per-request output +17.5%, median TTFT +11.0%, median E2E +6.8%, P95 E2E +16.1%. |
| 32,768 / 1 | Prefill +13.1%, aggregate output +9.9%, median TTFT +11.6%, median E2E +9.1%; per-request output -4.0%, median decode latency -4.2%. |
| 32,768 / 2 | Prefill +13.4%, aggregate output +9.6%, median TTFT +11.4%, median E2E +9.2%, median decode latency +4.4%. |
| 32,768 / 4 | Prefill +12.7%, aggregate output +10.6%, per-request output +7.2%, median TTFT +11.2%, median E2E +9.9%, P95 decode latency +10.3%. |
| 32,768 / 6 | Prefill +13.0%, aggregate output +10.7%, per-request output +8.8%, median TTFT +11.4%, median E2E +9.9%, P95 decode latency +10.1%. |

Interpretation: rc7 materially improves prefill and end-to-end latency from
8K onward. It is not an unconditional high-concurrency short-decode win; that
traffic class needs replication before making a product claim.

### Long matrix

This is the strongest performance evidence. Every long case improved prefill,
TTFT, and end-to-end latency. At concurrency 2, decode latency also improved.
At concurrency 1, decode throughput/latency was slightly worse even though
total request latency improved because prefill was substantially faster.

| Prompt / concurrency | Prefill tok/s | Aggregate output tok/s | Median TTFT | Median E2E | Median decode latency |
|---|---:|---:|---:|---:|---:|
| 131,072 / 1 | +9.6% (1674 -> 1835) | +8.8% (2.80 -> 3.04) | +8.8% (88.10s -> 80.39s) | +8.1% (91.57s -> 84.16s) | -6.9% (13.61 -> 14.56 ms/token) |
| 131,072 / 2 | +11.2% (1203 -> 1338) | +11.5% (2.80 -> 3.12) | +10.4% (135.85s -> 121.70s) | +10.3% (182.18s -> 163.37s) | +10.1% (180.57 -> 162.41 ms/token) |
| 300,000 / 1 | +9.0% (1414 -> 1542) | +8.9% (1.06 -> 1.15) | +8.3% (238.67s -> 218.87s) | +8.2% (242.39s -> 222.62s) | -1.8% (14.36 -> 14.63 ms/token) |
| 300,000 / 2 | +9.7% (1046 -> 1148) | +9.9% (1.06 -> 1.16) | +9.1% (361.82s -> 328.99s) | +9.1% (483.12s -> 438.95s) | +9.7% (473.81 -> 427.97 ms/token) |

P95 results have the same favorable long-prefill and E2E direction:

- 131K/1: TTFT +9.0%, E2E +8.5%;
- 131K/2: TTFT +10.6%, E2E +10.2%, decode latency +10.8%;
- 300K/1: TTFT +8.1%, E2E +8.0%;
- 300K/2: TTFT +9.1%, E2E +9.0%, decode latency +9.7%.

### Near-1M retrieval

Both arms accepted the same actual 900,073-token request and returned all
three ordered needles exactly. This is a real sparse-MLA long-context proof,
not a capacity estimate.

| Arm | Result | End-to-end elapsed |
|---|---|---:|
| Stage-C baseline | Passed ordered three-needle retrieval | 923.38s |
| rc7 candidate | Passed identical ordered retrieval | 886.19s |
| Candidate change | Passed | +4.0% lower elapsed latency |

No candidate router-counter JIT warning appeared in the short matrix, long
matrix, or near-1M proof.

## Capacity and Cache Interpretation

At the identical requested 1M envelope, startup logs reported about 2,576,691
KV tokens for Stage-C and 1,846,731 for rc7. This is an allocator/runtime
report, not proof that a particular dtype mathematically uses more or less
memory. The services differ in version, allocator, graph reservation, memory
accounting, and cache-format behavior. Long concurrent telemetry also showed
different utilization patterns, but the counters are not guaranteed to have
the same semantics across runtimes.

Proven facts:

- both profiles started with a one-million-token maximum context;
- both completed an actual 900,073-token retrieval request;
- the candidate completed the matched long A/B with favorable prefill and
  end-to-end latency.

Unproven facts:

- maximum simultaneous 1M-request capacity;
- bytes/token and exact allocation overhead for either runtime;
- whether an explicit candidate KV-cache memory configuration safely raises
  capacity without destabilizing graphs or hurting throughput.

Investigate this before claiming a capacity regression or compression gain.

## Remaining Work and Recommended Sequence

1. **Preserve the current good state.** Capture status, active/rollback
   profile, image parity, and private artifact locations before any change.
   Do not delete the cold-cache evidence for rc7.
2. **Explain candidate cache capacity.** Collect both runtimes' GPU memory
   profile, cache configuration, graph reservation, bytes/block, block count,
   and available memory at exactly the same envelope. Test only one valid
   allocator adjustment at a time, then rerun 32K, 300K/2, and 900K gates.
3. **Optionally warm the remaining speculative helpers.** Use a fresh cache to
   capture exact specialization keys for the three known helpers. Prefer an
   existing upstream generic warmup mechanism if it can express them. Add a
   narrow test/patch only after proving the required variants.
4. **Run an extended rc7 full soak.** Use the full qualification mode at the
   final 1M profile: longer context, concurrency, cancellation/recovery, and
   explicit error/JIT log scanning. Keep raw output private.
5. **Measure DSpark efficiency.** Add draft-position acceptance rate, queue
   time, cache utilization, and GPU memory telemetry to the A/B harness.
   Aggregate output throughput alone cannot explain speculative efficiency.
6. **Repeat representative cases.** Repeat 256/6, 8K/6, 32K/6, 131K/2, and
   300K/2 in another window and report medians, P95, and variation. This will
   establish whether short-prompt regressions are stable.
7. **Keep real NVFP4 DS-MLA separate.** Begin on a feature branch with
   record-layout and writer/reader parity tests for DeepSeek V4 448+64 before
   modifying this serving lane.
8. **Treat a newer upstream as a new RC.** Pin it, repeat source contracts,
   rebuild, cold-cache boot, full qualification, and fixture-locked A/B. Do
   not broadly cherry-pick newer changes into rc7 in place.

## Reproduction and Release Gates

### Source preparation and candidate qualification

```bash
# From repository root. Uses upstream.lock and ordered vLLM patch series.
runtime/scripts/prepare-source.sh

# The build helper has service-state and free-space guards.
./build-dspark-vllm-runtime.sh

# Before any activation, perform the non-mutating profile preflight.
vllm-switch validate deepseek-v4-flash-0731-v0271-canary

# After approved activation, write results only under ignored storage.
OUTPUT_ROOT="$PWD/.private/qualification" \
  cluster/scripts/run-qualification.sh --run \
  --label rc-next --model deepseek-v4-flash-0731-v0271-canary --mode full
```

Require two-node image parity and model-path checks; cold-cache full load;
mHC and native DSV4 dispatch confirmation; API contracts; context/concurrency
and retrieval gates; a soak; latency-aware compatible A/B; rollback evidence;
and an updated private implementation log before accepting a new image.

### Compatible A/B protocol

Create a fixture only on the baseline. Reuse it without alteration on the
candidate. The comparison program rejects incompatible reports.

```bash
python3 scripts/ab-matrix-0731.py \
  --base-url "$BASE_URL" --model "$BASELINE_MODEL" \
  --prompt-lengths 256,8192,32768 --concurrency 1,2,4,6 --trials 5 \
  --fixture "$RUN_DIR/short-prompts.json" --output "$RUN_DIR/baseline-short.json"

python3 scripts/ab-matrix-0731.py \
  --base-url "$BASE_URL" --model "$CANDIDATE_MODEL" \
  --prompt-lengths 256,8192,32768 --concurrency 1,2,4,6 --trials 5 \
  --fixture "$RUN_DIR/short-prompts.json" --output "$RUN_DIR/candidate-short.json"

python3 scripts/compare-ab-matrix.py \
  "$RUN_DIR/baseline-short.json" "$RUN_DIR/candidate-short.json" \
  --output "$RUN_DIR/short-comparison.md"
```

Use a separate immutable fixture for long context. Never compare arms with
different output limits, temperatures, trial counts, prompt fixtures, or
prefix-cache behavior.

## Next-Agent Checklist

- [ ] Read this document, [RUNTIME_V0271_GB10.md](RUNTIME_V0271_GB10.md),
  [NVFP4_DS_MLA.md](NVFP4_DS_MLA.md), and the upstream audit.
- [ ] Confirm service state with `vllm-switch status` without restarting it.
- [ ] Read the ignored private implementation log for exact cluster state;
  never commit it.
- [ ] Preserve lock-file pins and patch order while changing one concern at a
  time.
- [ ] Run `vllm-switch validate` before every switch.
- [ ] Activate a compatible baseline immediately before a candidate A/B
  switch so recorded-current rollback is meaningful.
- [ ] Treat FP8 DS-MLA as supported and true packed NVFP4 DS-MLA as research.
- [ ] Always report latency as well as throughput.
- [ ] Keep private cluster topology, raw requests, logs, and addresses out of
  tracked files.

## Conclusion

rc7 is a working vLLM 0.27.1 GB10 serving lane for DeepSeek V4 Flash 0731. It
fixes the real DSV4 top-k 256 dispatch gap, supplies graph-stable small-row
sparse-MLA scratch, carries the needed post-release DeepSeek V4/GB10 fixes,
and eliminates the observed DeepGEMM routed-expert counter JIT under the
exercised workload.

At the controlled one-million-token envelope it passed equivalent 900K
retrieval and improved long-context prefill, TTFT, and end-to-end latency by
roughly 8-11% at 131K/300K. The remaining work is tightly scoped: explain KV
pool capacity, optionally warm the remaining speculative helper kernels,
extend soak/telemetry, and keep true NVFP4 DS-MLA as a separate rigorous
implementation effort.
