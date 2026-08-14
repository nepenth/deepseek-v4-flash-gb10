# Changelog

This changelog begins with the independent GB10 runtime project. Historical
upstream lineage and prior recipe context are retained through attribution and
the detailed handoff, not as a claim that old profiles are current support.

## 0.1.1 - 2026-08-14

### Changed

- Live winner is rc7 Arm B: skip the `#31` CPU thinking-budget hook, default
  thinking off, apply the 0.27.1 suppress-stops rewrite, keep `max_model_len=1M`.
- Repository is the source of truth for campaign state; KB is the live mirror.

### Validated

- Exact 32k×6 on fresh 1M: 46.36 / 46.38 / 48.48 tok/s, 1.05×, MTP 95.1%.
- Phase 5 11/11, official encoding 4/4, 3× restart greedy identical, 1k soak
  1000/1000, 1.04M three-needle retrieval.

### Fixed

- `#31` host-scan hook measured at ~1.8× decode tax at seqs=6 and removed.
- Documented that same-profile `vllm-switch` does not recreate for env-only
  changes; force-remove both rank containers.

## 0.1.0 - 2026-08-12

### Added

- Pinned vLLM 0.27.1 GB10/SM121 build contract with an ordered 28-patch series.
- Two-node scripted `vllm-switch` control plane with preflight validation,
  active-profile tracking, state capture, and comparable-profile rollback.
- Controlled 1M Stage-C baseline profile and fixture-locked A/B tooling with
  TTFT, end-to-end latency, decode latency, prefill, and output metrics.
- Full GB10 handoff documenting source selection, patches, test results,
  operational constraints, and next execution work.
- Repository-local SVG architecture, qualification, and performance evidence
  diagrams.

### Changed

- Selected `fp8_ds_mla` as the supported DeepSeek V4 sparse-MLA KV cache path.
  `nvfp4_ds_mla` is documented as a historical compatibility spelling, not a
  native packed NVFP4 implementation.
- Standardized candidate serving on DeepGEMM MXFP4 experts, TP=2, DSpark K=5,
  global 256-token C128 pages, and native 64-token SWA pages.

### Fixed

- Added the exact upstream FlashInfer DSV4 SM120 192/256 dispatch delta needed
  by DeepSeek V4 DSpark sparse-MLA decode on GB10.
- Added direct small-row sparse-MLA runner scratch handling for <=64-row
  decode/prefill and mHC warmup.
- Added targeted DeepGEMM routed-expert counter warmup covering Triton runtime
  scalar divisibility variants; rc7 eliminated the observed under-load JIT
  event in qualification and A/B traffic.

### Validated

- Two-node ARM64 image build, image parity, cold-cache model startup, API
  contracts, 23-round 2x32K soak, deterministic retrieval, and 900K
  three-needle proof.
- Controlled 1M long-context A/B: rc7 improved prefill, TTFT, and end-to-end
  latency by approximately 8-11% at 131K and 300K prompts.
