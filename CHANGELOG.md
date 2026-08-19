# Changelog

This changelog begins with the independent GB10 runtime project. Historical
upstream lineage and prior recipe context are retained through attribution and
the detailed handoff, not as a claim that old profiles are current support.

## 0.1.3 - 2026-08-19

### Changed

- Server default is `DEFAULT_THINKING=max`. The `#31` CPU host-scan stays
  skipped. Clients that must not think send `chat_template_kwargs.thinking=false`.
- Start script copies every bind-mounted hotfix to the worker.

### Added

- `patches/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py` (Mia #48 port to
  vLLM 0.27.1, plus greedy logits-processing gate).
- `patches/hotfix-dsv4-issue55-tool-truncation.py`
- `patches/hotfix-dsv4-indexer-52492.py`

### Validated

- 32k×6 think-off: 43.92 / 45.39 / 46.79 tok/s, 1.07×, MTP 94.9%.
- 32k×4 think-off: 60.53 / 61.61 / 62.45 tok/s. Mia #90 not required.
- Off-think client still emits empty reasoning. Budget 8/32 closes.

## 0.1.2 - 2026-08-17

### Changed

- Synced the checked-in compose/env/start contract to the live rc7 Arm B + 1M
  winner. The previous env example still said `MAX_MODEL_LEN=400000` with the
  KV arena commented out.
- `docker-compose.dspark.yml` now mounts and applies the 0.27.1 suppress-stops
  rewrite, optionally applies the #31 hook, passes `--kv-cache-memory` and
  `--long-prefill-token-threshold 1024`, and defaults thinking off /
  `deep_gemm`.
- Start script now copies both runtime hotfixes to the worker.

### Added

- `patches/hotfix-dsv4-issue31-v0272.py` (present on the live cluster, missing
  from git). Winner keeps `DSPARK_SKIP_ISSUE31_HOTFIX=1`.
- `PROJECT-DECISIONS.md` for the winner knobs and rejected paths.

### Validated

- 2026-08-17T11:54:57Z: image `v0.27.1-gb10-rc7`,
  `max_model_len=1048576`, hook skipped, thinking off, KV arena 26.3 GiB.
  No image rebuild. Campaign C1 numbers unchanged (E13 103.45, E14 109.77).

## 0.1.1 - 2026-08-14

### Changed

- Live winner is rc7 Arm B: skip the `#31` CPU thinking-budget hook, keep
  `max_model_len=1M`. Thinking default later moved to max in 0.1.3.

### Validated

- Exact 32k×6 on fresh 1M: 46.36 / 46.38 / 48.48 tok/s, 1.05×, MTP 95.1%.
- Exact 128k C1 on live 1M: 99.91 tok/s, 1.00×, ITL 10.0 ms, MTP 95.6%.
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
