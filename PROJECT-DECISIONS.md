# PROJECT-DECISIONS — DeepSeek V4 Flash GB10

Material serving decisions only. Experiment numbers live in
`docs/CAMPAIGN_2026-08-14.md` and `experiments/2026-08-14-ledger.jsonl`.

## 2026-08-21 — Dual-HCA QSFP merge is live fabric

**Decision:** Keep both GB10 QSFP virtual HCAs in `NCCL_IB_HCA` /
`NCCL_SOCKET_IFNAME` with `NCCL_IB_MERGE_NICS=1` and jumbo 9000. Do **not**
pin `NCCL_IB_GID_INDEX`. Do **not** rebase the image.

**Rationale:** Isolated nccl-tests all_reduce 1 GiB busbw went 12.53 → 23.55
GB/s. IPv4 RoCEv2 GID indexes disagree across the two members, so a single
pin is wrong. Post-bounce think-off C1 returned to the ~75 tok/s band; 32k×6
spread stayed 1.00×; KV pool unchanged.

**Consequences:** Start script skips GID resolve when HCA is a comma list.
GLOO/TP remain single-ifname. Jumbo is a host netdev setting, not compose.

## 2026-08-14 — Live winner is rc7 Arm B + 1M, not a new image

**Decision:** Keep `dspark-vllm-gb10:v0.27.1-gb10-rc7` as the baked image.
Apply the winner as runtime env + bind-mount hotfixes, not a rebuild.

**Winner knobs:**

- `MAX_MODEL_LEN=1048576`
- `MAX_NUM_SEQS=6`
- `MAX_NUM_BATCHED_TOKENS=8192`
- GMU `0.84`
- `KV_CACHE_DTYPE=fp8_ds_mla`
- `KV_CACHE_MEMORY=28235618304` (26.3 GiB, 2.74M tokens, 2.61x at 1M)
- DSpark K=5, `MOE_BACKEND=deep_gemm`
- `LONG_PREFILL_TOKEN_THRESHOLD=1024`
- `DEFAULT_THINKING=max` (2026-08-19)
- `DSPARK_SKIP_ISSUE31_HOTFIX=1`
- GPU thinking budget ON (`patches/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py`, Mia #48 port)
- `#55` tool-truncation + `#52492` indexer capture guard ON
- suppress-stops 0.27.1 rewrite ON (`patches/hotfix-dsv4-suppress-stops-v0271.py`)
- V2 runner ON. NEVER set `VLLM_USE_V2_MODEL_RUNNER=0`.

**Rationale:** 32k x 6 is the production gate. C1 hid the #31 tax. Exact
issue27 on a fresh 1M bounce is 46.4 tok/s with hook off vs 24.5 with hook
on. Official 1M works (1.04M retrieval PASS). Correctness gates held
(Phase 5 11/11, encoding 4/4, 3x restart greedy identical, 1k soak
1000/1000).

**Consequences:** Deploy from the canary env example, not leftover Stage-C
or 400k docs. Same-profile `vllm-switch` does not apply env-only changes;
force-remove both rank containers.

## 2026-08-14 — Do not bake 0029 until A/B vs this winner

**Decision:** `runtime/patches/vllm/0029-warm-dspark-probabilistic-rejection-helpers.patch`
stays in-tree and OUT of `series`. rc7 remains the baked image.

**Rationale:** Five Triton helpers still JIT after official warmup. 0029
covers those keys. Baking without a 32k x 6 + C1 A/B would mix image
change with a known-good envelope.

## 2026-08-19 — Server default is thinking=max via Mia #48

**Decision:** `DEFAULT_THINKING=max`. Port Mia PR #48 to 0.27.1
(`VLLMValidationError` gate + greedy `_requires_logits_processing`).
Do NOT set `DEFAULT_THINKING_TOKEN_BUDGET` (that is the #39 cliff).
Clients that must not think (memory extractors) send
`chat_template_kwargs.thinking=false`.

**Measured:** A no-think client (`chat_template_kwargs.thinking=false`) still returns empty reasoning.
32k×6 think-off stayed 45.39 tok/s median (no #31 tax). 32k×4 was 61.61
tok/s — do not port Mia #90 inflight prefills.

## 2026-08-14 — Keep advertised 1M; shrink KV arena if you need free RAM

**Decision:** Do not cut `max_model_len` to free coexist headroom. Shrink
`KV_CACHE_MEMORY`. Tony-shaped 10.3 GiB still covers 1.04M single-request
and 32k x 6. The 26.3 GiB arena is concurrency luxury, not correctness.

**Rationale:** Cutting GMU 0.78 -> 0.70 collapsed usable context to 2816
tokens in Tony's notes. The ceiling is the arena, not the advertised
length.

## Rejected / do-not-resurrect

- Stock vLLM 0.27.1 / eugr GB10 canary as a Spark DS4 path
- Anemll 0.25 rebase of suppress-stops (missing `import sys` +
  `TokenizersBackend` factory)
- `nvfp4_ds_mla` as packed NVFP4 (FP8-record alias only)
- DSpark K=7
- `VLLM_USE_V2_MODEL_RUNNER=0`
- Raising GMU above 0.84
- r0b0tlab DSpark-v026-SM121 (Q 0.908 vs Stage-C 0.933 at 524k)
- Inventing a thinking-budget closer instead of Mia #48
- Using `decode-bench.py` as the 32k x 6 production number (harness lie;
  use exact `issue27`)
