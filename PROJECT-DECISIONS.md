# PROJECT-DECISIONS — DeepSeek V4 Flash GB10

Material serving decisions only. Experiment numbers live in
`docs/CAMPAIGN_2026-08-14.md` and `experiments/2026-08-14-ledger.jsonl`.

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
- `DEFAULT_THINKING=off`
- `DSPARK_SKIP_ISSUE31_HOTFIX=1`
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

## 2026-08-14 — Do not invent a thinking=max closer

**Decision:** Leave thinking default off. Port Mia PR #48 only if
`thinking=max` is required. Do not turn it on via
`DEFAULT_THINKING_TOKEN_BUDGET` (that is the #39 cliff).

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
