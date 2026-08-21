# 2026-08-21 thinking=max official eval

Protocol P4. Do **not** compare these Q scores to think-off spark-eval (~0.92).

## Setup

- Image `dspark-vllm-gb10:v0.27.1-gb10-rc7` (unchanged)
- Envelope 1,048,576 / seqs 6 / batch 8192 / GMU 0.84 / 26.3 GiB KV
- Dual-HCA QSFP merge on (see `notes/2026-08-21-dual-hca.md`)
- Server default `thinking=max` via Mia #48 GPU closer
- `#31` CPU hook skipped
- Coding hang-guard is a floor; per-request `thinking_token_budget=4096` is
  coding-only. Server omit-field budget stays unset

## Result

Official full-compare `dsv4-rc7-dualhca-20260821T012400Z`:

- Q **0.693** · capability 0.892 · runtime 1.000
- tools 1.000 · ops 0.792 · schema 0.800 · coding **0.875**
- Grok-4.5 **4.667 / 15**

Versus 2026-08-19 thinking=max official: Q **0.583** / coding **0.125**.

`run_state=FAIL` was a schema-answer-plan timeout plus an artifact_integrity
flag; hashed `integrity.json` was `ok: true`. Treat Q/coding as the score,
not the wrapper state.

## Do not

- Compare P4 Q to the 2026-08-10 think-off Polish row (Q 0.925 / Grok 4.812)
- Set `DEFAULT_THINKING_TOKEN_BUDGET` on the server to "help" coding
- Re-run think-off full-compare and paste it next to this table
