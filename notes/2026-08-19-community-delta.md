# Community delta vs live rc7 (2026-08-19)

> Superseded as *live* snapshot on 2026-08-21: thinking is now `max` and
> dual-HCA merge is on. Keep this note as the 2026-08-19 community review.

Review only. No bounce, no rebuild.

Live at review: `deepseek-v4-flash-0731-v0271-canary`, image
`dspark-vllm-gb10:v0.27.1-gb10-rc7` (`sha256:b15d2e208b87…` both ranks),
up 44h, `max_model_len=1048576`, seqs 6, batch 8192, GMU 0.84,
`fp8_ds_mla` + 26.3 GiB KV, k=5, thinking off, `#31` skipped,
suppress-stops on, busy_loop 2 ms on, long-prefill 1024.

## Verdict

Stay on official 0731 + rc7. Do not rebase onto eugr b12x, Anemll 0.25,
or Reederey c8r.

## eugr/spark-vllm-docker

- Pushed 2026-08-19 (gitignore / RadixArk Qwen DSpark). Last 0731 recipe
  edit is still 2026-08-01.
- Recipe still `vllm-node-b12x` + **fp8 KV** + instanttensor + B12X
  backends + thinking true/high + seqs 8 + GMU 0.85 + auto max_model_len.
- New env since our Aug 5 dossier: `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
  (we already set this).
- `#349` (open): Aug 15 `eugr/spark-vllm-b12x:latest` dies in CUDA graph
  profiling (`CUDA_ERROR_ILLEGAL_ADDRESS`). Aug 13 image works.
- Official vLLM recipes page now points GB10 at
  `eugr/spark-vllm-b12x:latest`. Docs pointer, not a Spark-DS4 winner.
- Head-node checkout `5415c1f` is 37 commits behind origin. Irrelevant to
  live DS4.

## MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

Still Anemll `0.1.1` (frozen 2026-07-15) + `nvfp4_ds_mla` alias +
thinking max. Real movement is hotfix/packaging, not a new image.

Shipped since our Aug 13/17 notes:

- 2026-08-19 `#90`: `DSPARK_MAX_INFLIGHT_PREFILLS=2` on Anemll (stock
  `--max-num-partial-prefills` rejected). 32k×c4 decode 8.2 → 24.6 tok/s.
  Live rc7 SchedulerConfig has **no** `max_num_partial_prefills`; we
  already have `--long-prefill-token-threshold 1024`.
- 2026-08-18 busy_loop 2 ms (issue `#79`). **Already live** on rc7
  (`busy_loop_s: float = 0.002`).
- 2026-08-18 assistant-final hotfix made **opt-in default off**.
- 2026-08-14 `#48` GPU thinking budget locally merged on Anemll. GitHub
  PR `#48` closed unmerged. Do not invent; port only if thinking=max.
- 2026-08-14 `#55` tool-truncation `finish_reason=length`. Hermes-relevant.

## Other referenced repos

- Anemll/dspark-vllm-gx10: last code 2026-07-15. Dead as a moving target.
- Tony 0731: today PR `#24` reasoning_effort / tokenizer import checks.
  Still Stage-C / `nvfp4_ds_mla`. Not a reason to leave rc7.
- Reederey87/dgx-spark-2x-deepseek-v4-flash: today promoted three
  one-file upstream layers on `main@48bada6` (0.27-content), thinking on,
  C1 ~34 / C8 ~88, KV pin 19.85 GiB / 3.02M `nvfp4_ds_mla`.
  Promoted: `#52492` (indexer short-ctx bake-in), `#51318` (C128A revert),
  `#52329` (cache logits-processing). HOLD: prefix-retention=4096.

## Upstream vLLM

- Latest release still **v0.27.1** (2026-08-11). Live pin
  `0.27.2.dev0+g6e448d0ea`.
- `#41834` SM12x still **open** (updated 2026-08-13).
- `#52447` NVFP4 0731 + DSpark + FP4 KV on SM121 still **open**.
- `#52492` / `#51318` / `#52329` merged upstream 2026-08-16/17.
- Live indexer `forward()` still has the unguarded short-context
  shortcut (`max_seq_len // compress_ratio <= topk_tokens` →
  `_fill_short_context_topk_indices` and return). No
  `is_current_stream_capturing()` on that branch. `#51318` is already
  in our runtime series. We keep `VLLM_USE_BREAKABLE_CUDAGRAPH=0`,
  which is the activation Reederey named — still the one correctness
  port worth a source-level A/B.

## Ops gap on our tree

`start-deepseek-v4-flash-dspark.sh` scp's compose, env, and
`hotfix-dsv4-issue31-v0272.py` only. Compose also bind-mounts
suppress-stops and busy-loop. Next bounce MUST scp those two as well.

## Fold-in ranking

1. Evaluate `#52492` 3-line capture guard as a bind-mount hotfix.
2. Fix start-script scp coverage (ops, no serve change until bounce).
3. Evaluate Mia `#55` tool-truncation for Hermes.
4. Port Mia `#90` inflight N=2 only if 32k×c4 starve reproduces on rc7.
5. Port Mia `#48` only if Chris wants server thinking=max.
6. Watch eugr `#349` and vLLM `#41834`. Do not cut over.
