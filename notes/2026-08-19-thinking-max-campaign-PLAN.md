# Campaign: thinking=max + recommended fold-ins (2026-08-19)

Chris go-ahead: server default `thinking=max`, port recommended
items, A/B, then spark-eval, then public-safe repo sweep. Pause
Hindsight for the exclusive window.

## Non-negotiable

- Keep image `dspark-vllm-gb10:v0.27.1-gb10-rc7`. Bind-mount hotfixes only.
- Port Mia PR #48. Do NOT invent a closer. Do NOT set
  `DEFAULT_THINKING_TOKEN_BUDGET` omit-field (that is the #39 cliff).
- Keep `_requires_logits_processing` greedy gate (temp=0 must fire closer).
- NEVER set `VLLM_USE_V2_MODEL_RUNNER=0`.
- Keep `#31` CPU host-scan skipped (`DSPARK_SKIP_ISSUE31_HOTFIX=1`).
- Hindsight MUST keep `thinking:false` (structured JSON in content).
- Public repo: no hostnames, IPs, usernames, tokens, private DNS.

## Sequence

1. Verify clients (Hindsight / Hermes / others).
2. Port patches into the head-node `~/vllm-v0271-gb10` + git src:
   - `#52492` indexer capture guard
   - start-script scp all bind-mounted hotfixes
   - Mia `#55` tool-truncation finish_reason
   - Mia `#48` GPU thinking budget (0.27.2 anchors + greedy gate)
   - `DEFAULT_THINKING=max` (no omit-field budget)
3. Pause Hindsight. Journal UTC.
4. Bounce canary (force-recreate both ranks; same-profile switch is not enough).
5. A/B:
   - clocks both ranks ~2100 under load
   - smoke `/v1/models`
   - thinking-off client still off (Hindsight shape)
   - thinking=max default produces reasoning; budget 8/32/256 closes
   - exact 32k×6 think-off ≥ ~43 tok/s (no #31 tax)
   - 32k×6 think-max recorded (not a fail if slower — tax must be bounded)
   - one 1M-class needle if #52492 is in
   - #90 inflight only if 32k×c4 starve still reproduces
6. spark-eval full-compare + grok-4.5 judge.
7. Restore Hindsight + canary + coverage audit.
8. Public repo sweep + push.

## Rollback

Previous winner: `DEFAULT_THINKING=off`, skip #31, suppress-stops +
busy-loop on, 1M / 6 / 8192 / 0.84 / 26.3 GiB KV.
