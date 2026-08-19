# 2026-08-17 — repo contract sync

## Done

- Audited `nepenth/deepseek-v4-flash-gb10` @ `448ea1b` against the live head node.
- Campaign docs/ledger/E12-E14 were already committed. Deployable contract was not.
- Copied missing `#31` hotfix from the live cluster.
- Synced compose, start script, and canary env example to the live 1M winner.
- Added `PROJECT-DECISIONS.md` and changelog 0.1.2.

## Evidence

- Live 2026-08-17T11:54:57Z: `dsv4-v0271-canary-vllm-dspark-1`, image
  `dspark-vllm-gb10:v0.27.1-gb10-rc7`, `max_model_len=1048576`,
  `DSPARK_SKIP_ISSUE31_HOTFIX=1`, `DEFAULT_THINKING=off`,
  `KV_CACHE_MEMORY=28235618304`, `LONG_PREFILL_TOKEN_THRESHOLD=1024`.
- suppress-stops file already matched git; issue31 file was live-only.
- 0029 remains in-tree and out of `series`.

## Next

- Do not bake 0029 until A/B.
- Do not bounce the live winner for this docs/contract sync.
