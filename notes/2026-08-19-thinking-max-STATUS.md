# Campaign status — 2026-08-19 thinking=max fold-in

- Pause start: `2026-08-19T15:14:13Z` (`HINDSIGHT_WAS=active`, now inactive)
- Bounce PID on head node: `630130` started `2026-08-19T15:15:08Z`
- Image unchanged: `dspark-vllm-gb10:v0.27.1-gb10-rc7`
- Live env: `DEFAULT_THINKING=max`, `# DEFAULT_THINKING_TOKEN_BUDGET=32768` stays commented
- `#31` CPU hook still skipped
- New bind-mounts: Mia `#48` GPU budget (0.27.1 port + greedy gate), `#55`, `#52492`
- Start script now scp's every bind-mounted hotfix (vllm-switch scripted path)

## Clients

- Hindsight extra body already `thinking:false` / `enable_thinking:false` / `include_reasoning:false` — keep
- All live Hermes profiles send `thinking:true` + `reasoning_effort:max` for 0731 — they want the new default

## Next after READY

1. Inspect argv + hotfix `--status`
2. Run `/tmp/ds4-ports/post-bounce-probes.py` on the head node
3. Exact 32k×6 think-off (must stay ~43+ tok/s)
4. 32k×c4 starve check before deciding on Mia `#90`
5. spark-eval + grok-4.5
6. Restore Hindsight + coverage
7. Public repo sweep
