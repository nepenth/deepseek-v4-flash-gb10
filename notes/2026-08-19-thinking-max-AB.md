# A/B after thinking=max bounce (2026-08-19)

READY 15:21:56Z. Image still `dspark-vllm-gb10:v0.27.1-gb10-rc7`.
`DEFAULT_THINKING=max`. `DEFAULT_THINKING_TOKEN_BUDGET` empty.
Hotfixes in-container: #48 GPU, #55, #52492, busy_loop 2ms. #31 CPU skipped.
Worker received all scp'd hotfix files.

## Functional

- no-think client-off: `PING-OK-17`, reasoning_len=0
- Server default: reasoning 111 chars, content `42`
- Budget 8 @ 0.6: reasoning 35 chars (closer fired)
- Budget 32 @ 0.6: reasoning 81 chars, content `42`
- Budget 8 @ temp 0: reasoning 45 chars (greedy gate fired)

## Decode gate (think-off, exact 32770)

- 32k×6: 43.92 / **45.39** / 46.79 tok/s, spread 1.07×, ITL 21–23 ms, 0 preempt, MTP 94.9%
  Prior winner post-#27: 44.63 / 44.65 / 46.61. **No #31 tax.**
- 32k×4: 60.53 / **61.61** / 62.45 tok/s, spread 1.03×. **No #90 starve.** Skip Mia #90.

Clocks ~2093 / 2080 after waves.
