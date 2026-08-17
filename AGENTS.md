# Agent notes

This repository is the source of truth for DeepSeek V4 Flash 0731 on the
two-node GB10 cluster. The KB page is a mirror.

- Current winner: `PROJECT-DECISIONS.md` + `docs/CAMPAIGN_2026-08-14.md`
- Machine state: `project-status.json`
- Image/patch contract: `docs/RUNTIME_V0271_GB10.md` + `runtime/patches/vllm/series`
- 0029 is staged, not baked. Do not add it to `series` without an A/B.
- Git author: `Chris <16943149+nepenth@users.noreply.github.com>`
- Do not commit host IPs, credentials, or `.env` files. Use `*.env.example`.
- Do not bounce the live cluster or rebuild the image from a docs sync.
