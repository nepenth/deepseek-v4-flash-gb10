# Agent notes

This repository is a **public recipe** for DeepSeek-V4-Flash-0731 on two-node
GB10. Ingest `LLM_README.md` before mutating docs or knobs.

- Live winner: `PROJECT-DECISIONS.md` + `project-status.json` → `live`
- Prove argv from `docker inspect` `Config.Cmd`, not leftover profiles
- Image/patch contract: `docs/RUNTIME_V0271_GB10.md` + `runtime/patches/vllm/series`
- 0029 is staged, not baked. Do not add it to `series` without an A/B
- Git author: preserve the repository's existing GitHub noreply author
- Do not commit host IPs, credentials, `.env` files, local DNS, or operator
  home paths. Use `*.env.example` + `CHANGEME` placeholders
- Run `runtime/scripts/check-private-data.sh` before every commit
- Do not bounce a live cluster or rebuild the image from a docs sync
