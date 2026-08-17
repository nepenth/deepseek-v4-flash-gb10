# Documentation Guide

## Current Documents

Start with these for the supported vLLM 0.27.1 GB10 runtime:

| Document | Purpose |
|---|---|
| [GB10 handoff](GB10_DSV4_HANDOFF_2026-08-12.md) | Detailed implementation, measured results, caveats, and next-agent execution plan. |
| [Runtime contract](RUNTIME_V0271_GB10.md) | Locked inputs, build and patch policy, and serving contract. |
| [Setup](SETUP.md) | Safe standalone deployment sequence. |
| [Control plane](CLUSTER_CONTROL_PLANE.md) | `vllm-switch` lifecycle, validation, and rollback. |
| [DeepSeek V4 Flash 0731](DEEPSEEK_V4_FLASH_0731.md) | Model-specific serving configuration and required validation. |
| [NVFP4 DS-MLA](NVFP4_DS_MLA.md) | Why FP8 DS-MLA is supported today and what true NVFP4 work requires. |
| [Asset registry](assets/README.md) | Editable SVG diagrams and their measured sources. |
| [2026-08-14 campaign](CAMPAIGN_2026-08-14.md) | Live Arm B + 1M winner, A/B numbers, failures, next loops. |
| [Decisions](../PROJECT-DECISIONS.md) | Winner knobs and rejected paths. |

## Historical References

The following files are retained as provenance or prior investigation notes.
They must not override the current runtime contract or be treated as live
cluster configuration without a new audit:

- `ENVS.md`
- `PATCHES.md`
- `GB10_EXECUTION_PLAN.md`
- `DSML_SYNTAX_TEMP_ASYMMETRY.md`
- `GLM-NEW-REPORT.md`

Use [UPSTREAM_AUDIT_2026-08-11.md](UPSTREAM_AUDIT_2026-08-11.md) for the
historical selection evidence behind the current recipe. Keep host-specific
notes, raw logs, and credential-bearing configuration in ignored `.private/`
storage rather than this documentation tree.
