# Documentation Guide

## Current Documents

LLMs should ingest [`../LLM_README.md`](../LLM_README.md) first.

Start with these for the supported vLLM 0.27.1 GB10 runtime:

| Document | Purpose |
|---|---|
| [LLM ingest](../LLM_README.md) | Machine-oriented contract, pitfalls, file map. |
| [DeepSeek V4 Flash 0731](DEEPSEEK_V4_FLASH_0731.md) | **Live** serving contract (1M, thinking=max, dual-HCA). |
| [2026-08-21 dual-HCA](../notes/2026-08-21-dual-hca.md) | Isolated busbw + think-off decode after fabric merge. |
| [2026-08-21 thinking=max eval](../notes/2026-08-21-thinkmax-eval.md) | Official full-compare + Grok-4.5 (protocol P4). |
| [2026-08-14 campaign](CAMPAIGN_2026-08-14.md) | 1M correctness / exact-length decode protocol (pre-fabric). |
| [Runtime contract](RUNTIME_V0271_GB10.md) | Locked inputs, build and patch policy. |
| [Setup](SETUP.md) | Safe standalone deployment sequence. |
| [Control plane](CLUSTER_CONTROL_PLANE.md) | `vllm-switch` lifecycle, validation, and rollback. |
| [NVFP4 DS-MLA](NVFP4_DS_MLA.md) | Why FP8 DS-MLA is supported today. |
| [Asset registry](assets/README.md) | Editable SVG diagrams and their measured sources. |
| [Decisions](../PROJECT-DECISIONS.md) | Winner knobs and rejected paths. |
| [GB10 handoff](GB10_DSV4_HANDOFF_2026-08-12.md) | Image/A/B history only. Envelope numbers in it are **not** live. |

## Historical References

The following files are retained as provenance or prior investigation notes.
They must not override the current runtime contract or be treated as live
cluster configuration without a new audit:

- `ENVS.md`
- `PATCHES.md`
- `GB10_EXECUTION_PLAN.md`
- `DSML_SYNTAX_TEMP_ASYMMETRY.md`

Use [UPSTREAM_AUDIT_2026-08-11.md](UPSTREAM_AUDIT_2026-08-11.md) for the
historical selection evidence behind the current recipe. Keep host-specific
notes, raw logs, and credential-bearing configuration in ignored `.private/`
storage rather than this documentation tree.
