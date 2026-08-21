# Documentation Assets

These SVGs are repository-local, editable evidence diagrams. They use no
external image host, private endpoint, or cluster identifier.

- **gb10-runtime-architecture.svg** — live 2026-08-21 contract (rc7, 1M,
  thinking=max, dual-HCA). Source: `docs/DEEPSEEK_V4_FLASH_0731.md`.
- **dual-hca-busbw.svg** — isolated nccl-tests 1 GiB busbw 12.53 → 23.55
  GB/s. Source: `notes/2026-08-21-dual-hca.md`.
- **long-context-ab.svg** — 2026-08-12 rc7 vs Stage-C image A/B only, **not**
  the fabric change. Source: `GB10_DSV4_HANDOFF_2026-08-12.md`.
- **qualification-gates.svg** — ordered evidence gates; rc7 qualified and
  dual-HCA promoted.

When changing a metric, update the corresponding table or note and the SVG
in the same commit. Do not replace measured percentages with rounded claims
that are not traceable to a dated report.
