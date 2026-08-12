# Documentation Assets

These SVGs are repository-local, editable evidence diagrams. They use no
external image host, private endpoint, or cluster identifier.

| Asset | Source data | Purpose |
|---|---|---|
| `gb10-runtime-architecture.svg` | Current runtime and control-plane contract | Shows the pinned build, two-node TP=2 deployment, and rollback boundary. |
| `long-context-ab.svg` | Final long A/B table in `GB10_DSV4_HANDOFF_2026-08-12.md` | Shows favorable prefill, TTFT, and end-to-end latency deltas for the four 131K/300K cases and the 900K proof. |
| `qualification-gates.svg` | `run-qualification.sh` and the final handoff | Shows the ordered evidence gates required for a new candidate. |

When changing a metric, update the corresponding table in the handoff and the
SVG in the same commit. Do not replace measured percentages with rounded claims
that are not traceable to the private raw A/B reports.
