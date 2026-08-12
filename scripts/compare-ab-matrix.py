#!/usr/bin/env python3
"""Compare two compatible ab-matrix-0731.py reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = (
    ("median_prefill_tok_s", "Prefill tok/s", True),
    ("median_aggregate_output_tok_s", "Aggregate output tok/s", True),
    ("median_output_tok_s", "Per-request output tok/s", True),
    ("median_ttft_s", "Median TTFT s", False),
    ("p95_ttft_s", "P95 TTFT s", False),
    ("median_elapsed_s", "Median end-to-end latency s", False),
    ("p95_elapsed_s", "P95 end-to-end latency s", False),
    ("median_decode_token_latency_ms", "Median decode-token latency ms", False),
    ("p95_decode_token_latency_ms", "P95 decode-token latency ms", False),
)


def load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def case_index(report: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(case["target_prompt_tokens"]), int(case["concurrency"])): case
        for case in report["cases"]
    }


def delta(before: float, after: float) -> float | None:
    return None if before == 0 else (after / before - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--output")
    args = parser.parse_args()
    before = load(args.before)
    after = load(args.after)
    for field in ("fixture_sha256", "max_tokens", "trials"):
        if before.get(field) != after.get(field):
            parser.error(f"reports differ in {field}; they are not a controlled A/B")
    before_cases = case_index(before)
    after_cases = case_index(after)
    if before_cases.keys() != after_cases.keys():
        parser.error("reports contain different matrix cases")

    comparison: dict[str, Any] = {
        "after": args.after,
        "before": args.before,
        "fixture_sha256": before["fixture_sha256"],
        "cases": [],
    }
    lines = [
        "| Prompt tokens | Concurrency | Metric | Before | After | Candidate change (positive is favorable) |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for key in sorted(before_cases):
        before_summary = before_cases[key]["summary"]
        after_summary = after_cases[key]["summary"]
        result = {"concurrency": key[1], "metrics": {}, "target_prompt_tokens": key[0]}
        for metric, label, higher_is_better in METRICS:
            old = float(before_summary[metric])
            new = float(after_summary[metric])
            raw_change = delta(old, new)
            favorable_change = raw_change if higher_is_better else (
                None if raw_change is None else -raw_change
            )
            result["metrics"][metric] = {
                "after": new,
                "before": old,
                "favorable_change_percent": favorable_change,
                "higher_is_better": higher_is_better,
                "raw_change_percent": raw_change,
            }
            change_text = (
                "n/a" if favorable_change is None else f"{favorable_change:+.1f}%"
            )
            lines.append(f"| {key[0]:,} | {key[1]} | {label} | {old:.2f} | {new:.2f} | {change_text} |")
        comparison["cases"].append(result)
    rendered = "\n".join(lines) + "\n"
    print(rendered, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        output.with_suffix(".json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
