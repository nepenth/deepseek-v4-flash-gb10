#!/usr/bin/env python3
"""Deterministic multi-needle long-context retrieval qualification."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def post_json(url: str, body: dict, timeout: float = 3600) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def token_count(base_url: str, model: str, prompt: str) -> int:
    return post_json(
        base_url.removesuffix("/v1") + "/tokenize",
        {"model": model, "prompt": prompt},
        timeout=600,
    )["count"]


def build_case(base_url: str, model: str, target: int) -> tuple[str, list[str], int]:
    codes = [f"N{target}-ALPHA-7319", f"N{target}-BRAVO-2846", f"N{target}-CHARLIE-9052"]
    unit = "archival filler record with no answer value. "
    seed = unit * max(1, target // 8)
    measured = token_count(base_url, model, seed)
    if measured < target - 256:
        seed += unit * max(1, (target - measured) // 4)
    third = len(seed) // 3
    prompt = (
        seed[:third]
        + f"\nIMPORTANT NEEDLE ONE: {codes[0]}\n"
        + seed[third : 2 * third]
        + f"\nIMPORTANT NEEDLE TWO: {codes[1]}\n"
        + seed[2 * third :]
        + f"\nIMPORTANT NEEDLE THREE: {codes[2]}\n"
        + "\nReturn the three needle codes in their original order and no other text."
    )
    return prompt, codes, token_count(base_url, model, prompt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lengths", default="8192,32768,131072,300000,380000")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "cases": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    for target in (int(value) for value in args.lengths.split(",")):
        prompt, codes, actual = build_case(args.base_url, args.model, target)
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 96,
            "chat_template_kwargs": {"thinking": False},
        }
        started = time.perf_counter()
        response = post_json(f"{args.base_url}/chat/completions", body)
        elapsed = time.perf_counter() - started
        text = (((response.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
        positions = [text.find(code) for code in codes]
        ok = all(position >= 0 for position in positions) and positions == sorted(positions)
        failed |= not ok
        case = {
            "target_tokens": target,
            "actual_tokens": actual,
            "codes": codes,
            "positions": positions,
            "elapsed_s": elapsed,
            "ok": ok,
            "response": text,
            "usage": response.get("usage"),
        }
        report["cases"].append(case)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(case, sort_keys=True), flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["ok"] = not failed
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
