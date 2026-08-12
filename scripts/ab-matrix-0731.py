#!/usr/bin/env python3
"""Run a repeatable DeepSeek V4 Flash API performance-matrix arm.

The first invocation creates a fixture file containing every exact request
prompt. Pass that same file to the other arm so that the server receives
byte-identical inputs. Each warmup and measured request has a unique first
cache block, preventing prefix caching from contaminating prefill timings.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FIXTURE_VERSION = 1
INSTRUCTION = (
    "\nProduce a continuous sequence of lowercase technical terms until the "
    "requested output limit. Do not reason aloud or add a preamble."
)


def request_json(url: str, body: dict[str, Any], timeout: int = 3600) -> dict[str, Any]:
    for attempt in range(4):
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.URLError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def tokenize_url(base_url: str) -> str:
    return base_url.removesuffix("/v1") + "/tokenize"


def token_count(base_url: str, model: str, prompt: str) -> int:
    return int(request_json(tokenize_url(base_url), {"model": model, "prompt": prompt}, 600)["count"])


def make_prompt(base_url: str, model: str, target_tokens: int, fixture_id: str) -> tuple[str, int]:
    prefix = f"ab matrix fixture {fixture_id} unique prefix "
    unit = "archival context benchmark datum with no answer value. "
    prompt = prefix + unit * max(1, target_tokens // 8)
    count = token_count(base_url, model, prompt)
    while count < target_tokens:
        prompt += unit * max(1, (target_tokens - count) // 4)
        count = token_count(base_url, model, prompt)
    return prompt, count


def parse_csv(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item)
    if not parsed or any(item < 1 for item in parsed):
        raise ValueError("values must be positive comma-separated integers")
    return parsed


def fixture_digest(fixture: dict[str, Any]) -> str:
    canonical = json.dumps(fixture["prompts"], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def load_or_create_fixture(
    path: Path,
    base_url: str,
    model: str,
    prompt_lengths: tuple[int, ...],
    concurrency: tuple[int, ...],
    trials: int,
) -> dict[str, Any]:
    if path.exists():
        fixture = json.loads(path.read_text(encoding="utf-8"))
        if fixture.get("version") != FIXTURE_VERSION:
            raise ValueError(f"unsupported fixture version in {path}")
        expected = {
            "prompt_lengths": list(prompt_lengths),
            "concurrency": list(concurrency),
            "trials": trials,
        }
        actual = {key: fixture.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"fixture shape differs: expected {expected}, found {actual}")
        for prompt in fixture.get("prompts", {}).values():
            text = prompt["text"]
            if hashlib.sha256(text.encode()).hexdigest() != prompt["sha256"]:
                raise ValueError(f"fixture prompt hash mismatch in {path}")
        return fixture

    prompts: dict[str, dict[str, Any]] = {}
    for length in prompt_lengths:
        for level in concurrency:
            for phase in ("warmup", "trial"):
                count = 1 if phase == "warmup" else trials
                for trial in range(1, count + 1):
                    for request_index in range(level):
                        fixture_id = f"p{length}-c{level}-{phase}{trial}-r{request_index}"
                        text, actual_tokens = make_prompt(base_url, model, length, fixture_id)
                        prompts[fixture_id] = {
                            "actual_tokens": actual_tokens,
                            "sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "text": text,
                        }
                        print(
                            json.dumps(
                                {
                                    "event": "fixture",
                                    "id": fixture_id,
                                    "target_tokens": length,
                                    "actual_tokens": actual_tokens,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

    fixture = {
        "version": FIXTURE_VERSION,
        "model": model,
        "prompt_lengths": list(prompt_lengths),
        "concurrency": list(concurrency),
        "trials": trials,
        "prompts": prompts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return fixture


def stream_one(base_url: str, model: str, prompt: str, max_tokens: int) -> dict[str, float | int]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt + INSTRUCTION}],
        "chat_template_kwargs": {"thinking": False},
        "ignore_eos": True,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    usage: dict[str, int] | None = None
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if first is None and (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
            ):
                first = time.perf_counter()
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    if not usage or not usage.get("completion_tokens"):
        raise RuntimeError("stream response did not include completion usage")
    first = first or finished
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage["completion_tokens"])
    ttft_s = first - started
    decode_s = max(finished - first, 0.001)
    return {
        "completion_tokens": completion_tokens,
        "decode_token_latency_ms": 1000.0 * decode_s / completion_tokens,
        "elapsed_s": finished - started,
        "output_tok_s": completion_tokens / decode_s,
        "prefill_tok_s": prompt_tokens / max(ttft_s, 0.001),
        "prompt_tokens": prompt_tokens,
        "ttft_s": ttft_s,
    }


async def run_request_group(
    base_url: str,
    model: str,
    prompts: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    requests = await asyncio.gather(
        *[
            asyncio.to_thread(stream_one, base_url, model, prompt["text"], max_tokens)
            for prompt in prompts
        ]
    )
    elapsed_s = time.perf_counter() - started
    completion_tokens = sum(int(request["completion_tokens"]) for request in requests)
    return {
        "aggregate_output_tok_s": completion_tokens / max(elapsed_s, 0.001),
        "elapsed_s": elapsed_s,
        "requests": requests,
        "total_completion_tokens": completion_tokens,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(trials: list[dict[str, Any]]) -> dict[str, float]:
    requests = [request for trial in trials for request in trial["requests"]]
    return {
        "median_decode_token_latency_ms": statistics.median(
            float(request["decode_token_latency_ms"]) for request in requests
        ),
        "median_elapsed_s": statistics.median(float(request["elapsed_s"]) for request in requests),
        "median_aggregate_output_tok_s": statistics.median(
            float(trial["aggregate_output_tok_s"]) for trial in trials
        ),
        "median_output_tok_s": statistics.median(float(request["output_tok_s"]) for request in requests),
        "median_prefill_tok_s": statistics.median(float(request["prefill_tok_s"]) for request in requests),
        "median_ttft_s": statistics.median(float(request["ttft_s"]) for request in requests),
        "p95_decode_token_latency_ms": percentile(
            [float(request["decode_token_latency_ms"]) for request in requests], 0.95
        ),
        "p95_elapsed_s": percentile([float(request["elapsed_s"]) for request in requests], 0.95),
        "p95_ttft_s": percentile([float(request["ttft_s"]) for request in requests], 0.95),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-lengths", default="256,8192,32768,131072")
    parser.add_argument("--concurrency", default="1,2,4,6")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prompt_lengths = parse_csv(args.prompt_lengths)
    concurrency = parse_csv(args.concurrency)
    if args.trials < 1 or args.max_tokens < 1:
        parser.error("--trials and --max-tokens must be positive")

    fixture_path = Path(args.prompt_cache)
    fixture = load_or_create_fixture(
        fixture_path,
        args.base_url,
        args.model,
        prompt_lengths,
        concurrency,
        args.trials,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "base_url": args.base_url,
        "cases": [],
        "fixture_sha256": fixture_digest(fixture),
        "max_tokens": args.max_tokens,
        "model": args.model,
        "prompt_cache": str(fixture_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "trials": args.trials,
    }
    for length in prompt_lengths:
        for level in concurrency:
            warmups = [fixture["prompts"][f"p{length}-c{level}-warmup1-r{index}"] for index in range(level)]
            await run_request_group(args.base_url, args.model, warmups, args.max_tokens)
            trials = []
            for trial in range(1, args.trials + 1):
                prompts = [
                    fixture["prompts"][f"p{length}-c{level}-trial{trial}-r{index}"]
                    for index in range(level)
                ]
                trials.append(await run_request_group(args.base_url, args.model, prompts, args.max_tokens))
            case = {
                "concurrency": level,
                "summary": summarize(trials),
                "target_prompt_tokens": length,
                "trials": trials,
            }
            report["cases"].append(case)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(case, sort_keys=True), flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
