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
import os
import re
import statistics
import subprocess
import threading
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
_METRICS_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
_METRICS_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)"
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


def metrics_url(base_url: str) -> str:
    return base_url.removesuffix("/v1") + "/metrics"


def _parse_prom_samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    samples: list[tuple[str, dict[str, str], float]] = []
    for raw in text.splitlines():
        if not raw or raw.startswith("#"):
            continue
        match = _METRICS_LINE_RE.match(raw)
        if not match:
            continue
        labels = dict(_METRICS_LABEL_RE.findall(match.group("labels") or ""))
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        samples.append((match.group("name"), labels, value))
    return samples


def scrape_vllm_metrics(url: str, timeout: int = 10) -> dict[str, Any] | None:
    """Snapshot /metrics. Missing endpoint must not break fixture-lock A/B."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text = response.read().decode()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    samples = _parse_prom_samples(text)
    accepted_by_pos: dict[str, float] = {}
    drafts = draft_tokens = accepted = None
    queue_sum = queue_count = None
    kv_usage = waiting = running = None
    cache_info: dict[str, str] = {}
    for name, labels, value in samples:
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            accepted_by_pos[str(labels.get("position", ""))] = value
        elif name == "vllm:spec_decode_num_drafts_total":
            drafts = value
        elif name == "vllm:spec_decode_num_draft_tokens_total":
            draft_tokens = value
        elif name == "vllm:spec_decode_num_accepted_tokens_total":
            accepted = value
        elif name == "vllm:request_queue_time_seconds_sum":
            queue_sum = value
        elif name == "vllm:request_queue_time_seconds_count":
            queue_count = value
        elif name == "vllm:kv_cache_usage_perc":
            kv_usage = value
        elif name == "vllm:num_requests_waiting":
            waiting = value
        elif name == "vllm:num_requests_running":
            running = value
        elif name == "vllm:cache_config_info":
            cache_info = labels
    return {
        "accepted": accepted,
        "accepted_by_pos": accepted_by_pos,
        "cache_info": cache_info,
        "draft_tokens": draft_tokens,
        "drafts": drafts,
        "kv_cache_usage_perc": kv_usage,
        "num_requests_running": running,
        "num_requests_waiting": waiting,
        "queue_count": queue_count,
        "queue_sum_s": queue_sum,
    }


def sample_gpu_memory() -> dict[str, Any]:
    """Best-effort host GPU / unified-memory sample. GB10 nvidia-smi is often N/A."""
    result: dict[str, Any] = {}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        rows = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            rows.append(
                {
                    "index": parts[0],
                    "memory_total": parts[2],
                    "memory_used": parts[1],
                    "utilization": parts[3],
                }
            )
        if rows:
            result["nvidia_smi"] = rows
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        parsed: dict[str, int] = {}
        for line in meminfo.splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            number = raw.strip().split()[0]
            if number.isdigit():
                parsed[key] = int(number)
        if parsed:
            result["host_meminfo_kib"] = {
                key: parsed[key]
                for key in ("MemTotal", "MemAvailable", "MemFree", "Cached", "SwapFree")
                if key in parsed
            }
    except OSError:
        pass
    extra = os.environ.get("DSV4_AB_GPU_MEM_CMD")
    if extra:
        try:
            completed = subprocess.run(
                extra,
                check=False,
                capture_output=True,
                shell=True,
                text=True,
                timeout=8,
            )
            result["gpu_mem_cmd"] = {
                "cmd": extra,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip()[:2000],
            }
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def _counter_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def summarize_spec_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    drafts = _counter_delta(before.get("drafts"), after.get("drafts"))
    draft_tokens = _counter_delta(before.get("draft_tokens"), after.get("draft_tokens"))
    accepted = _counter_delta(before.get("accepted"), after.get("accepted"))
    pos_before = before.get("accepted_by_pos") or {}
    pos_after = after.get("accepted_by_pos") or {}
    positions = sorted(set(pos_before) | set(pos_after), key=lambda item: int(item) if str(item).isdigit() else item)
    accepted_by_pos = {
        position: _counter_delta(pos_before.get(position), pos_after.get(position))
        for position in positions
    }
    queue_sum = _counter_delta(before.get("queue_sum_s"), after.get("queue_sum_s"))
    queue_count = _counter_delta(before.get("queue_count"), after.get("queue_count"))
    acceptance_by_pos_rate: dict[str, float | None] = {}
    for position, count in accepted_by_pos.items():
        if count is None or drafts in (None, 0):
            acceptance_by_pos_rate[position] = None
        else:
            acceptance_by_pos_rate[position] = count / drafts
    mean_acceptance_length = None
    if accepted is not None and drafts not in (None, 0):
        mean_acceptance_length = accepted / drafts
    mean_queue_s = None
    if queue_sum is not None and queue_count not in (None, 0):
        mean_queue_s = queue_sum / queue_count
    return {
        "accepted_tokens": accepted,
        "accepted_tokens_by_pos": accepted_by_pos,
        "acceptance_rate_by_pos": acceptance_by_pos_rate,
        "draft_tokens": draft_tokens,
        "drafts": drafts,
        "mean_acceptance_length": mean_acceptance_length,
        "mean_queue_s": mean_queue_s,
        "queue_count": queue_count,
        "queue_sum_s": queue_sum,
    }


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


class MetricsSampler:
    """Sample /metrics and optional GPU memory while a trial is in flight."""

    def __init__(self, url: str | None, interval_s: float = 0.5) -> None:
        self.url = url
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self.gpu_samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.url:
            return
        self._thread = threading.Thread(target=self._run, name="ab-metrics", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            snapshot = scrape_vllm_metrics(self.url)
            if snapshot is not None:
                snapshot["gpu_memory"] = sample_gpu_memory()
                self.samples.append(snapshot)
            if self._stop.wait(self.interval_s):
                break

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _peak_kv_usage(samples: list[dict[str, Any]]) -> float | None:
    values = [
        float(sample["kv_cache_usage_perc"])
        for sample in samples
        if sample.get("kv_cache_usage_perc") is not None
    ]
    return max(values) if values else None


def _peak_waiting(samples: list[dict[str, Any]]) -> float | None:
    values = [
        float(sample["num_requests_waiting"])
        for sample in samples
        if sample.get("num_requests_waiting") is not None
    ]
    return max(values) if values else None


async def run_request_group(
    base_url: str,
    model: str,
    prompts: list[dict[str, Any]],
    max_tokens: int,
    metrics_endpoint: str | None = None,
) -> dict[str, Any]:
    before = scrape_vllm_metrics(metrics_endpoint) if metrics_endpoint else None
    sampler = MetricsSampler(metrics_endpoint)
    sampler.start()
    started = time.perf_counter()
    try:
        requests = await asyncio.gather(
            *[
                asyncio.to_thread(stream_one, base_url, model, prompt["text"], max_tokens)
                for prompt in prompts
            ]
        )
    finally:
        sampler.stop()
    elapsed_s = time.perf_counter() - started
    after = scrape_vllm_metrics(metrics_endpoint) if metrics_endpoint else None
    completion_tokens = sum(int(request["completion_tokens"]) for request in requests)
    result: dict[str, Any] = {
        "aggregate_output_tok_s": completion_tokens / max(elapsed_s, 0.001),
        "elapsed_s": elapsed_s,
        "requests": requests,
        "total_completion_tokens": completion_tokens,
    }
    if before is not None and after is not None:
        telemetry = summarize_spec_delta(before, after)
        telemetry["peak_kv_cache_usage_perc"] = _peak_kv_usage(sampler.samples + [before, after])
        telemetry["peak_num_requests_waiting"] = _peak_waiting(sampler.samples + [before, after])
        telemetry["gpu_memory_before"] = sample_gpu_memory()
        telemetry["gpu_memory_after"] = sample_gpu_memory()
        cache_info = after.get("cache_info") or before.get("cache_info") or {}
        telemetry["cache_config"] = {
            key: cache_info.get(key)
            for key in (
                "block_size",
                "cache_dtype",
                "gpu_memory_utilization",
                "kv_cache_max_concurrency",
                "kv_cache_size_tokens",
                "num_gpu_blocks",
            )
            if key in cache_info
        }
        result["telemetry"] = telemetry
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [request for trial in trials for request in trial["requests"]]
    summary: dict[str, Any] = {
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
    telemetries = [trial["telemetry"] for trial in trials if trial.get("telemetry")]
    if telemetries:
        positions = sorted(
            {
                position
                for item in telemetries
                for position in (item.get("acceptance_rate_by_pos") or {})
            },
            key=lambda item: int(item) if str(item).isdigit() else item,
        )
        summary["median_mean_acceptance_length"] = _median_or_none(
            [
                float(item["mean_acceptance_length"])
                for item in telemetries
                if item.get("mean_acceptance_length") is not None
            ]
        )
        summary["median_queue_s"] = _median_or_none(
            [float(item["mean_queue_s"]) for item in telemetries if item.get("mean_queue_s") is not None]
        )
        summary["median_peak_kv_cache_usage_perc"] = _median_or_none(
            [
                float(item["peak_kv_cache_usage_perc"])
                for item in telemetries
                if item.get("peak_kv_cache_usage_perc") is not None
            ]
        )
        summary["acceptance_rate_by_pos"] = {
            position: _median_or_none(
                [
                    float(item["acceptance_rate_by_pos"][position])
                    for item in telemetries
                    if (item.get("acceptance_rate_by_pos") or {}).get(position) is not None
                ]
            )
            for position in positions
        }
        if telemetries[0].get("cache_config"):
            summary["cache_config"] = telemetries[0]["cache_config"]
        gpu_available = [
            int(sample["host_meminfo_kib"]["MemAvailable"])
            for item in telemetries
            for sample in (item.get("gpu_memory_after"), item.get("gpu_memory_before"))
            if isinstance(sample, dict) and "host_meminfo_kib" in sample
        ]
        if gpu_available:
            summary["median_host_mem_available_kib"] = statistics.median(gpu_available)
    return summary


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
    parser.add_argument(
        "--metrics-url",
        default="",
        help="Prometheus /metrics URL. Default: <base-url minus /v1>/metrics. "
        "Telemetry is omitted if the scrape fails; fixture lock is unchanged.",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Do not scrape /metrics or GPU memory. Fixture behavior is unchanged.",
    )
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
    metrics_endpoint = None if args.no_telemetry else (args.metrics_url or metrics_url(args.base_url))
    report: dict[str, Any] = {
        "base_url": args.base_url,
        "cases": [],
        "fixture_sha256": fixture_digest(fixture),
        "max_tokens": args.max_tokens,
        "metrics_url": metrics_endpoint,
        "model": args.model,
        "prompt_cache": str(fixture_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "telemetry": not args.no_telemetry,
        "trials": args.trials,
    }
    for length in prompt_lengths:
        for level in concurrency:
            warmups = [fixture["prompts"][f"p{length}-c{level}-warmup1-r{index}"] for index in range(level)]
            await run_request_group(
                args.base_url, args.model, warmups, args.max_tokens, metrics_endpoint=None
            )
            trials = []
            for trial in range(1, args.trials + 1):
                prompts = [
                    fixture["prompts"][f"p{length}-c{level}-trial{trial}-r{index}"]
                    for index in range(level)
                ]
                trials.append(
                    await run_request_group(
                        args.base_url,
                        args.model,
                        prompts,
                        args.max_tokens,
                        metrics_endpoint=metrics_endpoint,
                    )
                )
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
