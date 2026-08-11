#!/usr/bin/env python3
"""OpenAI API, parser, structured-output, and tool replay qualification."""

from __future__ import annotations

import argparse
import atexit
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def post(url: str, body: dict, *, stream: bool = False):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=600 if stream else 300)


def post_json(url: str, body: dict) -> dict:
    with post(url, body) as response:
        return json.load(response)


def message(response: dict) -> dict:
    return (response.get("choices") or [{}])[0].get("message") or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    chat_url = f"{args.base_url}/chat/completions"
    common = {
        "model": args.model,
        "temperature": 0.0,
        "chat_template_kwargs": {"thinking": False},
    }
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "cases": {},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        report["updated_at"] = datetime.now(timezone.utc).isoformat()
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    atexit.register(save)

    basic = post_json(
        chat_url,
        common | {"messages": [{"role": "user", "content": "Reply exactly: QUAL_OK"}], "max_tokens": 24},
    )
    basic_text = str(message(basic).get("content") or "")
    report["cases"]["completion"] = {"ok": "QUAL_OK" in basic_text, "response": basic}

    for effort in ("low", "high", "max"):
        reasoning = post_json(
            chat_url,
            {
                **common,
                "messages": [{"role": "user", "content": "What is 2+2? End with the digit 4."}],
                "max_tokens": 96,
                "chat_template_kwargs": {"thinking": True, "reasoning_effort": effort},
            },
        )
        reasoning_message = message(reasoning)
        reasoning_text = str(reasoning_message.get("reasoning_content") or "") + str(
            reasoning_message.get("content") or ""
        )
        report["cases"][f"reasoning_{effort}"] = {
            "ok": bool(reasoning_text.strip()) and "4" in reasoning_text,
            "response": reasoning,
        }

    stream_body = common | {
        "messages": [{"role": "user", "content": "Reply exactly: STREAM_OK"}],
        "max_tokens": 24,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    chunks: list[str] = []
    saw_done = False
    with post(chat_url, stream_body, stream=True) as response:
        for raw in response:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                saw_done = True
            elif line.startswith("data: "):
                event = json.loads(line[6:])
                choices = event.get("choices") or []
                if choices:
                    chunks.append(str((choices[0].get("delta") or {}).get("content") or ""))
    stream_text = "".join(chunks)
    report["cases"]["streaming"] = {
        "ok": saw_done and "STREAM_OK" in stream_text,
        "saw_done": saw_done,
        "response": stream_text,
    }

    cancel_body = common | {
        "messages": [{"role": "user", "content": "Write a long numbered list of integers."}],
        "max_tokens": 4096,
        "stream": True,
    }
    with post(chat_url, cancel_body, stream=True) as response:
        response.readline()
    time.sleep(1)
    after_cancel = post_json(
        chat_url,
        common | {"messages": [{"role": "user", "content": "Reply exactly: CANCEL_OK"}], "max_tokens": 24},
    )
    after_cancel_text = str(message(after_cancel).get("content") or "")
    report["cases"]["cancellation_recovery"] = {
        "ok": "CANCEL_OK" in after_cancel_text,
        "response": after_cancel,
    }

    structured = post_json(
        chat_url,
        common
        | {
            "messages": [{"role": "user", "content": "Return status ready and integer count 731."}],
            "max_tokens": 64,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "qualification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "const": "ready"},
                            "count": {"type": "integer", "const": 731},
                        },
                        "required": ["status", "count"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )
    structured_text = str(message(structured).get("content") or "")
    try:
        structured_value = json.loads(structured_text)
    except json.JSONDecodeError:
        structured_value = None
    report["cases"]["structured_output"] = {
        "ok": structured_value == {"status": "ready", "count": 731},
        "response": structured,
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_record",
                "description": "Look up one record by code.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    tool_user = {"role": "user", "content": "Call lookup_record exactly once with code DS731. Do not answer directly."}
    tool_response = post_json(
        chat_url,
        common | {"messages": [tool_user], "tools": tools, "tool_choice": "auto", "max_tokens": 128},
    )
    assistant = message(tool_response)
    calls = assistant.get("tool_calls") or []
    tool_ok = len(calls) == 1 and ((calls[0].get("function") or {}).get("name") == "lookup_record")
    arguments = {}
    if tool_ok:
        try:
            arguments = json.loads((calls[0].get("function") or {}).get("arguments") or "{}")
        except json.JSONDecodeError:
            tool_ok = False
    tool_ok = tool_ok and arguments == {"code": "DS731"}
    report["cases"]["tool_call"] = {"ok": tool_ok, "response": tool_response}

    replay_ok = False
    replay_response = None
    if calls:
        replay_messages = [
            tool_user,
            {key: value for key, value in assistant.items() if key in {"role", "content", "tool_calls", "reasoning_content"}},
            {
                "role": "tool",
                "tool_call_id": calls[0].get("id"),
                "content": '{"value":"REPLAY_OK"}',
            },
        ]
        replay_response = post_json(
            chat_url,
            common | {"messages": replay_messages, "tools": tools, "max_tokens": 64},
        )
        replay_ok = "REPLAY_OK" in str(message(replay_response).get("content") or "")
    report["cases"]["tool_replay"] = {"ok": replay_ok, "response": replay_response}

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["ok"] = all(case["ok"] for case in report["cases"].values())
    save()
    print(json.dumps({name: case["ok"] for name, case in report["cases"].items()}, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
