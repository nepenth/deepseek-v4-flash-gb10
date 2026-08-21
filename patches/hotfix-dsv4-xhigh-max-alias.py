#!/usr/bin/env python3
"""Alias reasoning_effort=xhigh -> max (Tony 2026-08-20 tokenizer overlay)."""
from __future__ import annotations

from pathlib import Path

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4.py")
OLD = """            elif reasoning_effort == "max":
                reasoning_effort = "max"
"""
NEW = """            elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
"""


def main() -> None:
    src = P.read_text()
    if 'reasoning_effort in ("max", "xhigh")' in src:
        print(f"[xhigh-alias] already applied ({P})")
        return
    if OLD not in src:
        raise SystemExit(f"[xhigh-alias] anchor not found in {P}")
    P.write_text(src.replace(OLD, NEW, 1))
    print(f"[xhigh-alias] patched {P}")


if __name__ == "__main__":
    main()
