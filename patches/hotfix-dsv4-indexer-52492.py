#!/usr/bin/env python3
"""Port of vLLM #52492: do not bake the short-context indexer shortcut.

`DeepseekV4Indexer.forward` skips learned indexer scoring when
`max_seq_len / compress_ratio <= topk`. Breakable (and some piecewise)
CUDA-graph capture uses short dummy metadata, so that branch can be
baked into the replay graph. Long cached prefixes then attend only the
first ~topk candidates.

This adds `and not torch.cuda.is_current_stream_capturing()` to the
shortcut. Capture always takes the full scoring path.

Usage:
  python3 hotfix-dsv4-indexer-52492.py
  python3 hotfix-dsv4-indexer-52492.py /path/to/vllm
"""
from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARK = "# [issue52492-hotfix] keep indexer scoring in captured graphs"
REL = "models/deepseek_v4/attention.py"

OLD = """            if indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens:
                # candidates num smaller than topk, every candidate is selected
                # but we still need to build k cache
"""

NEW = f"""            if (
                indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens
                and not torch.cuda.is_current_stream_capturing()  # {MARK}
            ):
                # candidates num smaller than topk, every candidate is selected
                # but we still need to build k cache
"""


def main() -> int:
    root = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--status" else (
        Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "--status" else DEFAULT_VLLM
    )
    path = root / REL
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        applied = path.is_file() and MARK in path.read_text(encoding="utf-8")
        print("issue52492 indexer capture guard :", "APPLIED" if applied else "NOT APPLIED")
        return 0
    if not path.is_file():
        print(f"[FAIL] missing {path}", file=sys.stderr)
        return 1
    source = path.read_text(encoding="utf-8")
    if NEW in source:
        print("[issue52492-hotfix] already applied")
        return 0
    if OLD not in source:
        print(f"[FAIL] anchor not found in {path}", file=sys.stderr)
        return 1
    path.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"[issue52492-hotfix] patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
