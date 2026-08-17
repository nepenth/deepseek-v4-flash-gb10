#!/usr/bin/env python3
"""Cut vLLM shm SpinCondition busy_loop_s from 1s to 2ms on GB10 TP>=2.

nacyot 2026-08-12: default 1s never expires during decode, so EngineCore /
Worker_TP spin 3-4 P-cores at full clock. 2ms uses the existing zmq sleep
path. Throughput unchanged; CPU/SoC drop.

Opt-out: DSPARK_SKIP_BUSY_LOOP_HOTFIX=1
"""
from __future__ import annotations

from pathlib import Path

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/shm_broadcast.py")
OLD = "        busy_loop_s: float = 1,"
NEW = "        busy_loop_s: float = 0.002,  # [gb10-busy-loop-2ms]"


def main() -> None:
    src = P.read_text()
    if NEW in src or "busy_loop_s: float = 0.002" in src:
        print(f"[busy-loop-hotfix] already applied ({P})")
        return
    if OLD not in src:
        raise SystemExit(f"[busy-loop-hotfix] anchor not found in {P}")
    P.write_text(src.replace(OLD, NEW, 1))
    print(f"[busy-loop-hotfix] patched {P} 1s -> 2ms")


if __name__ == "__main__":
    main()
