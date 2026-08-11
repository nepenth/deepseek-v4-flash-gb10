# SPDX-License-Identifier: MIT
from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtime"


def load_lock() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in (RUNTIME / "upstream.lock").read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


class RuntimeContractTests(unittest.TestCase):
    def test_release_and_gb10_pins(self) -> None:
        lock = load_lock()
        self.assertEqual(lock["VLLM_TAG"], "v0.27.1")
        self.assertEqual(lock["VLLM_VERSION"], "0.27.1")
        self.assertRegex(lock["VLLM_COMMIT"], r"^[0-9a-f]{40}$")
        self.assertEqual(lock["TORCH_CUDA_ARCH_LIST"], "12.1a")
        self.assertEqual(lock["FLASHINFER_VERSION"], "0.6.16.post3")

    def test_patch_series_is_complete_and_ordered(self) -> None:
        entries = [
            line.strip()
            for line in (RUNTIME / "patches/vllm/series").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        self.assertEqual(len(entries), 8)
        self.assertEqual(len(entries), len(set(entries)))
        for number, entry in enumerate(entries, 1):
            self.assertTrue(entry.startswith(f"{number:04d}-"), entry)
            patch = RUNTIME / "patches/vllm" / entry
            self.assertTrue(patch.is_file(), patch)
            self.assertRegex(patch.read_text()[:200], r"^From [0-9a-f]{40} ")

    def test_runtime_does_not_overlay_or_downgrade_upstream(self) -> None:
        build_script = (RUNTIME / "scripts/build-image.sh").read_text()
        dockerfile = (RUNTIME / "docker/Dockerfile.runtime").read_text()
        self.assertNotIn("overlay/vllm", build_script)
        self.assertNotIn("pip install", dockerfile)
        self.assertIn("torch_cuda_arch_list=$TORCH_CUDA_ARCH_LIST", build_script)

    def test_compose_uses_canonical_sparse_mla_dtype(self) -> None:
        compose = (ROOT / "docker-compose.dspark.yml").read_text()
        self.assertIn("KV_CACHE_DTYPE:-fp8_ds_mla", compose)
        self.assertNotIn("--kv-cache-dtype nvfp4_ds_mla", compose)
        self.assertIn("VLLM_ALLOW_SPEC_DEC_SAME_STEP_PREFIX_HIT:-2", compose)

    def test_nvfp4_status_is_not_overclaimed(self) -> None:
        research = (ROOT / "docs/NVFP4_DS_MLA.md").read_text()
        self.assertRegex(research, re.compile(r"not a packed 4-bit KV cache", re.I))
        self.assertIn("368-byte", research)
        self.assertIn("experimental research", research)


if __name__ == "__main__":
    unittest.main()
