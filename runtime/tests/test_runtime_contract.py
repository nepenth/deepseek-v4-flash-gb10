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
        self.assertEqual(lock["BUILD_MAX_JOBS"], "16")
        self.assertEqual(lock["BUILD_NVCC_THREADS"], "8")
        self.assertEqual(lock["FLASHINFER_VERSION"], "0.6.16.post3")
        self.assertEqual(
            lock["FLASHINFER_DSV4_SM120_COMMIT"],
            "24d7dfb2639083c5a4d418881099421fc800b7bb",
        )
        self.assertEqual(lock["FLASHINFER_DSV4_SM120_TOPK"], "192,256")
        self.assertRegex(
            lock["FLASHINFER_DSV4_SM120_PATCH_SHA256"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(lock["CUTLASS_DSL_VERSION"], "4.6.2")
        self.assertEqual(lock["QUACK_VERSION"], "0.6.4")
        self.assertEqual(
            lock["DEEPGEMM_COMMIT"],
            "2fd67329ec2942f65ba35d561256ab6ed3b903cb",
        )

    def test_patch_series_is_complete_and_ordered(self) -> None:
        entries = [
            line.strip()
            for line in (RUNTIME / "patches/vllm/series").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        self.assertEqual(len(entries), 28)
        self.assertEqual(len(entries), len(set(entries)))
        for number, entry in enumerate(entries, 1):
            self.assertTrue(entry.startswith(f"{number:04d}-"), entry)
            patch = RUNTIME / "patches/vllm" / entry
            self.assertTrue(patch.is_file(), patch)
            self.assertRegex(patch.read_text()[:200], r"^From [0-9a-f]{40} ")

    def test_runtime_uses_a_narrow_verified_upstream_overlay(self) -> None:
        build_script = (RUNTIME / "scripts/build-image.sh").read_text()
        dockerfile = (RUNTIME / "docker/Dockerfile.runtime").read_text()
        self.assertNotIn("overlay/vllm", build_script)
        self.assertNotIn("pip install", dockerfile)
        self.assertIn("torch_cuda_arch_list=$TORCH_CUDA_ARCH_LIST", build_script)
        self.assertIn("max_jobs=$BUILD_MAX_JOBS", build_script)
        self.assertIn("nvcc_threads=$BUILD_NVCC_THREADS", build_script)
        self.assertIn("FLASHINFER_DSV4_SM120_COMMIT", build_script)
        self.assertIn("0001-sm120-dsv4-192-256-topk.patch", dockerfile)
        self.assertIn("flashinfer_jit_cache/jit_cache/sparse_mla_sm120", dockerfile)
        self.assertIn("FLASHINFER_DSV4_SM120_PATCH_SHA256", dockerfile)
        legacy_start = (RUNTIME / "scripts/start-node.sh").read_text()
        self.assertIn("ALLOW_LEGACY_RUNTIME_COMPOSE", legacy_start)

    def test_compose_uses_canonical_sparse_mla_dtype(self) -> None:
        compose = (ROOT / "docker-compose.dspark.yml").read_text()
        self.assertIn("KV_CACHE_DTYPE:-fp8_ds_mla", compose)
        self.assertIn("KV_BLOCK_SIZE:-256", compose)
        self.assertNotIn("--kv-cache-dtype nvfp4_ds_mla", compose)
        self.assertIn("VLLM_ALLOW_SPEC_DEC_SAME_STEP_PREFIX_HIT:-2", compose)
        self.assertIn(
            "DSPARK_REVISION:-9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
            compose,
        )

    def test_nvfp4_status_is_not_overclaimed(self) -> None:
        research = (ROOT / "docs/NVFP4_DS_MLA.md").read_text()
        self.assertRegex(research, re.compile(r"not a packed 4-bit KV cache", re.I))
        self.assertIn("368-byte", research)
        self.assertIn("448-dimension", research)
        self.assertIn("512+64", research)
        self.assertIn("experimental research", research)

    def test_cluster_canary_preserves_first_light_contract(self) -> None:
        profile = (ROOT / "cluster/profiles/deepseek-v4-flash-0731-v0271-canary.conf").read_text()
        example = (ROOT / "cluster/environments/deepseek-v4-flash-0731-v0271-canary.env.example").read_text()
        compose = (ROOT / "docker-compose.dspark.yml").read_text()
        start = (ROOT / "start-deepseek-v4-flash-dspark.sh").read_text()
        self.assertIn("RESTART_POLICY=no", profile)
        self.assertIn("ROLLBACK_PROFILE=deepseek-v4-flash-0731-dspark", profile)
        legacy_profile = (ROOT / "cluster/profiles/deepseek-v4-flash-0731-dspark.conf").read_text()
        self.assertIn(
            "vllm-profile-runner start deepseek-v4-flash-0731-dspark", legacy_profile
        )
        self.assertIn("MAX_MODEL_LEN=393216", example)
        self.assertIn("MAX_NUM_SEQS=6", example)
        self.assertIn("MAX_NUM_BATCHED_TOKENS=4096", example)
        self.assertIn("GPU_MEMORY_UTILIZATION_TEXT=0.78", example)
        self.assertIn("MTP_NUM_TOKENS=5", example)
        self.assertIn("KV_CACHE_DTYPE=fp8_ds_mla", example)
        self.assertIn("KV_BLOCK_SIZE=256", example)
        self.assertIn("MOE_BACKEND=deep_gemm", example)
        self.assertIn("v0.27.1-gb10-rc4", profile)
        self.assertIn("v0.27.1-gb10-rc4", example)
        self.assertIn('MOE_BACKEND: "${MOE_BACKEND:-flashinfer_b12x}"', compose)
        self.assertIn("--moe-backend $${MOE_BACKEND:-flashinfer_b12x}", compose)
        self.assertIn("MOE_BACKEND=\"$MOE_BACKEND\"", start)
        self.assertIn("MOE_BACKEND='%s'", start)
        self.assertIn("VLLM_SWITCH_IMAGE=$DOCKER_IMAGE", profile)
        self.assertIn("VLLM_SWITCH_KV_BLOCK_SIZE=256", profile)
        self.assertIn("VLLM_SWITCH_FLASHINFER_WORKSPACE_BASE", profile)
        self.assertIn("VLLM_SWITCH_TRITON_CACHE_DIR", profile)
        self.assertIn('DSPARK_VLLM_IMAGE="$VLLM_SWITCH_IMAGE"', start)
        self.assertIn('KV_BLOCK_SIZE="$VLLM_SWITCH_KV_BLOCK_SIZE"', start)
        self.assertIn(
            'FLASHINFER_WORKSPACE_BASE="$VLLM_SWITCH_FLASHINFER_WORKSPACE_BASE"',
            start,
        )
        self.assertIn(
            'TRITON_CACHE_DIR="$VLLM_SWITCH_TRITON_CACHE_DIR"',
            start,
        )
        self.assertIn("requires KV_BLOCK_SIZE=256", start)
        self.assertIn("KV_BLOCK_SIZE=\"$KV_BLOCK_SIZE\"", start)
        self.assertIn("DSPARK_VLLM_IMAGE=\"$DSPARK_VLLM_IMAGE\"", start)
        self.assertIn("FLASHINFER_WORKSPACE_BASE", example)
        self.assertIn("TRITON_CACHE_DIR", example)
        self.assertIn('TRITON_CACHE_DIR: "${TRITON_CACHE_DIR:-', compose)
        self.assertIn("DSPARK_MODEL_HOST", compose)

    def test_deepgemm_router_counter_warmup_patch_is_included(self) -> None:
        patch = (
            RUNTIME / "patches/vllm/0028-warm-deepgemm-router-token-counter.patch"
        ).read_text()
        self.assertIn("_warmup_fused_moe_token_counter", patch)
        self.assertIn("DeepGemmFP4Experts", patch)
        self.assertIn("_FUSED_MOE_TOKEN_COUNTER_BLOCK_SIZES", patch)
        self.assertIn("(1, 2, 3, 6, 11, 22, 43, 86)", patch)

    def test_sm120_sparse_mla_small_decode_patch_is_included(self) -> None:
        patch = (
            RUNTIME / "patches/vllm/0027-dsv4-sm120-direct-small-decode.patch"
        ).read_text()
        self.assertIn("_SparseMLAPagedAttentionRunner", patch)
        self.assertIn("_reserve_sm120_decode_workspace", patch)
        self.assertIn("mid_out=mid_out", patch)
        self.assertIn("_FLASHINFER_SM120_DECODE_MAX_TOKENS = 64", patch)
        self.assertNotIn("kv_cache = kv_cache.view(-1, 64, 1, 584)", patch)

    def test_flashinfer_sm120_topk_overlay_is_complete(self) -> None:
        patch = (
            RUNTIME / "patches/flashinfer/0001-sm120-dsv4-192-256-topk.patch"
        ).read_text()
        self.assertIn("DSV4_DISPATCH(32, 256)", patch)
        self.assertIn("DSV4_DISPATCH(32, 192)", patch)
        self.assertIn("(32, 256)", patch)
        self.assertIn("SM120 sparse-MLA has no decode kernel", patch)

    def test_startup_fails_when_a_rank_exits_before_readiness(self) -> None:
        start = (ROOT / "start-deepseek-v4-flash-dspark.sh").read_text()
        self.assertIn("rank_exited_before_ready()", start)
        self.assertIn("ps --status exited -q vllm-dspark", start)
        self.assertIn("A vLLM rank exited before the API became ready.", start)

    def test_deploy_warns_about_stale_installed_control_plane(self) -> None:
        deploy = (ROOT / "cluster/deploy-to-sparks.sh").read_text()
        self.assertIn("installed_control_plane_stale", deploy)
        self.assertIn("install-control-plane.sh --install", deploy)
        self.assertIn("does not install", deploy)

    def test_tracked_cluster_examples_do_not_contain_private_addresses(self) -> None:
        tracked = [
            path
            for path in (ROOT / "cluster").rglob("*")
            if path.is_file() and "tests" not in path.parts and "__pycache__" not in path.parts
        ]
        tracked.append(ROOT / "docs/CLUSTER_CONTROL_PLANE.md")
        private_markers = (
            ".".join(("192", "168", "100", "")),
            ".".join(("10", "0", "10", "")),
            "gx10-" + "97f1",
            "gx10-" + "9dbe",
        )
        for path in tracked:
            text = path.read_text()
            for marker in private_markers:
                self.assertNotIn(marker, text, f"{marker} leaked into {path}")


if __name__ == "__main__":
    unittest.main()
