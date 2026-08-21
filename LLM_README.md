# LLM_README — DeepSeek-V4-Flash-0731 GB10 recipe

Ingest this file first. It is the machine-oriented contract for this
repository. Humans should read `README.md`. Structured live knobs:
`project-status.json` → `live`. Material decisions: `PROJECT-DECISIONS.md`.

Do not invent cluster hostnames, IPs, DNS, or operator usernames. Those are
intentionally absent. Fill them in an untracked env file.

---

## 1. Purpose

Public recipe to **serve** `deepseek-ai/DeepSeek-V4-Flash-0731` on **two**
NVIDIA DGX Spark GB10 nodes (SM121, ARM64, TP=2, DSpark speculative decode).

This is a text-serving lane. `ENABLE_VL_SIDECAR=0` is the validated path.

---

## 2. Canonical live contract

Copy these values. They were proven from `docker inspect` `Config.Cmd` on a
running two-node serve, not from leftover profile files.

```yaml
image: dspark-vllm-gb10:v0.27.1-gb10-rc7
vllm_tag: v0.27.1
vllm_commit: 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
checkpoint: deepseek-ai/DeepSeek-V4-Flash-0731
revision: 9e165c30e2704aec5d9d593cce3eebd58bbef1cb
served_model_names:
  - deepseek-ai/DeepSeek-V4-Flash-0731
  - deepseek-v4-flash-0731
topology: { tp: 2, pp: 1, backend: mp, ray: false, worker_first: true }
max_model_len: 1048576
max_num_seqs: 6
max_num_batched_tokens: 8192
gpu_memory_utilization: 0.84
kv_cache_dtype: fp8_ds_mla          # NOT packed NVFP4
kv_cache_memory_bytes: 28235618304  # 26.3 GiB → 2.74M tokens, 2.61× @1M
kv_block_size: 256
moe_backend: deep_gemm
dspark_k: 5                         # never 7
long_prefill_token_threshold: 1024
default_thinking: max
default_thinking_token_budget: unset  # empty; do not set 32768
skip_issue31_cpu_hook: true
hotfixes_on:
  - hotfix-dsv4-issue31-v2-thinking-budget-gpu.py  # Mia #48 port
  - hotfix-dsv4-issue55-tool-truncation.py
  - hotfix-dsv4-indexer-52492.py
  - hotfix-dsv4-suppress-stops-v0271.py
  - hotfix-gb10-busy-loop-2ms.py
  - hotfix-dsv4-xhigh-max-alias.py
fabric:
  nccl_ib_merge_nics: true
  nccl_ib_gid_index: unset
  jumbo_mtu: 9000
  gloo_tp_ifname: single            # not the dual-HCA list
execute_model_timeout_seconds: 1800
v2_runner: true                     # never VLLM_USE_V2_MODEL_RUNNER=0
```

Env template:
`cluster/environments/deepseek-v4-flash-0731-v0271-canary.env.example`

---

## 3. Why these knobs (root cause → fix)

| Symptom | Root cause | Fix |
|---|---|---|
| Weights load, then CUTLASS `scaled_mm` death | Stock vLLM / generic GB10 canary is not a DS4 SM121 path | Pin `v0.27.1-gb10-rc7` image + ordered patch series |
| Half decode speed on official 0731 vs preview | DSpark draft loader dropped shared-expert tensors | Keep k=5 + shared-expert path in the image |
| `content: null` with client `stop` | Stops match inside `<think>` (`Question:`) | `hotfix-dsv4-suppress-stops-v0271.py` (0.27.1 rewrite; do not bind-mount the 0.25 file) |
| 32k×6 ~24 tok/s, C1 still looks fine | #31 CPU thinking-budget host-scan | `DSPARK_SKIP_ISSUE31_HOTFIX=1` |
| thinking=max hangs / #39 cliff | Server omit-field `DEFAULT_THINKING_TOKEN_BUDGET` | GPU closer = Mia #48 port. Budget unset. |
| Greedy temp=0 ignores closer | V2 logits-processing gate | Keep `_requires_logits_processing` greedy gate in the #48 port |
| Packed-NVFP4 marketing | `nvfp4_ds_mla` was an FP8-record alias | Serve `fp8_ds_mla`. See `docs/NVFP4_DS_MLA.md` |
| ~100 Gb/s fabric on QSFP | GB10 enumerates **two** virtual HCAs; one is idle unless merged | List both, `NCCL_IB_MERGE_NICS=1`, GID **unset** |
| NCCL FATAL / one-NIC after GID pin | IPv4 RoCEv2 GID indexes disagree across members | Never pin a single `NCCL_IB_GID_INDEX` on dual-HCA |
| Env edit, serve unchanged | Same-profile `vllm-switch` does not recreate | `docker rm -f` both rank containers |
| Exact 1,048,576 prompt 400s | Chat template + needles overshoot by ~2 | Cap prompts ~8k under ceiling |
| Need RAM for coexist | Cutting `max_model_len` / GMU collapses usable context | Shrink `KV_CACHE_MEMORY`, keep advertised 1M |

---

## 4. Findings (do not mix protocols)

**P1 — exact-length think-off C1 / 32k×6 (2026-08-14 campaign)**
- Hook ON 32k×6: 24.5 tok/s. Hook OFF: 46.4 tok/s (1.05× spread, MTP 95%).
- 1.04M three-needle PASS, 20.7 min.
- Phase 5 11/11, encoding 4/4, 3× greedy restart identical, soak 1000/1000.

**P2 — isolated fabric (2026-08-21, serve down, nccl-tests 1 GiB all_reduce)**
- Single HCA MTU 1500: 12.53 GB/s busbw.
- Single HCA MTU 9000: 12.59 GB/s (jumbo alone is not the win).
- Dual-HCA merge, GID auto, MTU 9000: 23.55 GB/s.

**P3 — think-off ignore_eos after dual-HCA bounce (not P1)**
- 256 C1: 77.4 tok/s, TTFT 201 ms.
- 32k×6: 47.5 tok/s, spread 1.00×.
- KV pool unchanged: 2,740,813 tokens, 2.61× @1M.

**P4 — thinking=max spark-eval official + Grok-4.5 (2026-08-21)**
- Q 0.693, tools 1.000, coding 0.875, Grok 4.667/15.
- Prior thinking=max official (2026-08-19): Q 0.583, coding 0.125.
- Do not compare P4 Q to think-off spark-eval (~0.92). Coding hang-guard is a floor. Per-request `thinking_token_budget` is coding-only; do not set it as a server default.

---

## 5. Reproduce

```bash
cp cluster/environments/deepseek-v4-flash-0731-v0271-canary.env.example \
   cluster/environments/deepseek-v4-flash-0731-v0271-canary.env
cp .env.dspark.example .env.dspark
# fill WORKER_HOST MASTER_ADDR NCCL_IB_HCA NCCL_SOCKET_IFNAME VLLM_HOST_IP (untracked)

./prepare-dspark-model-cache.sh --official --yes
./build-dspark-vllm-runtime.sh          # if the rc7 image is not already present
./start-deepseek-v4-flash-dspark.sh     # worker first
curl -fsS http://127.0.0.1:8000/v1/models
./smoke-deepseek-v4-flash-dspark.sh
```

Verify argv:

```text
docker inspect <container> --format '{{json .Config.Cmd}}'
# require: --max-model-len 1048576, --max-num-seqs 6,
#          --gpu-memory-utilization 0.84, --kv-cache-memory 28235618304,
#          --long-prefill-token-threshold 1024
```

`start-deepseek-v4-flash-dspark.sh` must scp **every** bind-mounted hotfix to
the worker. A skipped hotfix file must still exist if compose mounts it.

---

## 6. Forbidden (do not resurrect)

- Stock vLLM 0.27.x / eugr generic GB10 canary as a Spark DS4 path
- Anemll 0.25 rebase of suppress-stops (missing `import sys` + `TokenizersBackend` factory)
- `nvfp4_ds_mla` as packed NVFP4
- DSpark k=7
- `VLLM_USE_V2_MODEL_RUNNER=0`
- `DEFAULT_THINKING_TOKEN_BUDGET=<n>` as the thinking=max closer
- Inventing a thinking-budget closer instead of porting Mia #48
- Raising GMU above 0.84
- Pinning `NCCL_IB_GID_INDEX` on a comma-list `NCCL_IB_HCA`
- Cutting `max_model_len` to free RAM
- Baking `runtime/patches/vllm/0029-warm-dspark-probabilistic-rejection-helpers.patch` without A/B (in-tree, **out of** `runtime/patches/vllm/series`)
- Using `decode-bench.py` as the 32k×6 production number (use exact-length issue27)
- Mixing P1/P3/P4 numbers in one table

---

## 7. File map

| Path | Role |
|---|---|
| `README.md` | Human recipe |
| `LLM_README.md` | This file |
| `project-status.json` | Machine state; `live` must match inspect |
| `PROJECT-DECISIONS.md` | Why / rejected |
| `cluster/environments/deepseek-v4-flash-0731-v0271-canary.env.example` | Canonical knobs |
| `.env.dspark.example` | Compose-launcher env (same knobs, placeholder hosts) |
| `docker-compose.dspark.yml` | MP launch + hotfix mounts |
| `start-deepseek-v4-flash-dspark.sh` | Worker-first; scp hotfixes |
| `patches/hotfix-*.py` | Bind-mount hotfixes, no rebuild |
| `runtime/` | Pinned source + patch series + image build |
| `docs/DEEPSEEK_V4_FLASH_0731.md` | Serving contract |
| `docs/NVFP4_DS_MLA.md` | Why fp8, not packed NVFP4 |
| `docs/CAMPAIGN_2026-08-14.md` | P1 evidence |
| `notes/2026-08-21-dual-hca.md` | P2/P3 evidence |
| `notes/2026-08-21-thinkmax-eval.md` | P4 evidence |
| `docs/GB10_DSV4_HANDOFF_2026-08-12.md` | Image/A/B **history**; envelope numbers in it are not live |
| `docs/SETUP.md` | vllm-switch operator path |

Historical / not the live path: `docs/ENVS.md`, `docs/PATCHES.md`,
`docs/OPS_400K_KVMEM_2026-08-13.md`, `recipe/`, `vllm_patch_gb10/`,
`plugins/dspark_vision_mcp/`.

---

## 8. Drift traps

1. **Compose YAML vs Cmd.** Flags after `vllm serve` can be dropped if the
   compose command is split across YAML newlines incorrectly. Always inspect
   `Config.Cmd`.
2. **Env vs Cmd.** Head env may still list empty `NCCL_IB_GID_INDEX=`. The
   start snippet must `unset` empty GID / merge vars. Empty-string is not unset.
3. **Leftover profiles.** `deepseek-v4-flash-0731-dspark.conf` / Stage-C 393k
   / 400k notes are historical. Do not infer live argv from them.
4. **`.env.dspark.example` vs canary.env.example.** Canary env is the winner.
   If they disagree, canary wins; fix the other file.
5. **Public tree.** No private DNS, no node short-names, no RFC1918
   inventory ranges, no operator home paths, no tokens. Gate:
   `runtime/scripts/check-private-data.sh`.
6. **Clients that must not think** (memory extractors, structured JSON in
   content) send `chat_template_kwargs.thinking=false`. Server default is max.

---

## 9. Attribution

Independent recipe, not a GitHub fork. Lineage and licenses: `CREDITS.md`.
Hotfix ideas we ported rather than invented: MiaAI-Lab #48 / #55 / #42
(suppress-stops) and related GB10 community work. We keep k=5, `fp8_ds_mla`,
and skip #31 on evidence in this tree.
