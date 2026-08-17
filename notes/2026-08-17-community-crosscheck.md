# 2026-08-17 community cross-check

Question: is the live rc7 Arm B + 1M winner still the best 2x Spark
0731 serve, and would NVIDIA official NVFP4 free RAM for a second model?

## Verdict

Stay on official `deepseek-ai/DeepSeek-V4-Flash-0731` + rc7 + current
knobs. Do not switch to `nvidia/DeepSeek-V4-Flash-NVFP4` for this
cluster. Coexist (Music3 / a second LLM) is a KV-arena cut, not a
weight-quant cut.

## Why the NVIDIA NVFP4 card is the wrong swap

- `nvidia/DeepSeek-V4-Flash-NVFP4` is ModelOpt of the **preview**
  `deepseek-ai/DeepSeek-V4-Flash`, released 2026-05-28, last modified
  2026-06-15. It is not 0731.
- NVIDIA's own card verifies vLLM on **GB300 TP=4** / B200, not GB10.
- Sister card `nvidia/DeepSeek-V4-Flash-nvfp4-DSpark` is the same
  preview backbone plus a DSpark head.
- Official 0731 experts are already MXFP4 (`I8` packed). Community
  0731-NVFP4 conversions (MJPansa and others) keep the same 304B
  indexed payload as official 0731. That does not free the 26.3 GiB
  we spent on KV.
- vLLM #52447 (2026-08-16): NVFP4 0731 + DSpark + `nvfp4` KV is **not**
  first-class on stock 0.27.1 SM121 (DeepGEMM SF error). jasl/PR #41834
  can load NVFP4 but the draft loader dies on missing scale tensors.

## What the live public recipes actually are

- Mia / Anemll 0.1.1: official 0731, 1M, seqs 6, batch 8192, **Anemll
  0.25 image**, `nvfp4_ds_mla`, GMU ~0.835, ~2.49M KV. Still the most
  copied 0731 recipe.
- Tony 0731: Patch 4 + `nvfp4_ds_mla` + k=5. Peak 78 tok/s is a
  count-to-300 warm C1. Typical ~55. That `nvfp4_ds_mla` spelling is
  still an FP8-record alias on this hardware class.
- Reederey kit (updated 2026-08-16): official 0731, vLLM main
  `@48bada6` (0.27-content) + gx10 overlay, thinking default **on**,
  C1 ~34 / C8 agg ~88, KV pin 19.85 GiB / 3.02M tokens. Closest
  "maintained 0.27" public kit. Different harness than our exact
  issue27 numbers.
- Stock vLLM `v0.27.1` (2026-08-11) is still not a Spark DS4 path.
  PR #41834 remains open.

## Our winner vs those recipes

Live 2026-08-17: rc7, official 0731, 1M, seqs 6, batch 8192, GMU 0.84,
`fp8_ds_mla`, KV 26.3 GiB / 2.74M, hook #31 off, thinking off,
suppress-stops on, long-prefill 1024.

We already measured the things the public READMEs still leave on:

- `#31` CPU hook is a 1.8x tax at 32k x 6
- `nvfp4_ds_mla` long-ctx decode collapse
- exact-ceiling builders 400; leave ~8k headroom
- 1.04M retrieval PASS, soak 1000/1000, 3x restart fingerprint

Do not rebase onto Anemll 0.25 or stock 0.27.1 for a headline tok/s.

## Coexist

Tony H3 factory (not Music3): keep advertised 1M, shrink KV to ~10.3
GiB / GMU 0.78, leave 16-18 GiB/node for a Comfy H3 that evicts like a
24 GB card. Cutting `max_model_len` or GMU blindly can collapse usable
context to thousands of tokens.

Our live 26.3 GiB arena leaves ~0.6 GiB. Official NVFP4 does not create
that headroom. The lever is `KV_CACHE_MEMORY`.

A 0731 vision sidecar (FlyCockpit encoder ~865 MB + projector ~40 MB)
is cheap once KV is cut. MiniMax Music 3 as a second resident model is
not; measure its footprint before promising it next to this winner.

## Sources pulled 2026-08-17

- https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4
- https://huggingface.co/nvidia/DeepSeek-V4-Flash-nvfp4-DSpark
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- https://huggingface.co/MJPansa/DeepSeek-V4-Flash-0731-NVFP4
- https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
- https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
- https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark
- https://github.com/tonyd2wild/ds4-h3-video-gen-factory
- https://github.com/Reederey87/dgx-spark-2x-deepseek-v4-flash
- https://github.com/vllm-project/vllm/releases/tag/v0.27.1
- https://github.com/vllm-project/vllm/pull/41834
- https://github.com/vllm-project/vllm/issues/52447
- https://github.com/vllm-project/vllm/issues/51758
- https://forums.developer.nvidia.com/t/deepseek-v4-flash-with-vision/379212
