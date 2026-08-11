# vLLM 0.27.1 patch series

`series` is applied in order to the commit pinned in `../../upstream.lock`.
Do not add a patch without verifying the clean source contract and recording
its upstream provenance here.

| Patch | Source | Purpose |
| --- | --- | --- |
| 0001 | vLLM `774348619` / PR #50580 | DeepSeek V4 0731 reasoning-effort prompts and mappings |
| 0002 | vLLM `8bcc916a9` / PR #51727 | tokenizer vocabulary-size crash fix |
| 0003 | vLLM `d40c3e3c0` / PR #50693 | DSpark warmup without a sparse index buffer |
| 0004 | vLLM `79c865b83` / PR #51430 | narrower DeepSeek V4 eager CUDA graph region |
| 0005 | jasl/vLLM `4cdb3f473` | port vLLM PR #42359 to the current cache-manager design |
| 0006 | jasl/vLLM `786582103` | cover every DeepSeek V4 cache group |
| 0007 | jasl/vLLM `f2cac6523` | preserve an explicit off switch and test semantics |
| 0008 | jasl/vLLM `15dc5af4d` | enable the guard for prefix-cached speculative serving |
| 0009 | vLLM PR #50796 | restore DeepGEMM SM120 layout support while retaining SITU |
| 0010 | jasl/vLLM `b8edada26` / vLLM PR #51318 | make C128A decode row stride capture-stable |
| 0011 | vLLM `47a4e410b` / PR #50183 | make V2 rejection-sampler argmax NaN-safe |
| 0012 | vLLM `d6af803f4` / PR #50276 | zero packed KV blocks using physical stride |
| 0013 | jasl/vLLM `264942766` | invalidate drafter metadata after mixed-batch layout rewrites |
| 0014 | jasl/vLLM `d7bddfeff` | bound tile-local sampled token IDs to the vocabulary |
| 0015 | jasl/vLLM `4eedf4876` | execute config-gated mHC TileLang warmup on CUDA |
| 0016 | vLLM `355a338b8` / PR #51602 | initialize the DSpark parallel drafting token correctly |
| 0017 | vLLM `d608dfabf` / PR #51438 | reserve MRV2 lookahead blocks during warmup |
| 0018 | vLLM `199644d41` / PR #50906 | bound sparse masked-MHA workspace and fall back safely |
| 0019 | vLLM `9e6be4a72` / PR #50365 | remove sparse-MLA index-remap atomic contention |
| 0020 | vLLM `b38e111d3` / PR #50613 | schedule chunked MLA context per request |
| 0021 | vLLM `0914ed2e8` / PR #51725 | budget speculative input slots adaptively |
| 0022 | vLLM `789c4f905` / PR #51566 | pair CUTLASS DSL 4.6.2 with QuACK 0.6.4 |
| 0023 | vLLM `3f142bd85` / PR #51296 | align parser thinking default with the tokenizer |

Upstream references:

- <https://github.com/vllm-project/vllm/pull/42359>
- <https://github.com/vllm-project/vllm/pull/41834>
- <https://github.com/vllm-project/vllm/pull/50796>
- <https://github.com/vllm-project/vllm/pull/51318>
- <https://github.com/jasl/vllm/tree/sm120-pr-41834-stable-preview-20260809>

The patches retain their original Apache-2.0 provenance and commit metadata.
