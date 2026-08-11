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

Upstream references:

- <https://github.com/vllm-project/vllm/pull/42359>
- <https://github.com/vllm-project/vllm/pull/41834>
- <https://github.com/jasl/vllm/tree/sm120-pr-41834-stable-preview-20260809>

The patches retain their original Apache-2.0 provenance and commit metadata.
