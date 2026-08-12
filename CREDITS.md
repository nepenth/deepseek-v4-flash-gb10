# Credits and Provenance

This is an independent repository. It is not a GitHub fork, but it preserves
the attribution, license notices, and source lineage of the public work it
builds upon. Do not remove upstream license headers or represent downstream
patches as upstream releases.

## Primary Foundations

| Project | Contribution to this repository |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | Serving engine, DeepSeek V4 integration, patch base, and runtime APIs. |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | Sparse MLA kernels and the upstream DSV4 192/256 dispatch implementation used as a narrow overlay. |
| [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | DeepSeek V4 Flash 0731 checkpoint and DSpark model architecture. |
| NVIDIA CUDA, NCCL, and Blackwell tooling | GB10/SM121 compute, communication, and build substrate. |
| DeepGEMM | MXFP4 routed-expert execution used by the candidate. |

## DSpark and DGX Spark Lineage

- [Rafael Caricio's DSpark vLLM work](https://github.com/rafaelcaricio/vllm/pull/1)
  and [deployment work](https://github.com/rafaelcaricio/spark_vllm_docker/pull/1)
  informed the original DSpark integration and deployment shape.
- [Fraser Price's DSpark runtime work](https://github.com/fraserprice/dspark-vllm)
  and the associated public model research informed earlier DeepSeek V4 Flash
  serving paths.
- [MiaAI-Lab's two-node DGX Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
  supplied the original two-node packaging, launch lineage, and historical
  qualification references from which this repository began.
- [Anemll's dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)
  informed the imported GB10 runtime/build lineage.
- [drowzeys / Keys concurrency work](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash)
  established important DSpark concurrency findings in historical overlays.
- [TonyD2Wild's DGX Spark recipe work](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
  informed earlier launch and long-context investigations.

## Research References

The true packed-NVFP4 cache discussion references public work from
[vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet) and
[b12x](https://github.com/local-inference-lab/b12x). These are research
references only. Their code is not used as a production build input here
because their packed 512+64 MLA layouts are not DeepSeek V4's 448+64 sparse
MLA geometry.

The hardware-focused patch selection also references vLLM
[PR #41834](https://github.com/vllm-project/vllm/pull/41834),
[PR #42359](https://github.com/vllm-project/vllm/pull/42359),
[PR #50796](https://github.com/vllm-project/vllm/pull/50796),
[PR #51318](https://github.com/vllm-project/vllm/pull/51318), and FlashInfer
[PR #4380](https://github.com/flashinfer-ai/flashinfer/pull/4380).

## License Notes

Repository-specific scripts and documentation are MIT-licensed under
[`LICENSE`](LICENSE). vLLM-derived files, the vLLM patch series, and other
upstream-derived code retain their Apache-2.0 or source-specific licensing.
Model weights, CUDA/NCCL, FlashInfer, TileLang, Triton, and base images remain
subject to their respective upstream licenses and terms. See the license and
notice files in the relevant imported source trees before redistribution.
