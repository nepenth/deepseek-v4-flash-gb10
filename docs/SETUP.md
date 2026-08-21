# Setup: Two-Node GB10 Runtime

This guide describes the current standalone runtime. It deliberately does not
embed node names, addresses, fabric names, checkpoint paths, or credentials.
Keep those values in the ignored deployed environment file and private
operation records.

## Prerequisites

- Two NVIDIA DGX Spark GB10 nodes with SM121-capable CUDA toolchain.
- A working tensor-parallel network path between both nodes.
- The same local DeepSeek V4 Flash 0731 checkpoint available on both nodes.
- Docker and Compose available on both nodes.
- Passwordless operator access appropriate for the local deployment policy.
- Sufficient free Docker-build storage; preserve validated build layers unless
  storage policy requires a documented cleanup.

## Source Contract

The source pin and build inputs are in [`../runtime/upstream.lock`](../runtime/upstream.lock).
Do not replace them with floating tags. Validate the source and ordered patch
series before a hardware build:

```bash
runtime/scripts/prepare-source.sh
```

The supported target is vLLM 0.27.1 on GB10 with DeepGEMM MXFP4 experts and
`fp8_ds_mla` cache. Read [RUNTIME_V0271_GB10.md](RUNTIME_V0271_GB10.md) and
[NVFP4_DS_MLA.md](NVFP4_DS_MLA.md) before changing the cache dtype.

## Environment and Profiles

Copy the tracked candidate environment example to its untracked deployed
counterpart and fill local values outside Git:

```bash
cp cluster/environments/deepseek-v4-flash-0731-v0271-canary.env.example \
   cluster/environments/deepseek-v4-flash-0731-v0271-canary.env
```

The final controlled candidate envelope is:

| Setting | Value |
|---|---:|
| Context | 1,048,576 |
| Max sequences | 6 |
| Max batched tokens | 8,192 |
| GPU memory utilization | 0.84 |
| DSpark speculative tokens | 5 |
| KV cache dtype | `fp8_ds_mla` |
| KV arena | 26.3 GiB (`28235618304`) |
| Default thinking | max |
| Fabric | dual-HCA merge, GID unset, jumbo 9000 |
| Global / SWA block size | 256 / 64 |

The matching legacy baseline profile is
`deepseek-v4-flash-0731-dspark-1m-baseline`. Use it immediately before a
candidate A/B switch so automatic rollback has a configuration-equivalent
target.

## Stage, Install, Validate, Activate

Each operation is intentionally separated:

```bash
# Copy repository files to both ranks; does not change service state.
./cluster/deploy-to-sparks.sh --apply

# Install new profiles/control plane; does not restart the live service.
./cluster/install-control-plane.sh --install

# Validate model mounts, image parity, and requirements on both ranks.
vllm-switch validate deepseek-v4-flash-0731-v0271-canary

# Inspect current service/API state.
vllm-switch status
```

Only activate or roll back in an approved maintenance window:

```bash
vllm-switch deepseek-v4-flash-0731-v0271-canary
vllm-switch rollback
```

See [CLUSTER_CONTROL_PLANE.md](CLUSTER_CONTROL_PLANE.md) for lifecycle,
rollback, and failure-capture details. Long builds or model startup must run
detached with private logs and be polled to final readiness or rollback.

## Build and Qualification

Build through the guarded entrypoint:

```bash
./build-dspark-vllm-runtime.sh
```

After activation, run qualification under ignored storage:

```bash
OUTPUT_ROOT="$PWD/.private/qualification" \
  cluster/scripts/run-qualification.sh --run \
  --label rc-next --model deepseek-v4-flash-0731-v0271-canary --mode full
```

The current operator envelope and fabric results:
[DEEPSEEK_V4_FLASH_0731.md](DEEPSEEK_V4_FLASH_0731.md) and
[notes/2026-08-21-dual-hca.md](../notes/2026-08-21-dual-hca.md).
The 2026-08-12 image A/B protocol remains in
[GB10_DSV4_HANDOFF_2026-08-12.md](GB10_DSV4_HANDOFF_2026-08-12.md).
