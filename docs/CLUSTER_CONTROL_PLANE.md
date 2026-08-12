# Cluster control plane

The repository-owned `cluster/vllm-switch` keeps the existing profile-based
operator workflow while supporting both launcher families found on the Spark
pair:

- `SERVICE_MODE=generic` builds the historical `launch-cluster.sh` argument
  arrays at runtime. Existing generic `.conf` profiles remain valid.
- `SERVICE_MODE=scripted` invokes explicit cluster start, lifecycle tracking,
  and stop commands. This is required for the specialized DeepSeek V4 service
  and the repository Compose candidate.

`vllm-switch` records the active profile in `.active-profile`; it no longer
infers identity by grepping systemd command strings. `list` never sources
profiles, `status` combines state with an API probe, `validate` checks both
model paths and requires the same Docker image ID on both ranks, and `render`
prints the unit without changing the service.

On activation the switch snapshots the effective unit, replaces the base unit,
and installs a managed final `zz-vllm-switch.conf` drop-in that clears historical `ExecStartPre`
directives. This prevents the old base/drop-in combination from running cache
and SHM cleanup twice. Scripted profiles declare their single pre-start hook in
the profile runner.

## Installation boundary

All scripts are inert until invoked. `cluster/deploy-to-sparks.sh` is a dry run
unless passed `--apply`, `cluster/install-control-plane.sh` requires
`--install`, and the image builder requires `--build`. Installing the control
plane backs up prior switch/profile files but does not stop or reload the live
service.

The first maintenance-window sequence is:

```bash
./cluster/deploy-to-sparks.sh --apply
ssh HEAD 'cd /home/USER/vllm-v0271-gb10 && ./cluster/install-control-plane.sh --install'
ssh HEAD 'vllm-switch adopt deepseek-v4-flash-0731-dspark'
ssh HEAD 'vllm-switch status'
ssh HEAD 'vllm-switch validate deepseek-v4-flash-0731-v0271-canary'
```

`deploy-to-sparks.sh --apply` only stages the repository. Whenever the deploy
changes `cluster/vllm-switch`, `cluster/vllm-profile-runner`, or either
profile, run `install-control-plane.sh --install` again before `validate` or
`switch`; those commands read the installed copies in `~/vllm-models`. The
deploy script reports a stale installed copy but deliberately does not install
it or change the service on its own.

`adopt` only writes profile state after verifying that the active systemd unit
contains the production launch markers. It does not restart the service.

## Candidate lifecycle

The candidate uses a canary-only served-model name and restart policy `no`.
The qualified rc7 profile runs at the controlled 1M A/B envelope. Before
switching to it, activate the matching Stage-C 1M baseline so the switch's
recorded-current-profile rollback returns to a configuration-equivalent arm.
If readiness fails, the switch captures both-rank state before activating that
recorded rollback profile. See
[GB10_DSV4_HANDOFF_2026-08-12.md](GB10_DSV4_HANDOFF_2026-08-12.md) for the
validated transition and operating constraints.

```bash
vllm-switch deepseek-v4-flash-0731-v0271-canary
vllm-switch status
vllm-switch rollback
```

The candidate mounts the existing node-local checkpoint read-only. Startup
requires matching checkpoint paths, deployment-file hashes, and image IDs on
both ranks. Worker startup precedes the head/API rank.

Cluster-specific hostnames, addresses, live image IDs, and mutation history
belong in the ignored `.private/CLUSTER_IMPLEMENTATION_LOG.md`, never in public
commits or issue reports.

## Qualification records

`cluster/scripts/capture-cluster-state.sh` archives both-rank manifests, logs,
the effective unit, API model data, Prometheus metrics, patch hashes, and source
state. `run-qualification.sh --run` adds completion/streaming/reasoning/tool/
structured-output contracts, cancellation recovery, concurrency, soak,
multi-needle retrieval, and percentile performance reports.

The full checkpoint hash pass is intentionally separate and refuses to run
while the serving unit is active:

```bash
./cluster/scripts/hash-checkpoint.sh --full
```

Image builds likewise refuse to run while serving and save their complete log,
source and patch manifests, image inspection, Python package inventory, and an
SPDX SBOM when `syft` is installed.
