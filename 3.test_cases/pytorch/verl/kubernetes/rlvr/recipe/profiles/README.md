# Instance Profiles for veRL RLVR Recipes

This directory contains instance-specific configuration profiles that
override the hardware-dependent parameters in the recipe scripts.

## How It Works

1. The recipe script (e.g., `run_grpo_configurable.sh`) calls `_detect.sh`
   to determine which profile to load
2. `_detect.sh` resolves the instance type (from env var, EC2 metadata, or
   GPU detection) and returns the path to the matching `.env` file
3. The recipe sources the `.env` file, overriding default values with
   instance-specific settings

## Detection Order

The profile is selected by the first method that succeeds:

1. **`INSTANCE_PROFILE` env var** — explicit override (e.g., `g5-12xlarge`)
2. **`INSTANCE_TYPE` env var** — from `setup/env_vars` (e.g., `g5.12xlarge`)
3. **EC2 instance metadata API** — works on bare metal and K8s with host networking
4. **GPU name from `nvidia-smi`** — fallback when metadata is unavailable

## Available Profiles

| Profile | Instance | GPU | VRAM | Status |
|---------|----------|-----|------|--------|
| [p5en-48xlarge.env](p5en-48xlarge.env) | p5en.48xlarge | 8x H200 | 80 GB | Tested |
| [g5-12xlarge.env](g5-12xlarge.env) | g5.12xlarge | 4x A10G | 24 GB | Tested |
| [p4de-24xlarge.env](p4de-24xlarge.env) | p4de.24xlarge | 8x A100 | 80 GB | Untested |

## What's in a Profile

Profiles contain **only instance-dependent parameters** — settings that
change based on hardware. Algorithm-specific settings (KL loss, reward
function, dataset, learning rate, etc.) stay in the recipe script.

Instance-dependent parameters include:

| Category | Parameters |
|----------|-----------|
| Cluster geometry | `NUM_GPU_PER_NODE`, `NUM_EFA_PER_NODE` |
| FSDP strategy | `ACTOR_STRATEGY`, `PARAM_OFFLOAD`, `OPTIMIZER_OFFLOAD`, `OFFLOAD_POLICY`, `MODEL_DTYPE`, `RESHARD_AFTER_FORWARD` |
| vLLM rollout | `TENSOR_PARALLEL_SIZE`, `GPU_MEMORY_UTILIZATION`, `ENFORCE_EAGER`, `ROLLOUT_DTYPE` |
| NCCL / EFA | `NCCL_PROTO`, `FI_EFA_USE_DEVICE_RDMA` |
| Training | `MAX_RESPONSE_LENGTH`, `LOG_PROB_MICRO_BSZ_PER_GPU` |
| Checkpoints | `SAVE_FREQ`, `MAX_ACTOR_CKPT_TO_KEEP`, `TEST_FREQ` |
| K8s resources | `WORKER_MEMORY`, `WORKER_CPU` |

## Creating a New Profile

1. Copy the closest existing profile:
   ```bash
   cp p5en-48xlarge.env g6e-12xlarge.env
   ```

2. Adjust the parameters based on the target instance's hardware.
   See [docs/instance-profiles/](../../../../../../docs/instance-profiles/)
   for hardware specs.

3. Key questions when creating a profile:
   - **GPU VRAM < 48 GB?** → Likely need FSDP2 + offloading
   - **No GPUDirect RDMA?** → Set `NCCL_PROTO=simple`, `FI_EFA_USE_DEVICE_RDMA=0`
   - **4 GPUs per node?** → Set `TENSOR_PARALLEL_SIZE=4`
   - **Less than 200 Gi allocatable RAM?** → Reduce `WORKER_MEMORY`

4. Test with a 2-step smoke run before committing:
   ```bash
   export INSTANCE_PROFILE=g6e-12xlarge
   export TOTAL_EPOCHS=1
   ./recipe/run_grpo_configurable.sh
   ```

5. Update the Tested Configurations table in the README and the
   [central compatibility matrix](../../../../../../docs/instance-compatibility.md).
