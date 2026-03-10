# NeMo Instance Profiles

NeMo uses Python-based NeMo-Run scripts (`run.py`, K8s scripts) that accept
environment variables via `--env_vars_file`. This directory provides per-instance
`env_vars_<instance>.json` files with the correct EFA/NCCL networking settings.

**Key difference from other test cases:** NeMo profiles only control networking
variables. Training parameters (TP, PP, MBS, GBS, etc.) are managed by NeMo
recipes and are passed as Python CLI arguments to the launch scripts, not via
the env vars file.

## Usage

### Slurm

```bash
# EFA instances (p5en, p5, p4de)
python run.py --container_image ~/aws-nemo.sqsh --nodes 2 \
    --env_vars_file ../profiles/env_vars_p5en.json

# Non-EFA instances (g5, g6e)
python run.py --container_image ~/aws-nemo.sqsh --nodes 2 \
    --ntasks_per_node 4 \
    --env_vars_file ../profiles/env_vars_g5.json
```

### Kubernetes (SkyPilot)

```bash
# EFA instances
python pretrain_mock_dataset.py --nodes 2 --gpu-devices 8 \
    --efa-devices 32 --env_vars_file ../profiles/env_vars_p5en.json

# Non-EFA instances
python pretrain_mock_dataset.py --nodes 1 --gpu-devices 4 \
    --env_vars_file ../profiles/env_vars_g6e.json
```

## Available Profiles

| Profile | Instance | GPUs | EFA | FI_PROVIDER | NVLS |
|---------|----------|------|-----|-------------|------|
| `env_vars_p5en.json` | p5en.48xlarge | 8x H200 | 32 | efa | 1 |
| `env_vars_p5.json` | p5.48xlarge | 8x H100 | 32 | efa | 0 |
| `env_vars_p4de.json` | p4de.24xlarge | 8x A100 | 4 | efa | 0 |
| `env_vars_g6e.json` | g6e.12xlarge | 4x L40S | 0 | (unset) | 0 |
| `env_vars_g5.json` | g5.12xlarge | 4x A10G | 0 | (unset) | 0 |

## Instance-Specific CLI Arguments

The `env_vars.json` profile handles networking, but you must also adjust the
CLI arguments to match the instance:

| Instance | `--ntasks_per_node` (Slurm) | `--gpu-devices` (K8s) | `--efa-devices` (K8s) |
|----------|---------------------------|----------------------|---------------------|
| p5en.48xlarge | 8 (default) | 8 (default) | 32 |
| p5.48xlarge | 8 (default) | 8 (default) | 32 |
| p4de.24xlarge | 8 (default) | 8 (default) | 4 |
| g6e.12xlarge | 4 | 4 | (omit) |
| g5.12xlarge | 4 | 4 | (omit) |

## Performance Recipes

The `PERFORMANCE.md` in the parent directory documents validated TP/PP/GBS
configurations for various models on H100/H200/B200. Use those recipes as-is
on p5en/p5 instances. For g5/g6e, smaller model configurations should be
selected (e.g., smaller batch sizes, fewer pipeline stages).

See [docs/instance-compatibility.md](../../../docs/instance-compatibility.md)
for full instance reference.
