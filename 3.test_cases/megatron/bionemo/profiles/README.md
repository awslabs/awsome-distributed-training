# BioNeMo Instance Profiles

Instance profiles configure GPU count, micro-batch size, and EFA/NCCL
networking variables for each supported EC2 instance type. Model architecture
parameters (num_layers, hidden_size, etc.) are handled by the training scripts
or BioNeMo config files.

## Auto-detection

The training scripts auto-detect the running instance type and source the
matching `.env` profile. Override with:

```bash
export INSTANCE_PROFILE=g5-12xlarge
```

See [docs/instance-compatibility.md](../../../docs/instance-compatibility.md)
for full details.

## Available Profiles

| Profile | Instance | GPUs | VRAM | EFA | Default MBS | Status |
|---------|----------|------|------|-----|-------------|--------|
| `p5en-48xlarge.env` | p5en.48xlarge | 8x H200 | 141 GB | 32 adapters | 256 | Supported |
| `p5-48xlarge.env` | p5.48xlarge | 8x H100 | 80 GB | 32 adapters | 256 | Supported |
| `p4de-24xlarge.env` | p4de.24xlarge | 8x A100 | 80 GB | 4 adapters | 256 | Supported (original target) |
| `g6e-12xlarge.env` | g6e.12xlarge | 4x L40S | 48 GB | None | 128 | Experimental |
| `g5-12xlarge.env` | g5.12xlarge | 4x A10G | 24 GB | None | 64 | Experimental |

## Model Compatibility

### ESM-1nv (BioNeMo 1.2, `2.esm1nv_pretrain.slurm`)

The key tunable is `MICRO_BATCH_SIZE`, which occupies ~85% of GPU memory at 256
on A100 80GB. Profile-sourced MBS values:

| Instance | VRAM | Profile MBS | Notes |
|----------|------|-------------|-------|
| p5en/p5/p4de | 80-141 GB | 256 | Original documented value |
| g6e | 48 GB | 128 | Estimated; tune based on actual usage |
| g5 | 24 GB | 64 | Estimated; may need further reduction |

### ESM-2 (BioNeMo 2.5, `bionemo_2.5/train-esm.sbatch`)

Uses fixed MBS=2 with 650M parameter model. Fits on all instance types.
The profile's `GPUS_PER_NODE` adjusts `--num-gpus` and SBATCH `--gpus-per-node`.
