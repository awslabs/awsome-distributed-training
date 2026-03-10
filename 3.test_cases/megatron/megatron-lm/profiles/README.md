# Megatron-LM Instance Profiles

Instance profiles configure GPU count, parallelism defaults (TP/PP), micro-batch
size, and EFA/NCCL networking variables for each supported EC2 instance type.

**Note on TP/PP coupling:** Megatron-LM's tensor and pipeline parallelism must
divide evenly into the available GPUs. The profiles set conservative defaults,
but you should tune TP/PP for your specific model size and node count. The GPT3
training script has built-in conditional logic that overrides these defaults
based on node count.

## Auto-detection

The training scripts auto-detect the running instance type and source the
matching `.env` profile. Detection order:

1. `INSTANCE_PROFILE` env var (explicit override, e.g. `g5-12xlarge`)
2. `INSTANCE_TYPE` env var
3. EC2 instance metadata API (IMDSv2)
4. GPU name from `nvidia-smi` (fallback)

To override auto-detection:

```bash
export INSTANCE_PROFILE=g5-12xlarge
```

See [docs/instance-compatibility.md](../../../docs/instance-compatibility.md)
for full details.

## Available Profiles

| Profile | Instance | GPUs | VRAM | EFA | Default TP/PP | Status |
|---------|----------|------|------|-----|---------------|--------|
| `p5en-48xlarge.env` | p5en.48xlarge | 8x H200 | 141 GB | 32 adapters | TP=8, PP=1 | Supported |
| `p5-48xlarge.env` | p5.48xlarge | 8x H100 | 80 GB | 32 adapters | TP=8, PP=1 | Supported |
| `p4de-24xlarge.env` | p4de.24xlarge | 8x A100 | 80 GB | 4 adapters | TP=4, PP=2 | Supported (original target) |
| `g6e-12xlarge.env` | g6e.12xlarge | 4x L40S | 48 GB | None | TP=4, PP=1 | Supported (medium models) |
| `g5-12xlarge.env` | g5.12xlarge | 4x A10G | 24 GB | None | TP=4, PP=1 | Supported (small models only) |

## Model Compatibility Matrix

### GPT3 (2.distributed-training.sbatch)

| Model | p5en | p5 | p4de | g6e | g5 |
|-------|------|----|------|-----|----|
| 1.7B  | Yes  | Yes | Yes | Yes | Yes |
| 3.6B  | Yes  | Yes | Yes | Yes | Yes |
| 7.5B (default) | Yes | Yes | Yes | Yes | Tight |
| 18.4B | Yes  | Yes | Yes | Tight | No |
| 39.1B | Yes  | Yes | Yes | No  | No |
| 76.1B+ | Yes | Yes | Yes | No  | No |

### Llama2 (pretrain-llama2.sbatch)

| Model | p5en | p5 | p4de | g6e | g5 |
|-------|------|----|------|-----|----|
| 7B (TP=1,PP=1) | Yes | Yes | Yes | Yes | Tight |
| 13B (TP=2,PP=1) | Yes | Yes | Yes | Tight | No |
| 70B (TP=4,PP=4) | Yes | Yes | Yes | No | No |

**Notes:**
- "Tight" means it may work but needs `--recompute-activations` and MBS=1
- "No" means the model will not fit in the available VRAM
- g5/g6e have 4 GPUs, so Llama2-70B's TP=4,PP=4 preset (requiring 16 GPUs)
  cannot run without adjusting to multi-node configurations

## Kubernetes Integration

The K8s `pytorchjob.yaml-template` already uses `envsubst` placeholders for
`GPU_PER_NODE`, `EFA_PER_NODE`, `TENSOR_PARALLEL`, etc. Source the profile
before running `envsubst` to set these variables:

```bash
# Detect instance and source profile
PROFILES_DIR="$(pwd)/../../profiles"
PROFILE_ENV=$("${PROFILES_DIR}/_detect.sh" "${PROFILES_DIR}")
source "$PROFILE_ENV"

# Set remaining model-specific variables
export NUM_LAYERS=36 HIDDEN_SIZE=4096 NUM_ATTENTION_HEADS=32
export SEQ_LENGTH=2048 MAX_POSITION_EMBEDDINGS=2048
export MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=288
export NUM_NODES=2 FI_PROVIDER=${FI_PROVIDER:-efa}

# Generate K8s manifest
cat pytorchjob.yaml-template | envsubst > pytorchjob.yaml
```
