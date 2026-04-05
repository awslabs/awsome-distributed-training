# Nemotron 3 Super — Performance Results

> Status: Pending validation (no GPU capacity available yet)

## Expected Configurations

### LoRA Fine-Tuning

| Instance | Nodes | GPUs | TP | EP | PP | GBS | MBS | Precision | Throughput |
|----------|-------|------|----|----|----|-----|-----|-----------|------------|
| p5.48xlarge | 1 | 8 | 2 | TBD | 1 | 64 | 1 | FP8 | TBD |
| p5en.48xlarge | 1 | 8 | 2 | TBD | 1 | 128 | 1 | FP8 | TBD |
| p6-B200 | 1 | 8 | 2 | TBD | 1 | 128 | 2 | FP8 | TBD |

### Full Fine-Tuning

| Instance | Nodes | GPUs | TP | EP | PP | GBS | MBS | Precision | Throughput |
|----------|-------|------|----|----|----|-----|-----|-----------|------------|
| p5.48xlarge | 2 | 16 | 2 | TBD | 1 | 64 | 1 | FP8 | TBD |
| p5en.48xlarge | 1-2 | 8-16 | 2 | TBD | 1 | 128 | 1 | FP8 | TBD |
| p6-B200 | 1 | 8 | 2 | TBD | 1 | 128 | 2 | FP8 | TBD |

### GRPO Reinforcement Learning

| Instance | Nodes | GPUs | TP | EP | PP | GBS | MBS | Precision | Throughput |
|----------|-------|------|----|----|----|-----|-----|-----------|------------|
| p5.48xlarge | 2+ | 16+ | 2 | TBD | 1 | TBD | 1 | FP8 | TBD |
| p5en.48xlarge | 2+ | 16+ | 2 | TBD | 1 | TBD | 1 | FP8 | TBD |
| p6-B200 | 1-2 | 8-16 | 2 | TBD | 1 | TBD | 1 | FP8 | TBD |

> **Note**: Expert parallelism (EP) values are TBD pending validation with the
> 512-expert LatentMoE architecture. The optimal EP degree will depend on the
> number of GPUs and memory constraints.

## Validation Checklist

- [ ] Container builds successfully
- [ ] EFA networking verified (`fi_info -p efa`)
- [ ] NCCL allreduce benchmark passes
- [ ] HF -> Megatron checkpoint conversion completes (FP8 variant)
- [ ] HF -> Megatron checkpoint conversion completes (BF16 variant)
- [ ] LoRA SFT runs to completion (100 steps)
- [ ] LoRA merge + HF export succeeds
- [ ] Full fine-tuning runs to completion
- [ ] GRPO training runs to completion
- [ ] Multi-node scaling verified (2+ nodes)
- [ ] FP8 precision validated on Hopper GPUs
