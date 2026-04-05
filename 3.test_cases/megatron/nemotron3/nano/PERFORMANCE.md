# Nemotron 3 Nano — Performance Results

> Status: Pending validation (no GPU capacity available yet)

## Expected Configurations

### LoRA Fine-Tuning

| Instance | Nodes | GPUs | TP | EP | PP | GBS | MBS | Precision | Throughput |
|----------|-------|------|----|----|----|-----|-----|-----------|------------|
| p5.48xlarge | 1 | 8 | 1 | 8 | 1 | 128 | 1 | BF16 | TBD |
| p5en.48xlarge | 1 | 8 | 1 | 8 | 1 | 128 | 1 | FP8 | TBD |
| p6-B200 | 1 | 8 | 1 | 8 | 1 | 128 | 2 | NVFP4 | TBD |
| p4de.24xlarge | 1 | 8 | 1 | 8 | 1 | 64 | 1 | BF16 | TBD |

### Full Fine-Tuning

| Instance | Nodes | GPUs | TP | EP | PP | GBS | MBS | Precision | Throughput |
|----------|-------|------|----|----|----|-----|-----|-----------|------------|
| p5.48xlarge | 2 | 16 | 1 | 8 | 1 | 128 | 1 | BF16 | TBD |
| p5en.48xlarge | 1 | 8 | 1 | 8 | 1 | 128 | 1 | FP8 | TBD |

### GRPO Reinforcement Learning

| Instance | Nodes | GPUs | TP | EP | PP | GBS | MBS | Precision | Throughput |
|----------|-------|------|----|----|----|-----|-----|-----------|------------|
| p5.48xlarge | 2 | 16 | 1 | 8 | 1 | TBD | 1 | BF16 | TBD |
| p5en.48xlarge | 1 | 8 | 1 | 8 | 1 | TBD | 1 | FP8 | TBD |

## Validation Checklist

- [ ] Container builds successfully
- [ ] EFA networking verified (`fi_info -p efa`)
- [ ] NCCL allreduce benchmark passes
- [ ] HF -> Megatron checkpoint conversion completes
- [ ] LoRA SFT runs to completion (100 steps)
- [ ] LoRA merge + HF export succeeds
- [ ] Full fine-tuning runs to completion
- [ ] GRPO training runs to completion
- [ ] Multi-node scaling verified (2+ nodes)
