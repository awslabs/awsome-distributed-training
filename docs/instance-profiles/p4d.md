# p4d Instance Family — NVIDIA A100 40 GB

> Covers: p4d.24xlarge

## Hardware at a Glance

| Spec | p4d.24xlarge |
|------|--------------|
| GPU | NVIDIA A100 40 GB (Ampere, SXM4) |
| GPU count | 8 |
| VRAM per GPU | 40 GB HBM2 |
| Memory bandwidth per GPU | 1,555 GB/s |
| BF16 TFLOPS per GPU | 312 (dense) / 624 (sparse) |
| TF32 TFLOPS per GPU | 156 (dense) / 312 (sparse) |
| FP8 per GPU | Not supported |
| GPU interconnect | NVSwitch, NVLink 3.0 (600 GB/s per GPU) |
| Total NVLink aggregate BW | 4.8 TB/s |
| GPUDirect RDMA | Yes |
| EFA | 4 adapters (EFAv1) |
| Network bandwidth | 400 Gbps |
| vCPUs | 96 (Intel Xeon 8275CL, dual-socket) |
| System memory | 1,152 GiB |
| Local NVMe | 8 x 1 TB |

## Key Characteristics

- **First-generation P-family for large-scale training** on AWS with NVSwitch
  and GPUDirect RDMA
- **40 GB VRAM** — half of p4de (80 GB); limits model size for TP-based
  training compared to the 80 GB variant
- **NVSwitch full-mesh** — all 8 GPUs communicate at 600 GB/s each; tensor
  parallelism is efficient within a node
- **Dual-socket Intel** — 4 GPUs per CPU socket; NUMA-aware placement matters
  for CPU offloading workloads
- **4 EFA adapters** — adequate for multi-node training, but 8x fewer than
  p5/p5e (32 adapters)

## Distributed Training Considerations

- **Max model size**: ~13B without offloading (BF16); ~30B+ with FSDP +
  offloading (1,152 GiB system RAM is generous)
- **Parallelism**: All strategies work — DP, FSDP, TP, PP; NVSwitch enables
  efficient tensor parallelism
- **vs p4de**: If your model fits in 40 GB per GPU with your chosen parallelism
  strategy, p4d is more cost-effective; otherwise use p4de (80 GB)
- **Multi-node**: 400 Gbps total is adequate but may bottleneck large all-reduce
  operations compared to p5's 3,200 Gbps

## NCCL / EFA Configuration

> For complete NCCL/EFA tuning guidance, see the
> [EFA Cheatsheet](../../1.architectures/efa-cheatsheet.md).

```bash
# p4d supports GPUDirect RDMA
export FI_EFA_USE_DEVICE_RDMA=1

# Standard EFA settings
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo,veth
```

## Comparison with p4de

| Aspect | p4d.24xlarge | p4de.24xlarge |
|--------|--------------|---------------|
| GPU VRAM | 40 GB HBM2 | 80 GB HBM2e |
| Memory bandwidth | 1,555 GB/s | 2,039 GB/s |
| NVSwitch | 600 GB/s (identical) | 600 GB/s (identical) |
| EFA / Network | 4 / 400 Gbps (identical) | 4 / 400 Gbps (identical) |
| System memory | 1,152 GiB (identical) | 1,152 GiB (identical) |
