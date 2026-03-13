# p4de Instance Family — NVIDIA A100 80 GB

> Covers: p4d.24xlarge, p4de.24xlarge

## Hardware at a Glance

| Spec | p4de.24xlarge | p4d.24xlarge |
|------|---------------|--------------|
| GPU | NVIDIA A100 80 GB (Ampere, SXM4) | NVIDIA A100 40 GB (Ampere, SXM4) |
| GPU count | 8 | 8 |
| VRAM per GPU | 80 GB HBM2e | 40 GB HBM2 |
| Memory bandwidth per GPU | 2,039 GB/s | 1,555 GB/s |
| BF16 TFLOPS per GPU | 312 (dense) / 624 (sparse) | 312 (dense) / 624 (sparse) |
| TF32 TFLOPS per GPU | 156 (dense) / 312 (sparse) | 156 (dense) / 312 (sparse) |
| FP8 per GPU | Not supported | Not supported |
| GPU interconnect | NVSwitch, NVLink 3.0 (600 GB/s per GPU) | NVSwitch, NVLink 3.0 (600 GB/s per GPU) |
| Total NVLink aggregate BW | 4.8 TB/s | 4.8 TB/s |
| GPUDirect RDMA | Yes | Yes |
| EFA | 4 adapters (EFAv1) | 4 adapters (EFAv1) |
| Network bandwidth | 400 Gbps | 400 Gbps |
| vCPUs | 96 (Intel Xeon 8275CL, dual-socket) | 96 (Intel Xeon 8275CL, dual-socket) |
| System memory | 1,152 GiB | 1,152 GiB |
| Local NVMe | 8 x 1 TB | 8 x 1 TB |

## Key Characteristics

- **80 GB HBM2e** — double the VRAM of p4d; sufficient for most models up to
  ~30B without offloading (BF16 with TP=8)
- **2 TB/s memory bandwidth** — 30% more than p4d (40 GB variant); improves
  throughput for memory-bandwidth-bound operations
- **NVSwitch full-mesh** — all 8 GPUs communicate at 600 GB/s each; tensor
  parallelism within a node is efficient
- **Fewer EFA adapters** (4 vs 32 on p5) — inter-node bandwidth is 400 Gbps
  vs 3,200 Gbps; multi-node scaling is less efficient
- **1,152 GiB system memory** — less than p5/p5en (2,048 GiB) but still ample for CPU offloading workloads

## Distributed Training Considerations

- **Max model size**: ~30B without offloading (BF16, TP=8); ~70B+ with FSDP +
  pipeline parallelism across multiple nodes
- **Parallelism**: All strategies — DP, FSDP, TP, PP; NVSwitch enables
  efficient tensor parallelism
- **Multi-node trade-off**: 4 EFA adapters (400 Gbps) works well for up to
  ~8-16 nodes; beyond that, gradient synchronization latency becomes
  significant vs p5's 32 adapters
- **MIG support**: A100 supports Multi-Instance GPU (up to 7 instances per GPU)
  for multi-tenant inference workloads

## NCCL / EFA Configuration

> For complete NCCL/EFA tuning guidance, see the
> [EFA Cheatsheet](../../1.architectures/efa-cheatsheet.md).

```bash
# p4de supports GPUDirect RDMA
export FI_EFA_USE_DEVICE_RDMA=1

# Standard EFA settings
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo,veth
```

## Comparison with p5en

| Aspect | p4de.24xlarge | p5en.48xlarge |
|--------|---------------|---------------|
| GPU | A100 80 GB HBM2e | H200 141 GB HBM3e |
| BF16 TFLOPS/GPU | 312 | 1,979 |
| Memory bandwidth | 2,039 GB/s | 4,800 GB/s |
| NVSwitch bandwidth | 600 GB/s | 900 GB/s |
| EFA adapters | 4 | 16 |
| Network bandwidth | 400 Gbps | 3,200 Gbps |
| System memory | 1,152 GiB | 2,048 GiB |
