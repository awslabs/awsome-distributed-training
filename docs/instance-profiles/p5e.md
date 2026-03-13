# p5e Instance Family — NVIDIA H200

> Covers: p5e.48xlarge

## Hardware at a Glance

| Spec | p5e.48xlarge |
|------|--------------|
| GPU | NVIDIA H200 (Hopper, SXM5) |
| GPU count | 8 |
| VRAM per GPU | 141 GB HBM3e |
| Memory bandwidth per GPU | 4,800 GB/s |
| BF16 TFLOPS per GPU | 990 (dense) / 1,979 (sparse) |
| FP8 TFLOPS per GPU | 1,979 (dense) / 3,958 (sparse) |
| TF32 TFLOPS per GPU | 495 (dense) / 989 (sparse) |
| GPU interconnect | NVSwitch, NVLink 4.0 (900 GB/s per GPU) |
| Total NVLink aggregate BW | 7.2 TB/s |
| GPUDirect RDMA | Yes |
| EFA | 32 adapters (EFAv2) |
| Network bandwidth | 3,200 Gbps |
| vCPUs | 192 (AMD EPYC 7R13) |
| System memory | 2,048 GiB |
| Local NVMe | 8 x 3.84 TB |

## Key Characteristics

- **141 GB HBM3e per GPU** — 76% more VRAM than H100 (80 GB); fits larger
  models per GPU, reducing parallelism overhead
- **4.8 TB/s memory bandwidth** — 43% more than H100; accelerates
  memory-bandwidth-bound operations (attention, large-batch inference)
- **Same Hopper compute** as H100 — identical TFLOPS; the advantage is purely
  in memory capacity and bandwidth
- **32 EFA adapters (EFAv2)** — same EFA generation as p5; 3,200 Gbps
  aggregate network bandwidth
- **PCIe Gen4** (AMD EPYC) — same as p5; for PCIe Gen5, see p5en

## Distributed Training Considerations

- **Max model size**: ~50B+ without offloading (BF16, TP=8); 141 GB per GPU
  significantly reduces the need for offloading or pipeline parallelism
- **Drop-in replacement for p5**: Same compute, same EFA gen, same PCIe;
  models that fit in 80 GB run identically, models that didn't now fit
- **When to choose p5e over p5en**: p5e has 32 EFA adapters (vs 16 on p5en)
  at the cost of PCIe Gen4 (vs Gen5). Choose p5e for workloads that are
  network-bandwidth-sensitive; choose p5en for CPU-GPU transfer-sensitive
  workloads

## NCCL / EFA Configuration

> For complete NCCL/EFA tuning guidance, see the
> [EFA Cheatsheet](../../1.architectures/efa-cheatsheet.md).

```bash
# p5e supports GPUDirect RDMA
export FI_EFA_USE_DEVICE_RDMA=1

# Standard EFA settings
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo,veth
```

## Comparison: p5 vs p5e vs p5en

| Aspect | p5.48xlarge | p5e.48xlarge | p5en.48xlarge |
|--------|-------------|--------------|---------------|
| GPU | H100 80 GB | H200 141 GB | H200 141 GB |
| Memory type | HBM3 | HBM3e | HBM3e |
| Memory bandwidth | 3,350 GB/s | 4,800 GB/s | 4,800 GB/s |
| BF16 TFLOPS | 990 (dense) / 1,979 (sparse) | 990 (dense) / 1,979 (sparse) | 990 (dense) / 1,979 (sparse) |
| EFA adapters | 32 (EFAv2) | 32 (EFAv2) | 16 (EFAv3) |
| PCIe | Gen4 (AMD) | Gen4 (AMD) | Gen5 (Intel) |
| CPU | AMD EPYC 7R13 | AMD EPYC 7R13 | Intel Sapphire Rapids |
