# g6e Instance Family — NVIDIA L40S

> Covers: g6e.xlarge through g6e.48xlarge

## Hardware at a Glance

| Spec | g6e.12xlarge | g6e.48xlarge |
|------|--------------|--------------|
| GPU | NVIDIA L40S (Ada Lovelace) | NVIDIA L40S (Ada Lovelace) |
| GPU count | 4 | 8 |
| VRAM per GPU | 48 GB GDDR6 | 48 GB GDDR6 |
| Memory bandwidth per GPU | 864 GB/s | 864 GB/s |
| BF16 TFLOPS per GPU | 362 (dense) / 733 (sparse) | 362 (dense) / 733 (sparse) |
| FP8 TFLOPS per GPU | 733 (dense) / 1,466 (sparse) | 733 (dense) / 1,466 (sparse) |
| TF32 TFLOPS per GPU | 183 (dense) / 366 (sparse) | 183 (dense) / 366 (sparse) |
| GPU interconnect | None (PCIe Gen4) | None (PCIe Gen4) |
| GPUDirect RDMA | No | No |
| EFA | 1 adapter (EFAv2) | 4 adapters (EFAv2) |
| Network bandwidth | 100 Gbps | 400 Gbps |
| vCPUs | 48 (AMD EPYC 7R13) | 192 (AMD EPYC 7R13) |
| System memory | 384 GiB | 1,536 GiB |
| Local NVMe | 2 x 1.9 TB | 4 x 1.9 TB |

## Key Characteristics

- **48 GB VRAM** — double the capacity of g5/g6; sufficient for models up to
  ~13B without offloading (BF16), or ~30B with FSDP + offloading
- **Ada Lovelace architecture** — FP8 Transformer Engine, 4th-gen Tensor Cores;
  ~3x the BF16 sparse compute of A10G
- **864 GB/s memory bandwidth** — significantly higher than L4 (300 GB/s) and
  A10G (600 GB/s); better utilization for memory-bound workloads
- **g6e.48xlarge has 4 EFA adapters** — 400 Gbps network, much better
  multi-node scaling than g5/g6 (which have only 1 adapter)
- **No NVLink, no GPUDirect RDMA** — inter-GPU communication still over PCIe;
  same NCCL settings as g5/g6

## Distributed Training Considerations

- **Max model size**: ~13B without offloading (BF16); ~30B+ with FSDP2 +
  CPU offloading on g6e.48xlarge (1,536 GiB system RAM)
- **Middle ground**: Between g5 (24 GB) and p4de (80 GB) — fits models that
  OOM on g5 but don't need the full p-family hardware
- **Parallelism**: FSDP or FSDP2; tensor parallelism over PCIe is possible
  but less efficient than NVSwitch-equipped instances
- **Multi-node on g6e.48xlarge**: 4 EFA adapters provide reasonable multi-node
  bandwidth (400 Gbps) — much better than g5.48xlarge (100 Gbps, 1 adapter)

## NCCL / EFA Configuration

> For complete NCCL/EFA tuning guidance, see the
> [EFA Cheatsheet](../../1.architectures/efa-cheatsheet.md).

```bash
# g6e does not support GPUDirect RDMA
export FI_EFA_USE_DEVICE_RDMA=0
export NCCL_PROTO=simple

# Standard EFA settings
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo,veth
```

## Available Sizes

| Size | GPUs | vCPUs | Memory (GiB) | Network (Gbps) | EFA | NVMe |
|------|------|-------|--------------|----------------|-----|------|
| g6e.xlarge | 1 | 4 | 32 | Up to 20 | No | 1 x 250 GB |
| g6e.2xlarge | 1 | 8 | 64 | Up to 20 | No | 1 x 450 GB |
| g6e.4xlarge | 1 | 16 | 128 | 20 | No | 1 x 600 GB |
| g6e.8xlarge | 1 | 32 | 256 | 25 | Yes | 2 x 450 GB |
| g6e.12xlarge | 4 | 48 | 384 | 100 | Yes (1) | 2 x 1.9 TB |
| g6e.16xlarge | 1 | 64 | 512 | 35 | Yes | 2 x 950 GB |
| g6e.24xlarge | 4 | 96 | 768 | 200 | Yes (2) | 2 x 1.9 TB |
| g6e.48xlarge | 8 | 192 | 1,536 | 400 | Yes (4) | 4 x 1.9 TB |
