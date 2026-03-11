# g6e Instance Family — L40S GPUs

> Covers: g6e.xlarge, g6e.2xlarge, g6e.4xlarge, g6e.8xlarge, g6e.12xlarge,
> g6e.16xlarge, g6e.24xlarge, g6e.48xlarge

## Hardware Summary

| Dimension | g6e.12xlarge | g6e.48xlarge |
|-----------|--------------|--------------|
| GPU | NVIDIA L40S | NVIDIA L40S |
| GPU VRAM | 48 GB GDDR6 | 48 GB GDDR6 |
| GPU Count | 4 | 8 |
| GPUDirect RDMA | **No** | **No** |
| EFA Adapters | 1 | 1 |
| NVLink | **None** (PCIe only) | **None** (PCIe only) |
| Node CPU Memory | ~168 Gi allocatable | ~768 Gi allocatable |
| Local Storage | 1 x 7.6 TB NVMe | 2 x 7.6 TB NVMe |

## Key Characteristics

- **Middle ground** between g5 (24 GB) and p4de (80 GB)
- **48 GB VRAM**: Double the capacity of g5
- **No NVLink**: Same as g5; GPU-to-GPU over PCIe only
- **No GPUDirect RDMA**: Same NCCL settings as g5
- **Single EFA adapter**: Same inter-node bandwidth limitations as g5

## Required NCCL / EFA Settings

```bash
# Same as g5 — no GPUDirect RDMA
export FI_EFA_USE_DEVICE_RDMA=0
export NCCL_PROTO=simple

export FI_PROVIDER=efa
export NCCL_DEBUG=WARN
```

## Comparison with g5

| Aspect | g6e.12xlarge | g5.12xlarge |
|--------|--------------|-------------|
| GPU | L40S 48 GB | A10G 24 GB |
| VRAM | 48 GB | 24 GB |
| NCCL settings | Identical | Identical |
| EFA | Identical | Identical |
