# g4dn Instance Family — NVIDIA T4

> Covers: g4dn.xlarge through g4dn.metal

## Hardware at a Glance

| Spec | g4dn.12xlarge | g4dn.metal |
|------|---------------|------------|
| GPU | NVIDIA T4 (Turing) | NVIDIA T4 (Turing) |
| GPU count | 4 | 8 |
| VRAM per GPU | 16 GB GDDR6 | 16 GB GDDR6 |
| Memory bandwidth per GPU | 320 GB/s | 320 GB/s |
| FP16 TFLOPS per GPU | 65 | 65 |
| FP8 per GPU | Not supported | Not supported |
| GPU interconnect | None (PCIe Gen3) | None (PCIe Gen3) |
| GPUDirect RDMA | No | No |
| EFA | 1 adapter (EFAv1) | 1 adapter (EFAv1) |
| Network bandwidth | 50 Gbps | 100 Gbps |
| vCPUs | 48 (Intel Xeon P-8259L) | 96 (Intel Xeon P-8259L) |
| System memory | 192 GiB | 384 GiB |
| Local NVMe | 1 x 900 GB | 2 x 900 GB |

## Key Characteristics

- **Lowest-cost NVIDIA GPU option** on AWS; best for inference, small-scale
  fine-tuning, and experimentation
- **16 GB VRAM** limits training to small models (< 3B parameters) without
  aggressive memory optimization
- **Turing architecture** — no BF16 Tensor Core support (FP16 only), no TF32,
  no FP8; older than Ampere
- **No NVLink, no GPUDirect RDMA** — inter-GPU communication over PCIe Gen3 is
  slow; multi-GPU training within a node is bottlenecked
- **Single EFA adapter** — limited inter-node bandwidth for distributed training

## Distributed Training Considerations

- **Max model size**: ~1-3B parameters with FP16, depending on optimizer and
  batch size
- **Parallelism**: Data parallelism only practical; tensor parallelism over
  PCIe Gen3 is too slow to be worthwhile
- **Multi-node**: Possible but limited by single EFA adapter (50-100 Gbps)
- **Recommended for**: Small model experimentation, inference workloads,
  data preprocessing pipelines

## NCCL / EFA Configuration

> For complete NCCL/EFA tuning guidance, see the
> [EFA Cheatsheet](../../1.architectures/efa-cheatsheet.md).

```bash
# g4dn does not support GPUDirect RDMA
export FI_EFA_USE_DEVICE_RDMA=0
export NCCL_PROTO=simple

# Standard EFA settings
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo,veth
```

## Available Sizes

| Size | GPUs | vCPUs | Memory (GiB) | Network (Gbps) | EFA | NVMe |
|------|------|-------|--------------|----------------|-----|------|
| g4dn.xlarge | 1 | 4 | 16 | Up to 25 | No | 1 x 125 GB |
| g4dn.2xlarge | 1 | 8 | 32 | Up to 25 | No | 1 x 225 GB |
| g4dn.4xlarge | 1 | 16 | 64 | Up to 25 | No | 1 x 225 GB |
| g4dn.8xlarge | 1 | 32 | 128 | 50 | Yes | 1 x 900 GB |
| g4dn.12xlarge | 4 | 48 | 192 | 50 | Yes | 1 x 900 GB |
| g4dn.16xlarge | 1 | 64 | 256 | 50 | Yes | 1 x 900 GB |
| g4dn.metal | 8 | 96 | 384 | 100 | Yes | 2 x 900 GB |
