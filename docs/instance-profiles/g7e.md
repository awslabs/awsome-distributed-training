# g7e Instance Family — NVIDIA RTX PRO 6000 Blackwell

> Covers: g7e.2xlarge through g7e.48xlarge

## Hardware at a Glance

| Spec | g7e.24xlarge | g7e.48xlarge |
|------|--------------|--------------|
| GPU | NVIDIA RTX PRO 6000 (Blackwell, GB202) | NVIDIA RTX PRO 6000 (Blackwell, GB202) |
| GPU count | 4 | 8 |
| VRAM per GPU | 96 GB GDDR7 | 96 GB GDDR7 |
| Memory bandwidth per GPU | 1,597 GB/s | 1,597 GB/s |
| BF16 TFLOPS per GPU | 125 (dense) / 252 (sparse) | 125 (dense) / 252 (sparse) |
| FP8 TFLOPS per GPU | 252 (dense) / 504 (sparse) | 252 (dense) / 504 (sparse) |
| FP4 TFLOPS per GPU | 504 (dense) / 1,008 (sparse) | 504 (dense) / 1,008 (sparse) |
| TF32 TFLOPS per GPU | 63 (dense) / 126 (sparse) | 63 (dense) / 126 (sparse) |
| GPU interconnect | None (PCIe Gen5 P2P) | None (PCIe Gen5 P2P) |
| GPUDirect RDMA | Yes | Yes |
| EFA | 2 adapters (EFAv4) | 4 adapters (EFAv4) |
| Network bandwidth | 800 Gbps | 1,600 Gbps |
| vCPUs | 96 (Intel Xeon Emerald Rapids) | 192 (Intel Xeon Emerald Rapids) |
| System memory | 1,024 GiB | 2,048 GiB |
| Local NVMe | 2 x 3.8 TB | 4 x 3.8 TB |

## Key Characteristics

- **96 GB GDDR7 per GPU** — more VRAM than H100 (80 GB) at a lower cost
  point; fits large models without the p-family price tag
- **Blackwell architecture** — 5th-gen Tensor Cores, FP4/FP6 support,
  2nd-gen Transformer Engine
- **GPUDirect RDMA + EFAv4** — first g-family instance with RDMA support;
  no `NCCL_PROTO=simple` workaround needed
- **PCIe Gen5 P2P** (no NVLink) — inter-GPU bandwidth is higher than
  previous g-family (Gen3/Gen4) but still significantly lower than NVSwitch
- **1,600 Gbps network on g7e.48xlarge** — 4x the bandwidth of g6e.48xlarge;
  approaches p4d/p4de network performance

## Distributed Training Considerations

- **Max model size**: ~30B without offloading (BF16); potentially ~70B+ with
  FSDP2 + offloading on g7e.48xlarge (2 TiB system RAM)
- **Unique positioning**: More VRAM than H100 (96 vs 80 GB) with RDMA support,
  but without NVSwitch — best for workloads that are memory-capacity-bound
  rather than interconnect-bound
- **FP4 support**: Blackwell's FP4 capability (2+ PFLOPS per GPU) enables
  new quantized training and inference workflows
- **Parallelism**: FSDP recommended; tensor parallelism over PCIe Gen5 is
  viable for small TP degrees (2-4) but NVSwitch instances will outperform
  for TP-heavy workloads

## NCCL / EFA Configuration

> For complete NCCL/EFA tuning guidance, see the
> [EFA Cheatsheet](../../1.architectures/efa-cheatsheet.md).

```bash
# g7e supports GPUDirect RDMA (first g-family with this support)
export FI_EFA_USE_DEVICE_RDMA=1

# Standard EFA settings
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo,veth
```

## Available Sizes

| Size | GPUs | vCPUs | Memory (GiB) | Network (Gbps) | EFA | NVMe |
|------|------|-------|--------------|----------------|-----|------|
| g7e.2xlarge | 1 | 8 | 64 | 50 | No | 1 x 1.9 TB |
| g7e.4xlarge | 1 | 16 | 128 | 50 | No | 1 x 1.9 TB |
| g7e.8xlarge | 1 | 32 | 256 | 100 | Yes (1) | 1 x 1.9 TB |
| g7e.12xlarge | 2 | 48 | 512 | 400 | Yes (1) | 1 x 3.8 TB |
| g7e.24xlarge | 4 | 96 | 1,024 | 800 | Yes (2) | 2 x 3.8 TB |
| g7e.48xlarge | 8 | 192 | 2,048 | 1,600 | Yes (4) | 4 x 3.8 TB |
