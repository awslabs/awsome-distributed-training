# p5en Instance Family — NVIDIA H200

> Covers: p5en.48xlarge

## Hardware at a Glance

| Spec | p5en.48xlarge |
|------|---------------|
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
| EFA | 16 adapters (EFAv3) |
| Network bandwidth | 3,200 Gbps |
| vCPUs | 192 (Intel Xeon Sapphire Rapids) |
| System memory | 2,048 GiB |
| Local NVMe | 8 x 3.84 TB |

## Key Characteristics

- **141 GB HBM3e per GPU** — highest VRAM capacity among Hopper-generation
  instances; 4.8 TB/s memory bandwidth
- **EFAv3** — 35% lower latency than EFAv2 (used on p5/p5e); reduces
  collective communication overhead for multi-node training
- **PCIe Gen5** (Intel Sapphire Rapids) — 4x the CPU-GPU bandwidth of p5/p5e
  (PCIe Gen4); benefits CPU offloading and data loading
- **16 EFA adapters** — half of p5/p5e's 32 adapters, but EFAv3's lower
  latency partially compensates; same 3,200 Gbps aggregate bandwidth
- **Primary target** for most test cases in this repository

## Distributed Training Considerations

- **Max model size**: ~50B+ without offloading (BF16, TP=8); 141 GB per GPU
  is generous for most model architectures
- **Reference configuration**: Most test cases in this repo are developed and
  tested on p5en.48xlarge; adapting to other instances requires the parameter
  adjustments documented in each profile
- **PCIe Gen5 advantage**: Significant for workloads with heavy CPU-GPU data
  movement — FSDP with CPU offloading, data loading pipelines, CPU-based
  preprocessing

## NCCL / EFA Configuration

> For complete NCCL/EFA tuning guidance, see the
> [EFA Cheatsheet](../../1.architectures/efa-cheatsheet.md).

```bash
# p5en supports GPUDirect RDMA
export FI_EFA_USE_DEVICE_RDMA=1

# Standard EFA settings
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^docker,lo,veth
```
