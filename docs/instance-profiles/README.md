# Instance Hardware Profiles

This directory contains hardware specifications and distributed training
guidance for each EC2 accelerated instance family relevant to this repository.

Each profile includes GPU/accelerator specs (VRAM, compute TFLOPS, memory
bandwidth), interconnect topology (NVLink, NeuronLink), network configuration
(EFA generation, GPUDirect RDMA), and practical guidance for distributed
training workloads.

## GPU / Accelerator Quick Reference

> All TFLOPS values are **dense** (without sparsity). NVIDIA typically headlines
> "with sparsity" numbers, which are 2× the dense values shown here.

| GPU | Arch | BF16/FP16 | FP8 | FP4 | TF32/FP32 | Memory | Mem BW |
|-----|------|-----------|-----|-----|-----------|--------|--------|
| [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/) | Turing | 65 (FP16) | — | — | — | 16 GB GDDR6 | 320 GB/s |
| [A10G](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) | Ampere | 125 | — | — | 62.5 | 24 GB GDDR6 | 600 GB/s |
| [A100 40 GB](https://www.nvidia.com/en-us/data-center/a100/) | Ampere | 312 | — | — | 156 | 40 GB HBM2 | 1,555 GB/s |
| [A100 80 GB](https://www.nvidia.com/en-us/data-center/a100/) | Ampere | 312 | — | — | 156 | 80 GB HBM2e | 2,039 GB/s |
| [L4](https://www.nvidia.com/en-us/data-center/l4/) | Ada Lovelace | 121 | 242 | — | 60 | 24 GB GDDR6 | 300 GB/s |
| [L40S](https://www.nvidia.com/en-us/data-center/l40s/) | Ada Lovelace | 362 | 733 | — | 183 | 48 GB GDDR6 | 864 GB/s |
| [H100 SXM](https://www.nvidia.com/en-us/data-center/h100/) | Hopper | 990 | 1,979 | — | 495 | 80 GB HBM3 | 3,350 GB/s |
| [H200 SXM](https://www.nvidia.com/en-us/data-center/h200/) | Hopper | 990 | 1,979 | — | 495 | 141 GB HBM3e | 4,800 GB/s |
| [RTX PRO 6000](https://www.nvidia.com/en-us/design-visualization/rtx-pro-6000/) | Blackwell | 125 | 252 | 504 | 63 | 96 GB GDDR7 | 1,597 GB/s |
| [B200 HGX](https://www.nvidia.com/en-us/data-center/b200/) | Blackwell | 2,250 | 4,500 | 9,000 | 1,100 | 179 GB HBM3e | 8,000 GB/s |
| [B200 Grace](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) | Blackwell | 2,500 | 5,000 | 10,000 | — | 185 GB HBM3e | 8,000 GB/s |
| [B300 HGX](https://www.nvidia.com/en-us/data-center/b300/) | Blackwell Ultra | 2,250 | 4,500 | 13,500 | 1,125 | 288 GB HBM3e | 8,000 GB/s |
| Trainium v1 | NeuronCore v2 | 190 | — | — | 47.5 (FP32) | 32 GB HBM | 820 GB/s |
| Trainium v2 | NeuronCore v3 | 667 | 1,300 | — | 181 (FP32) | 96 GiB HBM3 | 2,900 GB/s |
| Inferentia v2 | NeuronCore v2 | 190 | — | — | — | 32 GB HBM | 820 GB/s |

## NVIDIA Instance Quick Reference

| Instance Type | GPU | GPUs | GPU Memory | NVLink | NVLink BW | EFA | EFA BW |
|---------------|-----|------|------------|--------|-----------|-----|--------|
| [g4dn.metal](g4dn.md) | T4 | 8 | 128 GB | — | — | v1 (1) | 12.5 GB/s |
| [g5.48xlarge](g5.md) | A10G | 8 | 192 GB | — | — | v1 (1) | 12.5 GB/s |
| [g6.48xlarge](g6.md) | L4 | 8 | 192 GB | — | — | v2 (1) | 12.5 GB/s |
| [g6e.48xlarge](g6e.md) | L40S | 8 | 384 GB | — | — | v2 (4) | 50 GB/s |
| [g7e.48xlarge](g7e.md) | RTX PRO 6000 | 8 | 768 GB | — | — | v4 (4) | 200 GB/s |
| [p4d.24xlarge](p4d.md) | A100 40 GB | 8 | 320 GB | v3 | 4.8 TB/s | v1 (4) | 50 GB/s |
| [p4de.24xlarge](p4de.md) | A100 80 GB | 8 | 640 GB | v3 | 4.8 TB/s | v1 (4) | 50 GB/s |
| [p5.48xlarge](p5.md) | H100 | 8 | 640 GB | v4 | 7.2 TB/s | v2 (32) | 400 GB/s |
| [p5e.48xlarge](p5e.md) | H200 | 8 | 1,128 GB | v4 | 7.2 TB/s | v2 (32) | 400 GB/s |
| [p5en.48xlarge](p5en.md) | H200 | 8 | 1,128 GB | v4 | 7.2 TB/s | v3 (16) | 400 GB/s |
| [p6-b200.48xlarge](p6-b200.md) | B200 | 8 | 1,432 GB | v5 | 14.4 TB/s | v4 (8) | 400 GB/s |
| [p6-b300.48xlarge](p6-b300.md) | B300 | 8 | 2,100 GB | v5 | 14.4 TB/s | v4 (17) | 800 GB/s |
| [p6e-gb200.36xlarge](p6e-gb200.md) | B200 Grace | 4 | 740 GB | v5 | 7.2 TB/s | v4 (17) | 400 GB/s |

## Trainium / Inferentia Instance Quick Reference

| Instance Type | Accelerator | Chips | Cores | Total Memory | Interconnect | Interconnect BW | EFA | EFA BW |
|---------------|-------------|-------|-------|-------------|--------------|-----------------|-----|--------|
| [trn1.32xlarge](trn1.md) | Trainium v1 | 16 | 32 NeuronCores | 512 GB | NeuronLink v2 | 768 GB/s | v2 (8) | 100 GB/s |
| [trn1n.32xlarge](trn1.md) | Trainium v1 | 16 | 32 NeuronCores | 512 GB | NeuronLink v2 | 768 GB/s | v2 (16) | 200 GB/s |
| [trn2.48xlarge](trn2.md) | Trainium v2 | 16 | 128 NeuronCores | 1.5 TB | NeuronLink v3 | 1,024 GB/s/chip | v3 (16) | 400 GB/s |
| [trn2u.48xlarge](trn2.md) | Trainium v2 | 16 | 128 NeuronCores | 1.5 TB | NeuronLink v3 | 1,024 GB/s/chip | v3 (16) | 400 GB/s |
| [inf2.48xlarge](inf2.md) | Inferentia v2 | 12 | 24 NeuronCores | 384 GB | NeuronLink v2 | 192 GB/s | — | — |

For detailed per-instance hardware specs, interconnect topology, NCCL/EFA
configuration, and distributed training guidance, see the individual profiles
and comparison tables below.

---

## NVIDIA GPU Instances

### G-Family (PCIe-attached, no NVSwitch)

| Profile | GPU | Architecture | VRAM | BF16 TFLOPS | Memory BW | GPUDirect RDMA | Max EFA |
|---------|-----|-------------|------|-------------|-----------|----------------|---------|
| [g4dn](g4dn.md) | T4 | Turing | 16 GB GDDR6 | 65 (FP16) | 320 GB/s | No | 1 (EFAv1) |
| [g5](g5.md) | A10G | Ampere | 24 GB GDDR6 | 125 | 600 GB/s | No | 1 (EFAv1) |
| [g6](g6.md) | L4 | Ada Lovelace | 24 GB GDDR6 | 121 | 300 GB/s | No | 1 (EFAv2) |
| [g6e](g6e.md) | L40S | Ada Lovelace | 48 GB GDDR6 | 362 | 864 GB/s | No | 4 (EFAv2) |
| [g7e](g7e.md) | RTX PRO 6000 | Blackwell | 96 GB GDDR7 | 125 | 1,597 GB/s | **Yes** | 4 (EFAv4) |

### P-Family (NVSwitch full-mesh, GPUDirect RDMA)

| Profile | GPU | Architecture | VRAM | BF16 TFLOPS | Memory BW | NVLink BW/GPU | Max EFA |
|---------|-----|-------------|------|-------------|-----------|---------------|---------|
| [p4d](p4d.md) | A100 40 GB | Ampere | 40 GB HBM2 | 312 | 1,555 GB/s | 600 GB/s (v3) | 4 (EFAv1) |
| [p4de](p4de.md) | A100 80 GB | Ampere | 80 GB HBM2e | 312 | 2,039 GB/s | 600 GB/s (v3) | 4 (EFAv1) |
| [p5](p5.md) | H100 | Hopper | 80 GB HBM3 | 990 | 3,350 GB/s | 900 GB/s (v4) | 32 (EFAv2) |
| [p5e](p5e.md) | H200 | Hopper | 141 GB HBM3e | 990 | 4,800 GB/s | 900 GB/s (v4) | 32 (EFAv2) |
| [p5en](p5en.md) | H200 | Hopper | 141 GB HBM3e | 990 | 4,800 GB/s | 900 GB/s (v4) | 16 (EFAv3) |
| [p6-b200](p6-b200.md) | B200 | Blackwell | 179 GB HBM3e | 2,250 | 8,000 GB/s | 1,800 GB/s (v5) | 8 (EFAv4) |
| [p6-b300](p6-b300.md) | B300 | Blackwell Ultra | 288 GB HBM3e | 2,250 | 8,000 GB/s | 1,800 GB/s (v5) | 17 (EFAv4) |
| [p6e-gb200](p6e-gb200.md) | B200 (Grace) | Blackwell | 185 GB HBM3e | 2,500 | 8,000 GB/s | 1,800 GB/s (v5) | 17 (EFAv4) |

## AWS Trainium / Inferentia Instances

| Profile | Accelerator | Cores/Chip | HBM/Chip | BF16/Chip | Interconnect | Max EFA |
|---------|-------------|------------|----------|-----------|--------------|---------|
| [trn1](trn1.md) | Trainium v1 | 2 (NeuronCore v2) | 32 GB | ~190 TFLOPS | NeuronLink v2 (768 GB/s) | 16 (EFAv2) |
| [trn2](trn2.md) | Trainium v2 | 8 (NeuronCore v3) | 96 GiB | ~667 TFLOPS | NeuronLink v3 (1,024 GB/s/chip) | 16 (EFAv3) |
| [inf2](inf2.md) | Inferentia v2 | 2 (NeuronCore v2) | 32 GB | 190 TFLOPS | NeuronLink v2 (192 GB/s) | None |

## Quick Selection Guide

**By VRAM per GPU (ascending):**
T4 (16 GB) → L4/A10G (24 GB) → A100-40 (40 GB) → L40S (48 GB) →
A100-80/H100 (80 GB) → RTX PRO 6000 (96 GB) → H200 (141 GB) →
B200 (179-185 GB) → B300 (288 GB)

**By use case:**
- **Experimentation / fine-tuning (small models)**: g5, g6, g4dn
- **Development / medium models**: g6e, g7e, p4d
- **Production training**: p4de, p5, p5e, p5en
- **Large-scale foundation models**: p6-b200, p6-b300, p6e-gb200
- **Neuron SDK training**: trn1n, trn2, trn2u
- **Inference**: inf2, g4dn, g6

**By GPUDirect RDMA support:**
- **Yes**: p4d, p4de, p5, p5e, p5en, p6-b200, p6-b300, p6e-gb200, g7e
- **No**: g4dn, g5, g6, g6e

## NCCL / EFA Configuration Reference

For NCCL environment variables, EFA settings, and version-specific tuning
guidance, see the [EFA Cheatsheet](../../1.architectures/efa-cheatsheet.md).

The key configuration difference between instance families:

```bash
# Instances WITH GPUDirect RDMA (p-family, g7e):
export FI_EFA_USE_DEVICE_RDMA=1
# NCCL_PROTO can use default (LL/LL128)

# Instances WITHOUT GPUDirect RDMA (g4dn, g5, g6, g6e):
export FI_EFA_USE_DEVICE_RDMA=0
export NCCL_PROTO=simple
```

## EFA Generations

| Generation | Instances | Key Improvement |
|------------|-----------|-----------------|
| EFAv1 | g4dn, g5, p4d, p4de | First generation |
| EFAv2 | g6, g6e, p5, p5e | Higher throughput |
| EFAv3 | p5en, trn2, trn2u | 35% lower latency vs EFAv2 |
| EFAv4 | g7e, p6-b200, p6-b300, p6e-gb200 | Latest generation |
