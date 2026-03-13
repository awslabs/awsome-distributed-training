# inf2 Instance Family — AWS Inferentia v2

> Covers: inf2.xlarge, inf2.8xlarge, inf2.24xlarge, inf2.48xlarge

## Hardware at a Glance

| Spec | inf2.xlarge | inf2.24xlarge | inf2.48xlarge |
|------|-------------|---------------|---------------|
| Accelerator | AWS Inferentia v2 | AWS Inferentia v2 | AWS Inferentia v2 |
| Chips | 1 | 6 | 12 |
| NeuronCores (v2) | 2 | 12 | 24 |
| HBM per chip | 32 GB | 32 GB | 32 GB |
| Total accelerator memory | 32 GB | 192 GB | 384 GB |
| Memory bandwidth per chip | 820 GB/s | 820 GB/s | 820 GB/s |
| BF16 TFLOPS per chip | 190 | 190 | 190 |
| INT8 TOPS per chip | 380 | 380 | 380 |
| Chip interconnect | N/A (single chip) | NeuronLink v2 (192 GB/s) | NeuronLink v2 (192 GB/s) |
| EFA | No | No | No |
| Network bandwidth | Up to 15 Gbps | 50 Gbps | 100 Gbps |
| vCPUs | 4 | 96 | 192 |
| System memory | 16 GiB | 384 GiB | 768 GiB |

## Key Characteristics

- **Optimized for inference**, not training — same NeuronCore v2 as Trainium v1
  but without training-specific features
- **Same Neuron SDK** as Trainium — models compiled for Inferentia v2 use the
  same `torch-neuronx` and Neuron compiler toolchain
- **NeuronLink v2** on multi-chip instances — enables tensor parallelism for
  large model inference (e.g., LLM serving with 70B+ models)
- **Cost-effective inference** — lower cost per inference than NVIDIA GPU
  instances for many model architectures

## Use in This Repository

Inf2 instances are primarily used for inference workloads, not training. They
appear in this repository in the context of:

- Model serving and inference benchmarks
- Testing inference pipelines for models trained on Trainium or NVIDIA GPUs
- Neuron SDK compatibility validation

For training workloads, use [trn1/trn1n](trn1.md) or [trn2/trn2u](trn2.md).

## Configuration

```bash
# Neuron runtime configuration
export NEURON_RT_NUM_CORES=2   # Adjust per instance (2 per chip)

# For multi-chip inference (inf2.24xl / inf2.48xl)
# NeuronLink handles inter-chip communication automatically
```

## Available Sizes

| Size | Chips | NeuronCores | HBM | vCPUs | Memory (GiB) | Network (Gbps) | EFA |
|------|-------|-------------|-----|-------|--------------|----------------|-----|
| inf2.xlarge | 1 | 2 | 32 GB | 4 | 16 | Up to 15 | No |
| inf2.8xlarge | 1 | 2 | 32 GB | 32 | 128 | Up to 25 | No |
| inf2.24xlarge | 6 | 12 | 192 GB | 96 | 384 | 50 | No |
| inf2.48xlarge | 12 | 24 | 384 GB | 192 | 768 | 100 | No |
