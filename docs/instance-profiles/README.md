# Instance Profiles

This directory contains hardware specifications and tuning guidance for each
EC2 instance family used in this repository's test cases.

## NVIDIA GPU Instances

| Profile | GPU | VRAM | GPUDirect RDMA | EFA | Primary Use |
|---------|-----|------|----------------|-----|-------------|
| [g5.md](g5.md) | A10G | 24 GB | No | 1 | Cost-effective experimentation |
| [g6e.md](g6e.md) | L40S | 48 GB | No | 1 | Mid-range development/testing |
| [p4de.md](p4de.md) | A100 | 80 GB | Yes | 4 | Production training |
| [p5.md](p5.md) | H100 | 80 GB | Yes | 32 | Production training |
| [p5en.md](p5en.md) | H200 | 141 GB | Yes | 32 | Primary target (most test cases) |

## AWS Trainium Instances

| Profile | Accelerator | Memory | EFA | Primary Use |
|---------|-------------|--------|-----|-------------|
| [trn1.md](trn1.md) | Trainium v1/v2 | 32 GB HBM/core | 8-16 | Neuron SDK training |

## How to Use These Profiles

1. Identify which instance type you will be running on
2. Read the corresponding profile to understand the hardware constraints
3. Check the [compatibility matrix](../instance-compatibility.md) to see if
   your test case has been validated on that instance
4. Apply the NCCL/EFA settings and memory optimization strategies from the
   profile
5. If you validate a new test case + instance combination, update both the
   profile's "Tested Workloads" table and the central compatibility matrix
