# Instance Compatibility Framework — Implementation Plan

> Proposed: 2026-03-10 | Status: Approved, not yet started
> Branch: `feat/instance-compatibility-framework` (to be created)
> Context: After 11 OOM iterations getting veRL GRPO to work on g5.12xlarge (A10G 24GB),
> it became clear the repo lacks systematic guidance for running test cases across
> different GPU instance types.

## Problem Statement

The `awsome-distributed-training` repo has ~20 test cases across PyTorch, Megatron, JAX,
and NeuronX. Most are written and tested for a single instance type (usually p5en.48xlarge
with H200 80GB GPUs). When users try to run these on different instances (g5, p4de, g6e),
they hit undocumented failures:

- **7 of ~20 test cases have zero instance type guidance** in their READMEs
- **No centralized compatibility matrix** exists
- **CI only tests on P5 instances** — g5, p4de, trn1 are never validated
- **Scripts hardcode values** for one instance type without parameterized alternatives

Real-world impact: Running veRL GRPO on g5.12xlarge required 11 iterations to resolve
OOM failures caused by parameters that work fine on p5en but break on 24GB GPUs. Every
failure mapped to a parameter that differs by instance type (FSDP strategy, offload policy,
dtype, gpu_memory_utilization, TP degree, NCCL settings, checkpoint frequency).

## Current State (from repo analysis)

### Test Cases with Instance Type Documentation
| Test Case | Documented Instances |
|-----------|---------------------|
| FSDP | P4d(e), P5, P6-B200, G5.12xlarge, G5.xlarge, G4dn |
| NeMo 2.0 | H100 (p5en), H200, B200 (p6) — has PERFORMANCE.md |
| NeMo 1.0 | p4de.24xlarge (A100 80GB) |
| bionemo | p4de.24xlarge, P5 |
| distillation | P4d, P5, P5en (explicit list) |
| nanoVLM | g5 (optional section) |
| optimum-neuron | trn1.32xlarge, trn1n, trn2.48xlarge, trn2.3xlarge |
| ESM2 | g5.24xlarge, g5.12xlarge, p5.48xlarge |
| Stable Diffusion | P4de, P5 (perf comparison table) |
| veRL (rlvr) | p5en.48xlarge, g5.12xlarge (after our work) |

### Test Cases with NO Instance Type Guidance
JAX/Paxml, PyTorch DDP, DeepSpeed, torchtitan, TRL, MPT, Picotron

### CI Coverage
- FSDP: Full regression (Slurm container, Slurm venv, EKS) — P5 only
- Megatron-LM: Container build only (no training)
- Everything else: No CI

## Tiered Implementation Plan

### Tier 1: Instance Profile Documentation (LOW effort, HIGH ROI)

**Goal**: Every test case README has a "Tested Configurations" table. A central
document maps instance types to the key parameters that differ.

**Deliverables**:
1. `docs/instance-compatibility.md` at repo root — master reference
2. Per-test-case "Tested Configurations" table in each README
3. `docs/instance-profiles/` with one file per instance family

**Instance profile contents** (the 6 dimensions that matter):

| Dimension | Why It Matters | Example Difference |
|-----------|---------------|-------------------|
| GPU VRAM | FSDP strategy, offloading, TP, batch sizes | A10G 24GB needs FSDP2+offload; H100 80GB doesn't |
| GPUDirect RDMA | NCCL_PROTO, FI_EFA_USE_DEVICE_RDMA | g5: RDMA=0, PROTO=simple; p5: RDMA=1, PROTO=default |
| EFA count | Inter-node bandwidth | g5: 1 EFA; p5en: 32 EFA |
| NVLink topology | Intra-node TP efficiency | g5: no NVLink; p5: NVSwitch |
| Node CPU memory | Offloading feasibility | g5: 168Gi allocatable; p5en: ~700Gi |
| Storage size | Checkpoint frequency | 117GB/ckpt × save_freq=1 fills 1.2TB in 9 steps |

**Per-test-case table format**:
```markdown
## Tested Configurations

| Instance | GPUs | Model | Nodes | Key Settings | Status |
|----------|------|-------|-------|-------------|--------|
| p5en.48xlarge | 8×H200 80GB | Qwen3-8B | 4 | FSDP1, TP=2 | ✅ Tested |
| g5.12xlarge | 4×A10G 24GB | gpt-oss-20b | 3+1 head | FSDP2, offload, TP=4 | ✅ Tested |
| p4de.24xlarge | 8×A100 80GB | Qwen3-8B | 4 | FSDP1, TP=2 | 🔲 Untested |
```

**Effort**: ~2-3 days. Pure documentation, no code changes.

### Tier 2: Parameterized Config Profiles (MEDIUM effort, MEDIUM ROI)

**Goal**: Scripts auto-detect instance type and source the right profile. Users
can also explicitly select a profile.

**Deliverables**:
1. `profiles/` directory per test case with `.env` files per instance type
2. Auto-detection helper script using EC2 instance metadata
3. Updated recipe scripts that source profiles

**Profile structure** (using veRL as the template):
```
recipe/
├── run_grpo.sh                    # Main script — sources profile
├── profiles/
│   ├── _detect.sh                 # Auto-detect instance type
│   ├── p5en-48xlarge.env          # FSDP1, no offload, TP=2
│   ├── g5-12xlarge.env            # FSDP2, full offload, TP=4
│   ├── p4de-24xlarge.env          # FSDP1, ref offload, TP=2
│   └── README.md                  # How to create a new profile
```

**Auto-detection** (`_detect.sh`):
```bash
#!/bin/bash
# Detect from EC2 metadata (works on bare metal and K8s with host networking)
INSTANCE_TYPE=$(curl -s --connect-timeout 2 \
  http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null)

# Fallback: detect from GPU type
if [ -z "$INSTANCE_TYPE" ]; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  case "$GPU_NAME" in
    *A10G*)  INSTANCE_TYPE="g5.12xlarge" ;;
    *A100*)  INSTANCE_TYPE="p4de.24xlarge" ;;
    *H100*)  INSTANCE_TYPE="p5.48xlarge" ;;
    *H200*)  INSTANCE_TYPE="p5en.48xlarge" ;;
    *L40S*)  INSTANCE_TYPE="g6e.12xlarge" ;;
  esac
fi

PROFILE_NAME="${INSTANCE_TYPE//./-}"
echo "$PROFILE_NAME"
```

**Profile file** (`profiles/g5-12xlarge.env`):
```bash
# g5.12xlarge — 4× A10G 24GB, 1 EFA, no GPUDirect RDMA
# Requires aggressive CPU offloading for models >10B params

# FSDP
export ACTOR_STRATEGY=fsdp2
export OFFLOAD_POLICY=True
export MODEL_DTYPE=bf16
export PARAM_OFFLOAD=True
export OPTIMIZER_OFFLOAD=True
export RESHARD_AFTER_FORWARD=True

# vLLM
export GPU_MEMORY_UTILIZATION=0.6
export ENFORCE_EAGER=True
export TENSOR_PARALLEL_SIZE=4

# Network
export NCCL_PROTO=simple
export FI_EFA_USE_DEVICE_RDMA=0

# Training
export NUM_GPU_PER_NODE=4
export WORKER_MEMORY=150Gi
export MAX_RESPONSE_LENGTH=256

# Checkpoints (24GB GPUs → full offload → large checkpoints)
export SAVE_FREQ=20
export MAX_CKPT_TO_KEEP=3
```

**Effort**: ~1 week. Requires touching each test case's run scripts.

### Tier 3: Automated Multi-Instance Validation (HIGH effort, LONG-TERM ROI)

**Goal**: CI validates test cases across multiple instance types. Catches
regressions before they reach users.

**Deliverables**:
1. Smoke test framework — 2-step train + checkpoint for each config
2. CI matrix expansion (instance type × model size × framework)
3. Memory profiling artifacts (peak GPU at init/fwd/bwd/ckpt)
4. Performance baseline tracking

**Smoke test spec**:
```yaml
# .github/test-matrix.yml
smoke_tests:
  - test_case: pytorch/verl
    configs:
      - instance: p5en.48xlarge
        model: Qwen3-8B
        profile: p5en-48xlarge
        steps: 2
        expected_peak_gpu_gb: 45
      - instance: g5.12xlarge
        model: gpt-oss-20b
        profile: g5-12xlarge
        steps: 2
        expected_peak_gpu_gb: 22
  - test_case: pytorch/FSDP
    configs:
      - instance: p5en.48xlarge
        model: llama3_1_70b
        steps: 5
      - instance: g5.12xlarge
        model: llama3_1_8b
        steps: 5
```

**CI expansion**: Extend `fsdp-regression-test-container.yml` pattern:
```yaml
strategy:
  matrix:
    cluster: [p5-eks, g5-eks, p4de-slurm]
    model: [llama3_1_8b, llama3_1_70b]
    exclude:
      - cluster: g5-eks
        model: llama3_1_70b  # Won't fit without offloading
```

**Memory profiling**: Capture at each phase and store as artifact:
```python
# Inserted at key points in training loop
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"MEMORY_PROFILE phase=init peak_gb={peak:.2f}")
torch.cuda.reset_peak_memory_stats()
```

**Effort**: ~2-4 weeks. Requires CI infra for non-P5 clusters.

## Implementation Order

| Phase | Tier | Scope | Effort | Prereq |
|-------|------|-------|--------|--------|
| 1 | Tier 1 | Central docs + veRL READMEs | 2-3 days | None |
| 2 | Tier 1 | Remaining test case READMEs | 2-3 days | Phase 1 |
| 3 | Tier 2 | veRL profile system (template) | 2-3 days | Phase 1 |
| 4 | Tier 2 | FSDP profile system | 2-3 days | Phase 3 |
| 5 | Tier 2 | Remaining test cases | 1 week | Phase 4 |
| 6 | Tier 3 | Smoke test framework | 1 week | Phase 3 |
| 7 | Tier 3 | CI matrix expansion | 1-2 weeks | Phase 6 |

## Key Lessons from veRL g5 Experience (informing the profiles)

These are the specific parameters that differ by instance type, discovered
through 11 OOM iterations:

1. **FSDP2 vs FSDP1**: FSDP1 explicitly disables CPUOffload for actor role.
   On 24GB GPUs, this is fatal for models >10B params.

2. **offload_policy=True**: FSDP2-specific flag. Without it, actor stays on GPU.

3. **model_dtype=bf16**: veRL defaults actor to fp32. On 80GB GPUs this wastes
   space; on 24GB GPUs it's an instant OOM.

4. **gpu_memory_utilization**: Fraction of TOTAL GPU, not just KV cache.
   0.3 × 23GB = 6.9GB < 10GB model shard → OOM.

5. **enforce_eager=True**: CUDA graphs need extra workspace → OOM on 24GB.

6. **NCCL_PROTO=simple**: Required when FI_EFA_USE_DEVICE_RDMA=0 (g5, g6e).

7. **Checkpoint size**: Full FSDP state for 20B model across 12 GPUs = 117GB.
   save_freq=1 on 1.2TB FSx → disk full in 9 steps.

8. **nnodes must exclude non-GPU head pod**: Ray head without GPUs in K8s
   causes NCCL hang if included in nnodes count.

9. **WORKER_MEMORY**: g5.12xlarge allocatable is ~168Gi, not 200Gi. Requesting
   200Gi causes pod scheduling failure.

10. **expandable_segments incompatible with vLLM**: CuMemAllocator asserts.

## How to Resume This Work in a New Session

Paste this into a new OpenCode session:

```
I'm implementing the Instance Compatibility Framework for the
awsome-distributed-training repo. The full plan is at:
- In-repo: docs/plans/instance-compatibility-framework.md
- Local: /tmp/instance-compatibility-plan.md

Start by reading the plan, then create branch
feat/instance-compatibility-framework and begin Phase 1 (Tier 1):
central docs + veRL README updates.

Key context:
- Repo: /Users/nchkumar/Code/smml-work/awsome-distributed-training/
- The veRL test case at 3.test_cases/pytorch/verl/ already has g5 guidance
  in its README (added this session)
- Training job raysubmit_CQxnr9aau2TFZ3E2 may still be running on the
  HyperPod cluster (check with kubectl)
- The 6 key dimensions for instance profiles: GPU VRAM, GPUDirect RDMA,
  EFA count, NVLink topology, Node CPU memory, Storage size
```
```

## Active Training Job (for monitoring)

- Job ID: raysubmit_CQxnr9aau2TFZ3E2 (v12)
- Config: FSDP2 + offload_policy + save_freq=20 + resume from step 9
- Check: kubectl exec rayml-efa-head-hlgcf -- bash -c "ray job logs raysubmit_CQxnr9aau2TFZ3E2 2>&1 | grep -E 'step:' | tail -5"
- Cluster cost: ~$28/hr (4× ml.g5.12xlarge)
