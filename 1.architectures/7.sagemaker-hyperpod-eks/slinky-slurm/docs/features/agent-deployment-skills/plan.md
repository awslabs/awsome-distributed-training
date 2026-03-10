---
id: agent-deployment-skills
status: done
started: 2026-03-09
completed: 2026-03-09
---

# Agent Deployment Skills & Validation Tests — Plan

## Steps

- [x] 1. Create `docs/features/agent-deployment-skills/idea.md`
- [x] 2. Migrate `bash-testing.md` to official SKILL.md format
- [x] 2a. Create `.opencode/skills/bash-testing/SKILL.md` with YAML frontmatter
- [x] 2b. Remove old `.opencode/skills/bash-testing.md` flat file
- [x] 3. Create `deployment-preflight` skill
- [x] 4. Create `deploy-infrastructure` skill
- [x] 5. Create `build-slurm-image` skill
- [x] 6. Create `deploy-slurm-cluster` skill
- [x] 7. Create `validate-deployment` skill
- [x] 8. Create `tests/test_setup.bats` (12 tests)
- [x] 9. Create `tests/test_install.bats` (10 tests)
- [x] 10. Create `tests/test_destroy.bats` (7 tests)
- [x] 11. Run `bats tests/` — all 89 tests pass (45 deploy + 12 setup + 10 install + 7 destroy + 15 post-session updates)

## Completed: Skill Migration

Migrated the existing `bash-testing` skill from a flat markdown file to the
official OpenCode SKILL.md format:

```
.opencode/skills/bash-testing.md  -->  .opencode/skills/bash-testing/SKILL.md
```

Added YAML frontmatter:
```yaml
name: bash-testing
description: Patterns for unit testing bash scripts using bats-core, including
  AWS CLI mocking, jq/sed/awk testing, and cross-platform portability
```

Content preserved identically; only the file location and frontmatter changed.

## Completed: 5 New Agent Skills

All skills follow the `.opencode/skills/<name>/SKILL.md` format with YAML
frontmatter (`name`, `description`) and a consistent structure: Overview,
Prerequisites, Steps, Verification, Troubleshooting, References.

### Skill 1: `deployment-preflight`

Validates all prerequisites before running any deployment script. Organized
by deployment phase:

- **Phase 1** (`deploy.sh`): `aws`, `jq`/`terraform`, valid credentials,
  valid AZ ID
- **Phase 2** (`setup.sh`): `docker` (local build), `zip` (CodeBuild),
  ECR image (skip build)
- **Phase 3** (`install.sh`): `kubectl`, `helm`, `curl`, correct kubeconfig
  context, `slurm-values.yaml` exists

Includes a 6-step full preflight validation procedure and a troubleshooting
table with 8 common failure scenarios.

### Skill 2: `deploy-infrastructure`

Guides agents through `deploy.sh` for both CFN and TF paths:

- Node type selection (g5 vs p5 with instance type/count table)
- Example commands for all common invocations
- Internal workflow documentation (10 steps for CFN, 5 for TF)
- Post-deployment steps: `source env_vars.sh`, `aws eks update-kubeconfig`
- Verification: `kubectl cluster-info`, `kubectl get nodes`
- Troubleshooting: 6 failure scenarios

### Skill 3: `build-slurm-image`

Guides agents through `setup.sh` with all three image build paths:

- **CodeBuild (default)**: S3 bucket creation, build context packaging,
  CodeBuild stack deployment (CFN or TF), build trigger and polling
- **Local Docker** (`--local-build`): DLC ECR auth, platform-aware build
  (`docker buildx` on macOS, `docker build` on Linux), ECR push
- **Skip Build** (`--skip-build`): ECR image verification

Also documents SSH key generation, `resolve_helm_profile()` variable table
(7 variables for g5/p5), and the 10-variable template substitution.

### Skill 4: `deploy-slurm-cluster`

Guides agents through `install.sh`:

- Full install vs `--skip-setup` modes
- Internal phases A-E with exact Helm/kubectl commands for each:
  - A: Setup (calls `setup.sh` with pass-through flags)
  - B: MariaDB operator (v25.10.4) + MariaDB CR
  - C: Slurm operator (v1.0.1 OCI chart) + stale CRD cleanup
  - D: Slurm cluster (v1.0.1 OCI chart with values)
  - E: NLB configuration (IP detection, service patch, endpoint wait)
- Troubleshooting: 8 failure scenarios

### Skill 5: `validate-deployment`

Post-deployment health checks with 7 sequential steps:

1. Kubernetes pod health across 3 namespaces
2. Node health and instance type verification
3. Login service and NLB endpoint verification
4. SSH connectivity test
5. Slurm service validation (`sinfo`, `scontrol`, `sacctmgr`)
6. Basic test job (`srun hostname`, `sbatch --wrap`)
7. Optional: profile-specific Llama2 training job

Includes a ready-to-run quick validation script and a key logs reference
section for debugging.

## Completed: New Bats Tests (29 tests)

### `tests/test_setup.bats` — 12 tests

| Category | Count | Tests |
|----------|-------|-------|
| Argument parsing | 7 | `--help`, missing `--node-type`, missing `--infra`, invalid `--infra`, unknown option, `--skip-build` in usage, `--local-build` in usage |
| `resolve_helm_profile` | 3 | g5 sets all 7 variables, p5 overrides, invalid type |
| Template substitution | 2 | g5 no unresolved vars, p5 correct GPU/EFA/replicas |

### `tests/test_install.bats` — 10 tests

| Category | Count | Tests |
|----------|-------|-------|
| Argument parsing | 4 | `--help`, `--skip-setup` mentioned, pass-through options mentioned, unknown option |
| Version constants | 2 | cert-manager version, LB Controller chart version |
| Flag validation | 1 | `--skip-setup` fails when `slurm-values.yaml` missing |
| Install order | 1 | cert-manager before LB Controller before MariaDB |
| env_vars.sh deps | 2 | EKS_CLUSTER_NAME and VPC_ID sourced from env_vars.sh |

### `tests/test_destroy.bats` — 7 tests

| Category | Count | Tests |
|----------|-------|-------|
| Argument parsing | 4 | `--help`, missing `--infra`, invalid `--infra`, unknown option |
| Optional flags | 2 | `--region` in usage, `--stack-name` in usage |
| Non-interactive | 1 | Aborts when confirmation prompt answered with 'n' |

### Test Results

```
89 tests, 0 failures
```

All 45 existing tests in `test_deploy.bats` continue to pass. The 29 new
tests across the 3 new files all pass. An additional 15 tests were added
in later sessions for version constants, install order, env_vars.sh
dependency validation, and template substitution coverage.

## Deliverables

```
# Migrated skill
.opencode/skills/bash-testing/SKILL.md

# New skills
.opencode/skills/deployment-preflight/SKILL.md
.opencode/skills/deploy-infrastructure/SKILL.md
.opencode/skills/build-slurm-image/SKILL.md
.opencode/skills/deploy-slurm-cluster/SKILL.md
.opencode/skills/validate-deployment/SKILL.md

# New tests
tests/test_setup.bats    (12 tests)
tests/test_install.bats  (10 tests)
tests/test_destroy.bats  (7 tests)

# Feature docs
docs/features/agent-deployment-skills/idea.md
docs/features/agent-deployment-skills/plan.md
```

## Agent Deployment Workflow

With these skills, a future agent can deploy slinky-slurm end-to-end:

```
1. Load skill: deployment-preflight
   -> Validate CLI tools, AWS creds, region, AZ

2. Load skill: deploy-infrastructure
   -> Run deploy.sh --node-type <g5|p5> --infra <cfn|tf>
   -> Source env_vars.sh, update kubeconfig

3. Load skill: build-slurm-image
   -> Run setup.sh (CodeBuild, local Docker, or skip)
   -> Verify ECR image, slurm-values.yaml, SSH key

4. Load skill: deploy-slurm-cluster
   -> Run install.sh (or install.sh --skip-setup if step 3 done separately)
   -> Wait for NLB endpoint

5. Load skill: validate-deployment
   -> Check pods, nodes, Slurm services, SSH, test job
```
