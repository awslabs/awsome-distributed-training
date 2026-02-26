# Automated Customizable Health Check for Amazon SageMaker HyperPod SLURM Clusters


Open Questions on Product


1. We are unsure about wording and naming, you might see different wording in this doc (Leader node, orchestrator, worker job, child job. we haven’t unify them), they refers to the similar thing, prefer Product team to provide guidance as it will impact positions of this feature.
2. Need opinion on summary format of health check. It’s the customer facing log which delivers final health check results to customer.
3. For child jobs, customer might want to run somthing like nccl tests which takes multiple node as input, we can also support node grouping when triggering sub jobs. Currently we make default 
4. For Prolog script, do we want to trigger remediation if prolog detected failure? 
5. Need opinion on remediation logic on DCGM tests






[This doc is based on [GPU Health Check (DCGM) on Amazon SageMaker HyperPod Slurm Clusters](https://quip-amazon.com/Q7T0AfdW1Z8L) drafted by scripts author [Yuxuan Zhao](https://quip-amazon.com/RSK9EA803Rk)]

The Health Check Orchestrator is a flexible framework that automates health checks across your HyperPod Slurm cluster. It operates as a Slurm job, spawning child Slurm jobs on targeted worker nodes to run health check tests. Based on the collected results, it updates Slurm node features and automatically triggers remediation when health issues are identified.

A DCGM diagnostic script is provided as a reference implementation; however, the orchestrator is compatible with any script that adheres to the simple output contract. 

Additionally, a prolog script is included to configure DCGM diagnostics as a Slurm prolog, enabling automatic health checks before each job begins.



[TOC in markdown] (Quip support support table of content)



## **Key Capabilities**

* **Bring Your Own Health Checks** — The orchestrator is script-agnostic. Write your own diagnostic scripts and run them through the framework with minimal configuration.A DCGM diagnostic script is provided as a reference implementation.
* **Automatic Node Tagging** — After each check, nodes are automatically tagged with a Slurm feature (`HealthCheck:Passed`, `HealthCheck:Failed`, or `HealthCheck:Skipped`), allowing users to easily target healthy nodes when submitting jobs.
* **Automatic Remediation** — When a node fails a health check, it is automatically marked as failed and flagged for reboot or replacement—no manual intervention required.
* **Parallel Execution** — Health checks are distributed and executed simultaneously across all targeted nodes, significantly reducing end-to-end diagnostic time.
* **Flexible Targeting** — Choose your scope: run checks on specific nodes, across an entire partition, or within a particular instance group.
* **Prolog Integration Example** — A ready-to-use Slurm prolog script is provided to configure DCGM diagnostics as automatic pre-job health checks, ensuring nodes are verified before workloads are scheduled.

```
┌────────────────────────────────────────────────────────────────────┐
│                          SLURM CLUSTER                             │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    MainJob (Single Node)                     │  │
│  │                                                              │  │
│  │  1. Submit worker jobs ──────────────────┐                   │  │
│  │                                          │                   │  │
│  │                                          ▼                   │  │
│  │       ┌──────────────────────────────────────────┐           │  │
│  │       │        Worker Slurm Job (Target Nodes)   │           │  │
│  │       │                                          │           │  │
│  │       │   ┌────────┐  ┌────────┐  ┌────────┐     │           │  │
│  │       │   │ Node A │  │ Node B │  │ Node C │     │           │  │
│  │       │   │  test  │  │  test  │  │  test  │     │           │  │
│  │       │   └────────┘  └────────┘  └────────┘     │           │  │
│  │       │                                          │           │  │
│  │       └──────────────────────────────────────────┘           │  │
│  │              ▲            ▲            ▲                     │  │
│  │  2. Poll     │            │            │                     │  │
│  │  for results │            │            │                     │  │
│  │  ────────────┘────────────┘────────────┘                     │  │
│  │                                                              │  │
│  │  3. Process results & update Slurm features                  │  │
│  │                                                              │  │
│  │  4. Apply remediation (if needed)                            │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

Before getting started, ensure:

1. **SSH Access** — You have SSH access to your HyperPod Slurm cluster nodes. Ensure you have the necessary permissions to update node states via `scontrol`, attach features to nodes, and submit jobs using `sbatch`.
2. **jq Installed** — The `jq` command-line tool is installed on both the head node and all compute nodes.
3. **Shared Filesystem** — A shared filesystem (e.g., `/fsx`) is accessible from all nodes for storing scripts and health check results.
4. **Health Check Script** — You have a custom health check script ready, or you can use the provided `dcgm.sh` script for GPU diagnostics.



## Quick Start

### Step 1: Copy Scripts to Shared Storage

Place the orchestrator and your health check script on shared storage:

```
`cd ``/``fsx``/``ubuntu`
`git clone https``:``//github.com/awslabs/awsome-distributed-training.git `
`cd awsome``-``distributed``-``training``/``1.architectures``/``5.sagemaker``-``hyperpod``/``tools``/``health``-``check`
`chmod +x health_check_orchestrator.sh 
chmod +x dcgm.sh`
```



### Step 2: Submit the Health Check Job

```
sbatch --output=/fsx/ubuntu/health-check-results/health-check-orchestrator-%x-%j.out health_check_orchestrator.sh \
--target-partition ml.g5.xlarge \
--output-dir /fsx/ubuntu/health-check-results \
--test-script /fsx/ubuntu/dcgm.sh \
--test-script-args '{"level": 2}'
```

What this command does:

|Flag	|Value	|Meaning	|
|---	|---	|---	|
|`--target-partition`	|`ml.g5.xlarge`	|Test all available nodes in the `ml.g5.xlarge` partition	|
|`--output-dir`	|`/fsx/ubuntu/health-check-results`	|Save all logs and results of child slurm jobs to this directory	|
|`--output`	|`/fsx/ubuntu/health-check-results/health-check-orchestrator-%x-%j.out`	|Instruct Slurm to connect the batch script's standard error directly to file `health-check-orchestrator-<jobname>-<jobid>.out`	|
|`--test-script`	|`dcgm.sh`	|Use `dcgm.sh` as the health-check script on each target compute node	|
|`--test-script-args`	|`'{"level": 2}'`	|Run DCGM Level 2 diagnostics (quick check)	|

You'll see output like:

```
Submitted batch job <job-id>
```



### Step 3: Monitor Progress

Check health check orchestrator slurm job status:

```
scontrol show job <job-id>
```



### Step 4: Review Results

Once the health check orchestra job reached terminal status, check the output directory:

```
ls /fsx/ubuntu/health-check-results/
```

If the  health check orchestra job completed, you’ll find

* `health-check-orchestrator-health_check_main_job-<job-id>.log`: The orchestrator's main log with the overall summary.
* `worker_<node-name>_<timestamp>.log`: Output from the worker job running on each node.
* `health_check_summary_<job-id>_<timestamp>.txt`: One-line-per-node results.
    

Open the summary log to see the summary of test results on target compute nodes:

```
# cat health_check_summary_<job-id>_<timestamp>.log

==========================================
=== Health Check Summary ===
==========================================
Overall Status: Passed
Remediation applied: true
Nodes PASSED: ip-10-1-4-240 ip-10-1-52-200 ip-10-1-114-79
Output directory: /fsx/ubuntu/health-check-results
Per-node worker logs:
  ip-10-1-4-240: /fsx/ubuntu/health-check-results/worker_ip-10-1-4-240_20260225_202125.log (exit code 0)
  ip-10-1-52-200: /fsx/ubuntu/health-check-results/worker_ip-10-1-52-200_20260225_202125.log (exit code 0)
  ip-10-1-114-79: /fsx/ubuntu/health-check-results/worker_ip-10-1-114-79_20260225_202125.log (exit code 0)
==========================================
```

If any node fails, you'll see similar logs like this:

```
==========================================
=== Health Check Summary ===
==========================================
Overall Status: Failed
Remediation applied: true
Nodes PASSED: ip-10-1-114-79
Nodes SKIPPED: ip-10-1-52-200
  ip-10-1-52-200 (Skipped): Simulated dcgmi timeout after 1800s
Nodes requiring REBOOT: ip-10-1-4-240
  ip-10-1-4-240 (Failed): Simulated GPU isolation failure (DCGM ISOLATE severity)
Output directory: /fsx/ubuntu/dhc
Per-node worker logs:
  ip-10-1-4-240: /fsx/ubuntu/dhc/worker_ip-10-1-4-240_20260225_192603.log (exit code 0)
  ip-10-1-52-200: /fsx/ubuntu/dhc/worker_ip-10-1-52-200_20260225_192603.log (exit code 0)
  ip-10-1-114-79: /fsx/ubuntu/dhc/worker_ip-10-1-114-79_20260225_192603.log (exit code 0)
==========================================
```



### Step 5: Proceed your Slurm workloads on Healthy Nodes


After the health check completes, each node is tagged with a Slurm feature (`HealthCheck:Passed`, `Failed`, or `Skipped`). Use Slurm's `--constraint` (`-C`) flag to schedule jobs only on nodes that passed, it matches the `--constraint` value against each node's ActiveFeatures, so only nodes whose health check result is `Passed` will be eligible for your job.

Submit a batch job on healthy nodes only:

```
sbatch -C "HealthCheck:Passed" -N 2 my_training_job.sh
```

Combine with a partition:

```
sbatch -p ml.g5.xlarge  -C "HealthCheck:Passed" -N 4 my_training_job.sh
```



## Customize Your Own Health Check

While DCGM diagnostics are provided as a reference, the `health_check_orchestrator.sh` is designed to be script-agnostic. You can plug in any custom health check script — it just needs to adhere to a simple output contract.

### Output Contract

Your script **must** print exactly one line to stdout in this format:

```
HEALTH_CHECK_RESULT:<host-name>:<status>:<remediation>:<reason>
```

|Field	|Required	|Valid Values	|Description	|
|---	|---	|---	|---	|
|`host-name`	|Yes	|The node's hostname (use `$(hostname)`)	|Identifies which slurm compute node 	|
|`status`	|Yes	|`Passed`, `Failed`, `Skipped`	|`Passed` = healthy, `Failed` = health check failed, `Skipped` = test execution failed	|
|`remediation`	|Yes	|`none`, `reboot`, `replace`	|What action to take: `none` = no action, `reboot` = reboot the node, `replace` = replace the node	|
|`reason`	|Optional	|Free-form text (no newlines)	|Human-readable explanation of the failure	|

### How Each Status Is Handled

|Status	|Slurm Feature Set	|Remediation Applied?	|Node State	|
|---	|---	|---	|---	|
|`Passed`	|`HealthCheck:Passed`	|No	|Stays available	|
|`Failed`	|`HealthCheck:Failed`	|Yes (if `--remediate true`)	|Set to `FAIL` with reason	|
|`Skipped`	|`HealthCheck:Skipped`	|No (inconclusive)	|Stays available	|

### Environment Variables Available to Your Script

The orchestrator passes these environment variables to your script:

|Variable	|Description	|Example	|
|---	|---	|---	|
|`HC_TEST_PARAMS`	|The JSON string from `--test-script-args`	|`{"level": 2}`	|
|`HC_RESULTS_DIR`	|The output directory (same as `--output-dir`)	|`/fsx/ubuntu/health-check`	|
|`HC_TIMESTAMP`	|A timestamp string for file naming	|`20260223_200355`	|



## Configuration Options 

### Orchestrator Slurm Job Options 

### Target Selection (Specify exactly one of the following options)

|Flag	|Description	|Example	|
|---	|---	|---	|
|`--target-nodes`	|Specific node names (comma-separated or Slurm expression)	|`ip-10-0-1-100,ip-10-0-1-101`	|
|`--target-partition`	|All nodes in a Slurm partition	|`ml.g5.xlarge`	|
|`--instance-group`	|All nodes in a HyperPod instance group	|`worker-group-1`	|

### Required Flags

|Flag	|Description	|Example	|
|---	|---	|---	|
|`--test-script`	|Path to the health-check script	|`/fsx/ubuntu/dcgm.sh`	|
|`--output-dir`	|Absolute path for logs and results	|`/fsx/ubuntu/dhc`	|

### Optional Flags

|Flag	|Description	|Example	|
|---	|---	|---	|
|`--test-script-args`	|JSON object of parameters for the test script	|'{"level": 3}'	|
|`--remediate`	|Apply automatic remediation (true or false)	|TRUE	|

### Leader Node Exclusion

If the orchestrator job runs on a node that is also included in the target list (e.g., when targeting an entire partition that contains the leader node), it automatically excludes that node from the health check targets. This is because a worker job cannot be scheduled on a node already occupied by the orchestrator. A warning is logged when this occurs:

```
WARNING: Leader node ip-10-1-52-200 is in target list — excluding to avoid deadlock
```



### DCGM-Specific Parameters (passed via `-`**`-test-script-args`**)

|JSON Key	|Description	|Default	|Valid Values	|
|---	|---	|---	|---	|
|`--level`	|DCGM diagnostic level	|4	|`2`, `3`, `4`	|

DCGM Levels Explained:

|Level	|Duration	|What It Tests	|
|---	|---	|---	|
|2	|~2 minutes	|Quick hardware checks — memory, PCIe, basic GPU health. Best for routine pre-job checks.	|
|3	|~10–15 minutes	|Medium diagnostics — adds stress tests for GPU compute and memory bandwidth.	|
|4	|Up to ~3 hours	|Full diagnostics — comprehensive stress testing including NVLink, PCIe bandwidth, thermal stress. Best for initial cluster validation.	|



### More examples

Test two specific nodes with Level 3 diagnostics:

```
sbatch --output=/fsx/ubuntu/dhc/leader_%j.log health_check_orchestrator.sh \
--target-nodes ip-10-1-114-79,ip-10-1-4-240 \
--output-dir /fsx/ubuntu/dhc \
--test-script /fsx/ubuntu/dcgm.sh \
--test-script-args '{"level": 3}'
```

Test an instance group with remediation disabled (dry-run) and specify which is the leader node:

```
sbatch --nodelist=ip-10-1-4-240 --output=/fsx/ubuntu/dhc/leader_%j.log health_check_orchestrator.sh \
--instance-group worker-group-1 \
--output-dir /fsx/ubuntu/dhc \
--test-script /fsx/ubuntu/mock_one_fail.sh \
--test-script-args '{"level": 2}' \
--remediate false
```

Test a partition with a custom timeout:

```
sbatch --time=240 --output=/fsx/ubuntu/dhc/leader_%j.log health_check_orchestrator.sh \
--target-partition ml.g5.xlarge \
--output-dir /fsx/ubuntu/dhc \
--test-script /fsx/ubuntu/dcgm.sh
```



## Automatic Pre-Job DCGM Checks (Slurm Prolog)

If you want every job to automatically verify GPU health before it starts, you can configure the Slurm Prolog. This runs a DCGM Level 2 check each time a node is allocated to a job. If the check fails, the node is drained and the job is re-queued to a healthy node.


### Step 1: Place Scripts on Shared Storage

Both `prolog_dcgm.sh` and `dcgm.sh` must be in the **same directory** on the shared filesystem, and both must be executable:

### Step 2: Edit Slurm Configuration

Open the Slurm configuration file (typically at `/opt/slurm/etc/slurm.conf`) and add these two lines:

```
Prolog=/fsx/ubuntu/prolog_dcgm.sh
PrologFlags=Alloc
```

* `Prolog`: Points to the prolog script on the shared filesystem.
* `PrologFlags=Alloc`: Tells Slurm to run the prolog only once per allocation on each node (not once per task).

### Step 3: Restart/Reconfigure Slurm

Apply the configuration change:

```
srun -N1 hostname
```

Check the prolog logs:

```
ls /fsx/ubuntu/prolog_logs/
cat /fsx/ubuntu/prolog_logs/<hostname>_prolog_<job_id>.log
```

### What Happens When You Submit a Job

* Slurm allocates a node to your job.
* Before your job starts, `prolog_dcgm.sh` runs on the allocated node.
* It first checks a local cache — if the node passed DCGM within the last hour, it skips the test and your job starts immediately without delay.
* If no valid cache exists, it runs a DCGM Level 2 diagnostic (~2 minutes).
* If the test passes — your job starts normally.
* If the test fails — the node can be drained (if DRAIN_ON_FAILURE set to true), and Slurm blocked your actual job from running.
* If the test is inconclusive (e.g., dcgmi crashed) — the node is marked Skipped and your job proceeds (conservative approach to avoid blocking jobs).

### Configurable Defaults

You can customize the prolog behavior by editing the top of `prolog_dcgm.sh`:

```
readonly DCGM_LEVEL=2                        # DCGM diagnostic level (2-4)
readonly CACHE_TTL_HOURS=1                   # If a cached result exists and is less than CACHE_TTL_HOURS hours old, skip the prolog check. Set to 0 to disable caching entirely.
readonly UPDATE_FEATURES=true                # Update Slurm node features
readonly DRAIN_ON_FAILURE=false              # If true, exit 1 on failure so Slurm drains the node and requeues the job. If false, only update the feature to Failed and let the job proceed.
readonly PROLOG_BASE_DIR="/fsx/ubuntu/health_check_prolog"
readonly LOG_DIR="${PROLOG_BASE_DIR}/logs"
readonly CACHE_DIR="${PROLOG_BASE_DIR}/cache"
```

Level 2 is recommended for the prolog because it takes only ~2 minutes. Higher levels (3 or 4) would delay job start significantly.
