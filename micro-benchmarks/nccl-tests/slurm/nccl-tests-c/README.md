# NCCL Send/Recv Ring Test

A minimal NCCL point-to-point benchmark that measures send/recv latency and bandwidth across GPUs in a ring topology. Designed to run on AWS SageMaker HyperPod clusters with Slurm and EFA.

## What it does

Each MPI rank sends a 64 MB buffer to its right neighbour and receives from its left neighbour in a ring pattern using `ncclSend`/`ncclRecv` grouped calls. After a warm-up phase, it runs 100 timed iterations and reports average latency and bandwidth.

## Prerequisites

- CUDA toolkit
- NCCL library
- MPI (OpenMPI is pre-installed on HyperPod at `/opt/amazon/openmpi`)
- At least 2 GPUs

## Building

The Slurm script builds automatically before running. To build manually:

```bash
make
```

Override paths if needed:

```bash
make CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr MPI_HOME=/opt/amazon/openmpi
```

## Running on SageMaker HyperPod (Slurm)

Submit the job:

```bash
sbatch run_nccl_test.sbatch
```

Monitor output:

```bash
squeue -u $USER
tail -f nccl_test_<jobid>.out
```

### Customizing the Slurm job

Edit `run_nccl_test.sbatch` to adjust:

| Parameter | Default | Description |
|---|---|---|
| `--nodes` | 2 | Number of nodes |
| `--ntasks-per-node` | 8 | MPI ranks per node (one per GPU) |
| `--gpus-per-node` | 8 | GPUs per node (8 for p4d/p5, 4 for p3dn) |
| `--partition` | not set | Add if your cluster uses named partitions |

## Running locally with mpirun

```bash
make
mpirun -np 2 ./nccl_sendrecv_test
```

## Configuration

Compile-time constants in `nccl_sendrecv_test.c`:

| Define | Default | Description |
|---|---|---|
| `MSG_SIZE` | 64 MB | Buffer size per send/recv |
| `ITERATIONS` | 100 | Timed iterations |
| `WARMUP_ITERATIONS` | 10 | Untimed warm-up iterations |

## Example output

```
---------- NCCL Send/Recv Ring Test ----------
Ranks           : 16
Message size    : 67108864 bytes (64 MB)
Iterations      : 100
Avg latency     : 2.450 ms
Algo BW (1-dir) : 27.39 GB/s
Bus BW (bidir)  : 54.78 GB/s
-----------------------------------------------
```
