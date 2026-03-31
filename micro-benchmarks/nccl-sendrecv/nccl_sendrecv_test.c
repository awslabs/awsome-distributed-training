/* Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

#define MSG_SIZE (64 * 1024 * 1024)  /* 64 MB */
#define ITERATIONS 100
#define WARMUP_ITERATIONS 10

#define CUDACHECK(cmd) do {                                 \
    cudaError_t err = (cmd);                                \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n",          \
                __FILE__, __LINE__,                         \
                cudaGetErrorString(err));                    \
        MPI_Abort(MPI_COMM_WORLD, 1);                      \
    }                                                       \
} while (0)

#define NCCLCHECK(cmd) do {                                 \
    ncclResult_t res = (cmd);                               \
    if (res != ncclSuccess) {                               \
        fprintf(stderr, "NCCL error %s:%d: %s\n",          \
                __FILE__, __LINE__,                         \
                ncclGetErrorString(res));                    \
        MPI_Abort(MPI_COMM_WORLD, 1);                      \
    }                                                       \
} while (0)

int main(int argc, char *argv[])
{
    int rank, nranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (nranks < 2) {
        if (rank == 0)
            fprintf(stderr, "Need at least 2 ranks\n");
        MPI_Finalize();
        return 1;
    }

    /* One GPU per MPI rank */
    int local_rank;
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    CUDACHECK(cudaSetDevice(local_rank));

    /* Create NCCL communicator from MPI ranks */
    ncclUniqueId nccl_id;
    if (rank == 0)
        NCCLCHECK(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, nranks, nccl_id, rank));

    /* Allocate device buffers */
    size_t msg_size = MSG_SIZE;
    void *send_buf, *recv_buf;
    CUDACHECK(cudaMalloc(&send_buf, msg_size));
    CUDACHECK(cudaMalloc(&recv_buf, msg_size));
    CUDACHECK(cudaMemset(send_buf, rank + 1, msg_size));
    CUDACHECK(cudaMemset(recv_buf, 0, msg_size));

    /* CUDA stream and events for timing */
    cudaStream_t stream;
    cudaEvent_t ev_start, ev_end;
    CUDACHECK(cudaStreamCreate(&stream));
    CUDACHECK(cudaEventCreate(&ev_start));
    CUDACHECK(cudaEventCreate(&ev_end));

    /* Ring neighbours */
    int send_peer = (rank + 1) % nranks;
    int recv_peer = (rank - 1 + nranks) % nranks;

    int iterations;

    /* Warm-up */
    iterations = WARMUP_ITERATIONS;
    for (int i = 0; i < iterations; i++) {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclSend(send_buf, msg_size, ncclChar,
                           send_peer, comm, stream));
        NCCLCHECK(ncclRecv(recv_buf, msg_size, ncclChar,
                           recv_peer, comm, stream));
        NCCLCHECK(ncclGroupEnd());
        CUDACHECK(cudaStreamSynchronize(stream));
    }

    /* Timed iterations */
    iterations = ITERATIONS;
    double total_ms = 0.0;

    for (int i = 0; i < iterations; i++) {
        MPI_Barrier(MPI_COMM_WORLD);

        CUDACHECK(cudaEventRecord(ev_start, stream));
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclSend(send_buf, msg_size, ncclChar,
                           send_peer, comm, stream));
        NCCLCHECK(ncclRecv(recv_buf, msg_size, ncclChar,
                           recv_peer, comm, stream));
        NCCLCHECK(ncclGroupEnd());
        CUDACHECK(cudaEventRecord(ev_end, stream));
        CUDACHECK(cudaEventSynchronize(ev_end));

        float ms;
        CUDACHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
        total_ms += (double)ms;
    }

    /* Use the slowest rank's total time for an accurate end-to-end picture */
    double max_total_ms;
    MPI_Reduce(&total_ms, &max_total_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Report from rank 0 */
    double avg_ms = max_total_ms / iterations;
    /* Algorithmic BW: single-direction throughput (msg_size / time) */
    double algo_bw_gbps = (double)msg_size / (avg_ms * 1e-3) / 1e9;
    /* Bidirectional throughput: 2x algo BW since each rank sends and receives
       simultaneously.  Note: this is NOT the same as nccl-tests "busBw", which
       normalises by the collective's communication pattern. */
    double bidir_gbps = (2.0 * msg_size) / (avg_ms * 1e-3) / 1e9;

    if (rank == 0) {
        printf("---------- NCCL Send/Recv Ring Test ----------\n");
        printf("Ranks           : %d\n", nranks);
        printf("Message size    : %zu bytes (%.0f MB)\n",
               msg_size, (double)msg_size / (1024.0 * 1024.0));
        printf("Iterations      : %d\n", iterations);
        printf("Avg latency     : %.3f ms\n", avg_ms);
        printf("Algo BW (1-dir) : %.2f GB/s\n", algo_bw_gbps);
        printf("Bidir throughput : %.2f GB/s\n", bidir_gbps);
        printf("-----------------------------------------------\n");
    }

    /* Cleanup */
    CUDACHECK(cudaEventDestroy(ev_start));
    CUDACHECK(cudaEventDestroy(ev_end));
    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaFree(send_buf));
    CUDACHECK(cudaFree(recv_buf));
    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
}
