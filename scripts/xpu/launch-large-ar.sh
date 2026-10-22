#!/bin/bash
# Large all-reduce probe on the free cards (2,3) via the native XCCL SYCL backend.
# Validates the fix for the oneCCL/shm large-all-reduce deadlock.
# NEVER touches cards 0,1 (serving). ONEAPI_DEVICE_SELECTOR stays unset.
set -e
cd /home/haxor/AI/verl

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-2,3}
unset ONEAPI_DEVICE_SELECTOR

# --- the working B70 CCL_SYCL block (from k8v4-pins) ---
export CCL_SYCL_ALLREDUCE_LL=ring
export CCL_SYCL_ALLREDUCE_SIMPLE_READ=1
export CCL_SYCL_COPY_ENGINE=1
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLTOALL_TMP_BUF=1

# Do NOT set CCL_ATL_TRANSPORT=mpi / I_MPI_FABRICS=shm here — that is the
# host-memory path that deadlocks. Let the native xccl SYCL device backend run.

.venv/bin/torchrun --nproc_per_node=2 --master-port=${MASTER_PORT:-29511} \
    scripts/xpu/test_large_allreduce.py
