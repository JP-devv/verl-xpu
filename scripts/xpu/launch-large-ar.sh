#!/bin/bash
# Large all-reduce probe on the free cards (2,3) via the native XCCL SYCL backend.
# Validates the fix for the oneCCL/shm large-all-reduce deadlock.
# NEVER touches cards 0,1 (serving). ONEAPI_DEVICE_SELECTOR stays unset.
set -e
cd /home/haxor/AI/verl

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-2,3}
unset ONEAPI_DEVICE_SELECTOR

# --- the working B70 CCL_SYCL device-backend block (from k8v4-pins) ---
export CCL_SYCL_ALLREDUCE_LL=ring
export CCL_SYCL_ALLREDUCE_SIMPLE_READ=1
export CCL_SYCL_COPY_ENGINE=1
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLTOALL_TMP_BUF=1

# This oneCCL build's CCL_ATL_TRANSPORT only accepts ofi|mpi (NO 'shm' transport;
# shm is a *fabric* inside the mpi transport). ofi has no provider on this host, so
# use the mpi transport + the shm fabric, and let the CCL_SYCL copy-engine (DMA) path
# move the data — that is what scales to large all-reduces without the compute-path
# deadlock. ONEAPI_ROOT satisfies oneCCL's check_impi_env.
export CCL_ATL_TRANSPORT=mpi
export I_MPI_FABRICS=shm
export ONEAPI_ROOT=/home/haxor/AI/verl/.venv

# mpiexec (NOT torchrun): the oneCCL mpi transport needs a real MPI bootstrap.
# mpi-ar-wrapper.sh maps PMI_* rank vars -> torch.distributed env:// vars.
.venv/bin/mpiexec -n 2 bash scripts/xpu/mpi-ar-wrapper.sh
