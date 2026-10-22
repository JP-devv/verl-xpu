#!/bin/bash
# Large all-reduce probe on the throwaway venv (torch 2.13.0+xpu + oneCCL 2022.0.0
# native XCCL). NO CCL_ATL_TRANSPORT=mpi — the 2022.0.0 native XCCL backend handles
# the transport directly. B70 CCL_SYCL block for the copy-engine path.
set -e
cd /home/haxor/AI/verl
export ZE_AFFINITY_MASK=2,3
unset ONEAPI_DEVICE_SELECTOR
export CCL_SYCL_ALLREDUCE_LL=ring
export CCL_SYCL_ALLREDUCE_SIMPLE_READ=1
export CCL_SYCL_COPY_ENGINE=1
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLTOALL_TMP_BUF=1
.venv-xpu2026/bin/torchrun --nproc_per_node=2 --master-port=29517 \
    scripts/xpu/test_large_allreduce.py
