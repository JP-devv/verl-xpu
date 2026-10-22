#!/bin/bash
# Map Intel-MPI rank vars (set by mpiexec) to torch.distributed env:// vars, then
# run the large-all-reduce probe. The oneCCL xccl MPI transport needs a real mpiexec
# bootstrap (torchrun does not provide one on this host -> "Duplicate ranks").
export RANK="${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}"
export WORLD_SIZE="${PMI_SIZE:-${OMPI_COMM_WORLD_SIZE:-1}}"
export LOCAL_RANK="${PMI_LOCAL_RANK:-${PMI_RANK:-0}}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29511}
exec /home/haxor/AI/verl/.venv/bin/python /home/haxor/AI/verl/scripts/xpu/test_large_allreduce.py
