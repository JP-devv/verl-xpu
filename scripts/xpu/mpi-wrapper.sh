#!/usr/bin/env bash
# Wrapper: map Intel-MPI rank vars (set by mpiexec) to torch.distributed env:// vars,
# then exec the verl SFT trainer. Used because oneCCL's xccl MPI transport needs a
# real mpiexec bootstrap (torchrun does not provide one on this host).
set -euo pipefail

export RANK="${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}"
export WORLD_SIZE="${PMI_SIZE:-${OMPI_COMM_WORLD_SIZE:-1}}"
export LOCAL_RANK="${PMI_LOCAL_RANK:-${PMI_RANK:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

PY="${PY:-/home/haxor/AI/verl/.venv/bin/python}"
exec "$PY" -m verl.trainer.sft_trainer "$@"
