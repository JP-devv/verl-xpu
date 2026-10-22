#!/usr/bin/env bash
# verl → Intel Arc XPU: 2-GPU launch skeleton (cards 2,3 only).
#
# Cards 0,1 are the serving lane (Swift-Qwen-27B TP2) and must stay untouched.
# This script pins every process to the free pair and forces the XPU platform.
#
# Device pinning uses ZE_AFFINITY_MASK (the Level Zero analogue of
# CUDA_VISIBLE_DEVICES). ONEAPI_DEVICE_SELECTOR must stay UNSET — the two
# visibility mechanisms conflict.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
VENV="$REPO/.venv"

# --- GPU pinning: ONLY the two free cards (2,3) ------------------------------
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-2,3}"
unset ONEAPI_DEVICE_SELECTOR            # must NOT be set alongside ZE_AFFINITY_MASK
export VERL_PLATFORM=intel              # force the intel/xpu/xccl platform
#
# oneCCL (xccl) transport: this host has no usable OFI/RDMA provider, so the default
# OFI transport fails to init. Force the MPI transport: ONEAPI_ROOT points at the
# venv's bundled Intel MPI (satisfies oneCCL's check_impi_env) and I_MPI_FABRICS=shm
# uses shared memory for this single-node run. Launch with mpiexec (oneCCL needs a real
# MPI bootstrap; torchrun doesn't provide one) — see scripts/xpu/mpi-wrapper.sh.
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-mpi}"
export ONEAPI_ROOT="${ONEAPI_ROOT:-$VENV}"
export I_MPI_FABRICS="${I_MPI_FABRICS:-shm}"

# The venv is self-contained (its own sycl-rt 2025.2.1); do NOT source the
# system /opt/intel/oneapi/setvars.sh (2025.3) — it breaks torch 2.9.1.

# --- venv --------------------------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "venv missing at $VENV — create it first (see STAGE.md)" >&2
  exit 1
fi
PY="$VENV/bin/python"

# --- sanity: confirm we see exactly the two pinned cards ---------------------
"$PY" - <<'EOF'
import torch
assert torch.xpu.is_available(), "torch.xpu not available"
n = torch.xpu.device_count()
print(f"[xpu] torch {torch.__version__} | visible devices: {n} (expect 2 via ZE_AFFINITY_MASK)")
assert n == 2, f"expected 2 visible XPU devices, got {n}"
from verl.plugin.platform import get_platform
p = get_platform()
print(f"[platform] {p.device_name} / {p.vendor_name} / {p.communication_backend_name()}")
assert (p.device_name, p.vendor_name, p.communication_backend_name()) == ("xpu", "intel", "xccl"), \
    f"platform fell back: {p.device_name}/{p.vendor_name}/{p.communication_backend_name()}"
EOF

# --- launch ------------------------------------------------------------------
# Starting point: single node, 2 GPUs, FSDP engine. Replace with the real
# config (model path / batch sizes / algorithm) as needed.
exec "$PY" -m verl.trainer.main_ppo \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=2 \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/val.parquet \
  data.train_batch_size=16 \
  actor_rollout_ref.model.path="$HOME/models/swift-27b" \
  actor_rollout_ref.actor.strategy=fsdp \
  algorithm.adv_estimator=grpo \
  "$@"
