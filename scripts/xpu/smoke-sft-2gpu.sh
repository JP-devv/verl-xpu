#!/usr/bin/env bash
# Minimal isolated verl XPU SFT smoke test on cards 2,3 ONLY.
#
# Purpose: prove the official verl-hardware-plugin end-to-end on the free pair —
#   1. get_platform() -> xpu/intel/xccl (plugin discovered),
#   2. the FSDP XPU engine runs a real forward+backward+xccl-grad-sync on 2 cards,
#   3. cards 0,1 (the serving lane) are never touched.
#
# Uses the tiny Qwen2.5-0.5B + a synthetic GSM8K-style parquet, so it is fast and
# memory-trivial. eager attention (the only XPU-validated attention path).
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
VENV="$REPO/.venv"
PY="$VENV/bin/python"

# --- GPU pinning: ONLY the two free cards (2,3) ------------------------------
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-2,3}"
unset ONEAPI_DEVICE_SELECTOR            # must NOT be set alongside ZE_AFFINITY_MASK
export VERL_PLATFORM=intel              # force the intel/xpu/xccl platform
#
# oneCCL (xccl) transport: this host has no usable OFI/RDMA provider (fi_getinfo
# returns 0), so the default OFI transport fails to init. Force the MPI transport
# instead. The venv bundles Intel MPI (IMPI); ONEAPI_ROOT points at it (satisfies
# oneCCL's check_impi_env) and I_MPI_FABRICS=shm makes IMPI use shared memory for
# this single-node run (no RDMA fabric). Processes are launched with mpiexec so
# oneCCL gets a real MPI bootstrap (torchrun does not provide one).
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-mpi}"
export ONEAPI_ROOT="${ONEAPI_ROOT:-$VENV}"
export I_MPI_FABRICS="${I_MPI_FABRICS:-shm}"

MODEL_PATH="${MODEL_PATH:-/tmp/xpu-sanity/qwen2.5-0.5b}"
DATA_DIR="${DATA_DIR:-/tmp/verl-xpu-smoke}"
SAVE_PATH="${SAVE_PATH:-/tmp/verl-xpu-smoke/ckpt}"

# --- preflight: 2 visible devices + platform is xpu/intel/xccl ---------------
"$PY" - <<'EOF'
import torch
assert torch.xpu.is_available(), "torch.xpu not available"
n = torch.xpu.device_count()
print(f"[preflight] visible XPU devices: {n}")
assert n == 2, f"expected exactly 2 visible XPU devices (ZE_AFFINITY_MASK=2,3), got {n}"
from verl.plugin.platform import get_platform
p = get_platform()
print(f"[preflight] platform: {p.device_name}/{p.vendor_name}/{p.communication_backend_name()}")
assert (p.device_name, p.vendor_name, p.communication_backend_name()) == ("xpu", "intel", "xccl")
print("[preflight] OK")
EOF

# --- launch: 2-GPU FSDP SFT via mpiexec (oneCCL MPI transport) --------------
# mpi-wrapper.sh maps IMPI's PMI_RANK/PMI_SIZE to torch.distributed's env:// vars
# (RANK/WORLD_SIZE/LOCAL_RANK/MASTER_*), then execs the SFT trainer.
exec "$VENV/bin/mpiexec" -n 2 \
  bash "$HERE/mpi-wrapper.sh" \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.messages_key=messages \
  data.micro_batch_size_per_gpu=4 \
  data.train_batch_size=8 \
  data.max_length=512 \
  optim.lr=1e-4 \
  engine=fsdp \
  engine.use_torch_compile=false \
  model.enable_gradient_checkpointing=false \
  engine.wrap_policy.min_num_params=100000000 \
  model.path="$MODEL_PATH" \
  model.use_remove_padding=false \
  +model.override_config.attn_implementation=eager \
  trainer.default_local_dir="$SAVE_PATH" \
  trainer.project_name=verl-xpu-smoke \
  trainer.experiment_name=qwen2_5_0_5b-fsdp-2xpu \
  trainer.logger='["console"]' \
  trainer.total_epochs=1 \
  "$@"
