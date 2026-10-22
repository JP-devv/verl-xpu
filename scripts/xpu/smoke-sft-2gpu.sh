#!/usr/bin/env bash
# Minimal isolated verl XPU FSDP SFT smoke test on cards 2,3 ONLY.
#
# Proves the official verl-hardware-plugin end-to-end on the free pair:
#   1. get_platform() -> xpu/intel/xccl (plugin discovered),
#   2. the FSDP XPU engine runs a real forward+backward+xccl-grad-sync on 2 cards,
#   3. cards 0,1 (the serving lane) are never touched.
#
# Uses the tiny Qwen2.5-0.5B + a synthetic GSM8K-style parquet (fast, memory-trivial).
# eager attention (the only XPU-validated attention path).
#
# VENV: defaults to the throwaway 2026 venv (.venv-xpu2026) whose oneCCL 2022.0.0
# native-XCCL backend handles the FSDP many-reduce-scatter backward (the old
# 2021.16.1/mpi stack in .venv crashes on it). Override with VENV=<path> to use
# another venv. Native XCCL is launched with torchrun (no MPI bootstrap needed).
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
VENV="${VENV:-$REPO/.venv-xpu2026}"
PY="$VENV/bin/python"

# --- GPU pinning: ONLY the two free cards (2,3) ------------------------------
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-2,3}"
unset ONEAPI_DEVICE_SELECTOR            # must NOT be set alongside ZE_AFFINITY_MASK
export VERL_PLATFORM=intel              # force the intel/xpu/xccl platform
#
# Native XCCL (oneCCL 2022.0.0+): NO CCL_ATL_TRANSPORT / I_MPI_FABRICS — the native
# backend handles the transport directly and is launched with torchrun. The B70 SYCL
# device-backend block routes large collectives over the DMA copy-engine.
export CCL_SYCL_COPY_ENGINE="${CCL_SYCL_COPY_ENGINE:-1}"
export CCL_SYCL_ALLREDUCE_LL="${CCL_SYCL_ALLREDUCE_LL:-ring}"
export CCL_SYCL_ALLREDUCE_SIMPLE_READ="${CCL_SYCL_ALLREDUCE_SIMPLE_READ:-1}"
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD="${CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD:-4294967296}"
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD="${CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD:-4294967296}"
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD="${CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD:-4294967296}"
export CCL_SYCL_ALLTOALL_TMP_BUF="${CCL_SYCL_ALLTOALL_TMP_BUF:-1}"

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

# --- launch: 2-GPU FSDP SFT via torchrun (native XCCL) -----------------------
# NOTE: `torchrun ... -m module` — do NOT pass "$PY" as the first arg (torchrun
# prepends sys.executable itself, which would make the worker [python, python, -m, ...]).
exec "$VENV/bin/torchrun" --nproc_per_node=2 --master-port="${MASTER_PORT:-29519}" \
  -m verl.trainer.sft_trainer \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.messages_key=messages \
  data.micro_batch_size_per_gpu=1 \
  data.train_batch_size=2 \
  data.max_length=128 \
  data.use_dynamic_bsz=false \
  optim.lr=1e-4 \
  engine=fsdp \
  engine.use_torch_compile=false \
  engine.forward_prefetch=true \
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
