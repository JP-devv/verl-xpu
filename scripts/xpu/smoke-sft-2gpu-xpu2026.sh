#!/usr/bin/env bash
# FSDP SFT smoke test on the THROWAWAY venv (torch 2.13.0+xpu + oneCCL 2022.0.0
# native XCCL). Uses torchrun (NOT mpiexec) — native XCCL needs no MPI bootstrap.
# NO CCL_ATL_TRANSPORT=mpi. B70 CCL_SYCL block for the copy-engine path.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="/home/haxor/AI/verl"
VENV="$REPO/.venv-xpu2026"
PY="$VENV/bin/python"

export ZE_AFFINITY_MASK=2,3
unset ONEAPI_DEVICE_SELECTOR
export VERL_PLATFORM=intel
# native XCCL: NO CCL_ATL_TRANSPORT=mpi. B70 CCL_SYCL block:
export CCL_SYCL_ALLREDUCE_LL=ring
export CCL_SYCL_ALLREDUCE_SIMPLE_READ=1
export CCL_SYCL_COPY_ENGINE=1
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLTOALL_TMP_BUF=1

MODEL_PATH="/tmp/xpu-sanity/qwen2.5-0.5b"
DATA_DIR="/tmp/verl-xpu-smoke"
SAVE_PATH="/tmp/verl-xpu-smoke/ckpt-xpu2026"

# preflight
"$PY" - <<'PYEOF'
import torch
assert torch.xpu.is_available(), "torch.xpu not available"
n = torch.xpu.device_count()
print(f"[preflight] visible XPU devices: {n}")
assert n == 2, f"expected 2 visible XPU devices, got {n}"
from verl.plugin.platform import get_platform
p = get_platform()
print(f"[preflight] platform: {p.device_name}/{p.vendor_name}/{p.communication_backend_name()}")
assert (p.device_name, p.vendor_name, p.communication_backend_name()) == ("xpu", "intel", "xccl")
print("[preflight] OK")
PYEOF

# launch: 2-GPU FSDP SFT via torchrun (native XCCL)
exec "$VENV/bin/torchrun" --nproc_per_node=2 --master-port=29519 -m verl.trainer.sft_trainer \
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
  trainer.experiment_name=qwen2_5_0_5b-fsdp-2xpu-xpu2026 \
  trainer.logger='["console"]' \
  trainer.total_epochs=1
