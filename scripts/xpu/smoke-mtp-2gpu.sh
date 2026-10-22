#!/usr/bin/env bash
# Swift-27B MTP-head FSDP SFT on cards 2,3 (throwaway 2026 venv, native XCCL).
# Trains ONLY the MTP head (424.7M params); the 27.36B backbone is frozen.
# Uses the remote-code model class (auto_map + trust_remote_code) so verl's
# AutoModelForImageTextToText.from_pretrained loads Qwen3_5MTPHead.
set -euo pipefail
REPO="/home/haxor/AI/verl"
VENV="$REPO/.venv-xpu2026"
PY="$VENV/bin/python"

export ZE_AFFINITY_MASK=2,3
unset ONEAPI_DEVICE_SELECTOR
export VERL_PLATFORM=intel
export CCL_SYCL_COPY_ENGINE=1
export CCL_SYCL_ALLREDUCE_LL=ring
export CCL_SYCL_ALLREDUCE_SIMPLE_READ=1
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLTOALL_TMP_BUF=1

MODEL_PATH="${MODEL_PATH:-/home/haxor/models/swift-27b-mtp}"
DATA_DIR="${DATA_DIR:-/tmp/verl-xpu-smoke}"
SAVE_PATH="${SAVE_PATH:-/tmp/verl-xpu-smoke/ckpt-mtp}"

# preflight
"$PY" - <<'EOF'
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
EOF

# launch: 2-GPU FSDP SFT of the MTP head via torchrun (native XCCL)
exec "$VENV/bin/torchrun" --nproc_per_node=2 --master-port="${MASTER_PORT:-29523}" \
  -m verl.trainer.sft_trainer \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.messages_key=messages \
  data.micro_batch_size_per_gpu=1 \
  data.train_batch_size=2 \
  data.max_length=256 \
  data.use_dynamic_bsz=false \
  optim.lr=1e-4 \
  engine=fsdp \
  engine.use_torch_compile=false \
  engine.forward_prefetch=true \
  model.enable_gradient_checkpointing=false \
  engine.wrap_policy.min_num_params=100000000 \
  model.path="$MODEL_PATH" \
  model.trust_remote_code=true \
  model.use_remove_padding=false \
  +model.override_config.attn_implementation=eager \
  trainer.default_local_dir="$SAVE_PATH" \
  trainer.project_name=verl-xpu-mtp \
  trainer.experiment_name=swift27b-mtp-head-fsdp-2xpu \
  trainer.logger='["console"]' \
  trainer.total_epochs=1
