#!/usr/bin/env bash
# Build the throwaway 2026 venv: torch 2.13.0+xpu + oneCCL 2022.0.0 + SYCL 2026.0.0.
# pip/TMPDIR must NOT be the 31G tmpfs (/tmp) — the 819MB torch wheel unpack hits EDQUOT.
export TMPDIR=/var/tmp/xpu2026-tmp
mkdir -p "$TMPDIR"
# Order matters: torch FIRST (it installs the matched 2026.0.0 Intel/SYCL/MKL stack),
# then pure-Python deps with --no-deps (the 2026 stack must not be replaced by 2025.2.1).
set -euo pipefail

REPO=/home/haxor/AI/verl
V="$REPO/.venv-xpu2026"
XPU_IDX=https://download.pytorch.org/whl/xpu

echo "=== [1/5] recreate venv ==="
rm -rf "$V"
python3.12 -m venv "$V"
"$V/bin/pip" install -q --upgrade pip

echo "=== [2/5] torch 2.13.0+xpu (brings the matched 2026.0.0 stack) ==="
"$V/bin/pip" install --resume-retries 30 torch==2.13.0+xpu --extra-index-url "$XPU_IDX"

echo "=== [3/5] verify torch before any other install ==="
"$V/bin/python" -c "import torch; print('torch', torch.__version__, 'xpu', torch.xpu.is_available(), 'devices', torch.xpu.device_count())"

echo "=== [4/5] pure-Python deps (--no-deps: do NOT let 2025.2.1 Intel stack in) ==="
# /tmp/reqs-xpu2026d.txt = main-venv freeze minus torch / oneCCL / SYCL / MKL / nvidia / triton
"$V/bin/pip" install --no-deps --resume-retries 30 -r /tmp/reqs-xpu2026d.txt

echo "=== [5/5] editable installs from the repo (verl + hardware plugin) ==="
"$V/bin/pip" install --no-deps --no-build-isolation -e "$REPO"
"$V/bin/pip" install --no-deps --no-build-isolation -e /home/haxor/AI/verl-hardware-plugin

echo "=== FINAL VERIFY ==="
"$V/bin/python" - <<'PY'
import torch
print('torch      ', torch.__version__)
print('xpu avail  ', torch.xpu.is_available(), 'devices', torch.xpu.device_count())
import transformers, accelerate, packaging, pyarrow
print('transformers', transformers.__version__)
print('accelerate  ', accelerate.__version__)
print('packaging   ', packaging.__version__)
print('pyarrow     ', pyarrow.__version__)
from verl.plugin.platform import get_platform
p = get_platform()
print('platform    ', p.device_name, p.vendor_name, p.communication_backend_name())
PY
for p in oneccl intel-sycl-rt mkl; do
  printf '%-14s ' "$p"; "$V/bin/pip" show "$p" 2>/dev/null | awk '/^Version:/{print $2}'
done
echo "VENV2026 SETUP DONE"
