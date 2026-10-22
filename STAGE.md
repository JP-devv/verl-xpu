# verl → Intel Arc XPU: Stage Setup

Goal: port **verl** to run on the 4× Intel Arc Pro B70 (BMG-G31) on haxor, and use
**two** of the cards to finetune / experiment on **Swift-Qwen-27B** (a `qwen3_5`
GDN-hybrid 27B with an un-finetuned MTP head). The other two cards keep serving
the agent model and must not be touched.

## GPU allocation (do not cross the line)

| xpu-smi id | PCI BDF | /dev/dri | Role |
|---|---|---|---|
| 0 | `0a:00.0` | card0 | **Serving** — Swift-Qwen-27B TP2 (vLLM, in container) |
| 1 | `0e:00.0` | card1 | **Serving** — Swift-Qwen-27B TP2 (vLLM, in container) |
| 2 | `14:00.0` | card2 | **FREE — ours** (finetune / XPU port) |
| 3 | `18:00.0` | card3 | **FREE — ours** (finetune / XPU port) |

Pin every run to the free pair with `ZE_AFFINITY_MASK=2,3` and leave
`ONEAPI_DEVICE_SELECTOR` **unset** (an empty value is invalid and conflicts with the
affinity mask). Never let a training/rollout process grab cards 0/1.

## Model

- Working copy: **`~/models/swift-27b`** (52 GB, 18 safetensors + config).
- It is a **full copy** of the pristine `~/models/Qwen3.8-27B` — the original is
  left untouched so finetuning checkpoints can't corrupt it.
- Config: `model_type=qwen3_5`, `Qwen3_5ForConditionalGeneration`, bf16,
  64 layers (16 full-attention + 48 linear/GDN, `full_attention_interval=4`),
  hidden 5120, vocab 248320, max_pos 262144, **`mtp_num_hidden_layers=1`**
  (MTP present in the architecture, not yet finetuned), multimodal (vision_config).

## Python environment

- **`~/AI/verl/.venv`** — Python 3.12, `torch==2.9.1+xpu` (from
  `https://download.pytorch.org/whl/xpu`), plus verl's core deps.
- **Self-contained oneAPI**: the venv carries its own `intel_sycl_rt-2025.2.1`,
  `mkl-2025.2.0`, `oneccl-2021.16.1`, `umf`, `intel_opencl_rt-2025.2.1` — the
  "known-clean torch 2.9.1 / sycl-rt 2025.2.1" combo.
- ⚠️ **Do NOT `source /opt/intel/oneapi/setvars.sh`** (the system oneAPI is
  **2025.3**). Its `libur`/`libsycl` mismatch torch 2.9.1 and breaks the import
  (`undefined symbol: urEnqueueCooperativeKernelLaunchExp, LIBUR_LOADER_0.12`).
  Run the venv python directly; it finds its own 2025.2.1 libs.
- Verified: `torch.xpu.is_available() == True`, `torch.xpu.device_count() == 4`.
- Excluded `liger-kernel` (CUDA-only kernels, useless on XPU).

## Platform port (the core artifact)

> **Course-corrected.** An early hand-rolled `verl/plugin/platform/platform_xpu.py`
> (registered in `platform_manager.py`) was **reverted** — the correct path is the
> official out-of-tree **`verl-hardware-plugin`** (`github.com/verl-project/verl-hardware-plugin`),
> which registers `intel`/`xpu`/`xccl` without core patching and carries the oneCCL
> AVG all-reduce fix, `ZE_AFFINITY_MASK` device-pin fix, NUMA affinity patch, and VTune/ITT
> profiler registration that a hand-rolled draft misses. Install it into the venv
> (`pip install -e .`) and verify `get_platform()` returns `xpu intel xccl`.

## MTP head finetune (DONE — see `mtp_work/MTP_FT.md`)

Reconstructed the `mtp.*` head the base model ignores, measured its acceptance, and
finetuned it — all on the free cards, serving lane untouched:

- **Baseline acceptance: 34.55%** (WikiText-103 TEST, 32,256 positions).
- **Finetuned (6 epochs): 35.56%** — a +1.0pp (~2.9% relative) improvement.
- Memory-safe two-phase training (full model on cards to harvest `h_i`, then the small
  MTP head trains alone on one card) — a naive 54GB in-RAM load OOMs the host and kills
  the serving lane, so every run is RAM-watchdog-guarded.
- Finetuned head: `~/models/swift-27b-mtp-ft/mtp_head_final.pt`.

## XPU smoke test (DONE — plugin verified end-to-end)

Installed `verl-hardware-plugin` into the venv and verified the full stack on cards 2,3
(serving lane 0,1 untouched):

- **`get_platform()` → `xpu / intel / xccl`** — the plugin is auto-discovered via the
  `verl.plugins` entry-point; `import verl` resolves to the local `~/AI/verl` (0.10.0.dev).
- **xccl collectives work**: 2-proc `all_reduce` over the `xccl` backend returns the
  correct sum (rank0=1 + rank1=2 → 3).
- **FSDP engine works**: a 2-GPU FSDP wrap + forward + backward + optimizer step
  completes (`scripts/xpu/` minimal test) — this exercises the plugin's
  `FSDPXPUEngineWithLMHead` + the oneCCL AVG→SUM monkeypatch.
- **Device pinning works**: `ZE_AFFINITY_MASK=2,3` + `ONEAPI_DEVICE_SELECTOR` unset keeps
  every process on the free pair.

**oneCCL transport gotcha (the real blocker for real-model runs):** this host has no
usable OFI/RDMA provider (`fi_getinfo` returns 0), so the default OFI transport fails to
init. The fix is the **MPI transport**: `CCL_ATL_TRANSPORT=mpi` + `ONEAPI_ROOT=$VENV`
(satisfies oneCCL's `check_impi_env`) + `I_MPI_FABRICS=shm`, launched via `mpiexec`
(oneCCL needs a real MPI bootstrap; torchrun doesn't provide one). The `shm` fabric
handles small collectives (the minimal FSDP test) but **deadlocks on real-model-sized
(~1GB) gradient all-reduces** — so a full SFT/GRPO run of a real model hangs in the
backward. That's a limitation of the venv's self-contained oneCCL/IMPI stack (built for
OFI/RDMA), not of the plugin. A host with RDMA (or a oneCCL with a working software
provider) would clear it.

## Status

- [x] Model staged (`~/models/swift-27b`), original intact
- [x] Venv with working torch-XPU (4 devices visible)
- [x] MTP head reconstructed + baseline measured (34.55%)
- [x] MTP head finetuned (35.56%) — `~/models/swift-27b-mtp-ft/`
- [x] Core-tree `PlatformXPU` draft reverted (use official `verl-hardware-plugin`)
- [x] `verl-hardware-plugin` installed + `get_platform()` → `xpu/intel/xccl`
- [x] XPU smoke test: xccl collectives + FSDP engine + device pinning verified on 2,3
- [ ] Real-model FSDP run (blocked by oneCCL `shm` fabric large-all-reduce deadlock)

## Next

1. To run a real model end-to-end, the oneCCL transport needs a working provider:
   either RDMA/OFI on the host, or a oneCCL build with a software (tcp/sockets) provider
   that handles large all-reduces. The plugin itself is ready.
2. Pick the training engine (FSDP) and wire a `ppo_trainer`/`sft_trainer` config for
   `device=xpu`, 2 GPUs, once the transport is sorted.
3. Smoke-test end-to-end on cards 2,3 before scaling.
