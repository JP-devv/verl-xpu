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

## MTP head finetune (diverse eval — see `mtp_work/MTP_FT.md`)

Reconstructed the `mtp.*` head the base model ignores, then finetuned it on a
**domain-matched** corpus (json / python / tool_call / prose) and evaluated on a
**disjoint held-out test set** (no sequence shared with train/val) — the unbiased
number, all on the free cards, serving lane untouched:

| Task | Baseline → Finetuned | Delta | Verdict |
|------|----------------------|------:|---------|
| json | 41.3% → 58.5% | **+17.2pp** | significant |
| tool_call | 57.3% → 83.9% | **+26.7pp** | significant |
| python | 42.2% → 39.6% | **−2.6pp** | **regressed** |
| prose | 34.3% → 34.5% | +0.2pp | noise |
| **OVERALL** | 41.7% → 49.3% | **+7.6pp** | significant |

The distribution-matched finetune is a real, significant **+7.6pp overall** with large
gains on the structured domains, **but it is not uniform** — python regressed and prose
is flat, i.e. the head overfit the train mix. Next iteration: broader/balanced corpus
(Nemotron text for prose, more diverse python) + stronger regularization, re-measured on
the disjoint test set. Memory-safe two-phase training (RAM-watchdog-guarded); finetuned
head at `~/models/swift-27b-mtp-ft/mtp_head_final.pt`.

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

**oneCCL transport — SOLVED (large all-reduce now works).** This host has no usable
OFI/RDMA provider (`fi_getinfo` returns 0), so the default OFI transport fails to init.
The venv's oneCCL (2021.16.1) `CCL_ATL_TRANSPORT` accepts **only `ofi` | `mpi`** — there
is no `sycl` transport and **no `shm` transport** (`shm` is a *fabric* inside the `mpi`
transport, set via `I_MPI_FABRICS=shm`). The `CCL_SYCL_*` strings tune the SYCL *device
backend* (GPU-side kernels) that runs within a transport. The old `mpi`+`shm` path
deadlocked on ~1GB all-reduces because it moved data through host memory on the compute
path. **The working recipe (validated):**
```
CCL_ATL_TRANSPORT=mpi  I_MPI_FABRICS=shm  ONEAPI_ROOT=$VENV   # real MPI transport + shm fabric
CCL_SYCL_COPY_ENGINE=1                                # large moves over the DMA copy-engine
CCL_SYCL_ALLREDUCE_LL=ring  CCL_SYCL_ALLREDUCE_SIMPLE_READ=1
CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296  CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296  CCL_SYCL_ALLTOALL_TMP_BUF=1
```
launched via **`mpiexec -n 2`** (the oneCCL mpi transport needs a real MPI bootstrap;
torchrun yields "Duplicate ranks"), with `ZE_AFFINITY_MASK=2,3` and `ONEAPI_DEVICE_SELECTOR`
unset. **Result: 512 MiB bf16 all-reduce in 0.03 s (~36 GB/s), correct sum — PASS**
(`scripts/xpu/test_large_allreduce.py` + `scripts/xpu/launch-large-ar.sh` +
`scripts/xpu/mpi-ar-wrapper.sh`). This unblocks real-model FSDP on cards 2,3. A host with
RDMA (or a newer oneCCL with a software provider) would be cleaner, but this is enough.

## Status

- [x] Model staged (`~/models/swift-27b`), original intact
- [x] Venv with working torch-XPU (4 devices visible)
- [x] MTP head reconstructed + diverse finetune: **+7.6pp overall** on disjoint test
  (json +17.2, tool_call +26.7 significant; python −2.6 regressed; prose noise)
- [x] Core-tree `PlatformXPU` draft reverted (use official `verl-hardware-plugin`)
- [x] `verl-hardware-plugin` installed + `get_platform()` → `xpu/intel/xccl`
- [x] XPU smoke test: xccl collectives + FSDP engine + device pinning verified on 2,3
- [x] **Large all-reduce UNBLOCKED**: `CCL_ATL_TRANSPORT=mpi`+`I_MPI_FABRICS=shm`+
  `CCL_SYCL_COPY_ENGINE=1`+B70 block via `mpiexec` → 512 MiB in 0.03 s (~36 GB/s), PASS
- [x] **XPU-optimized training path**: FSDP-sharded (device_map) all-on-XPU MTP
  trainer — 27B sharded across cards 2,3 (no card holds the full 54GB), MTP head on
  the last card sharing the main embed (no 2.4GB local copy → no OOM). **Verified
  running with per-step logging:** epoch 0 in 199s, val 0.4207 → 0.4778 (+5.7pp),
  per-step loss + batch-acc every 50 steps, no CPU h_i round-trip.

## Next (priority: build the XPU-optimized training path — the greater value)

1. **All-on-XPU single-phase MTP trainer** (`mtp_work/train_mtp3.py`): keep the frozen
   27B + the MTP head on the cards and compute `h_i` on the fly — no CPU round-trip.
   This is the shape verl's FSDP engine uses and removes the phase-2 CPU bottleneck.
2. **Land the large-allreduce fix:** `CCL_ATL_TRANSPORT=shm` + `CCL_SYCL_COPY_ENGINE=1`
   (+ B70 block). Validate with `scripts/xpu/test_large_allreduce.py`, then
   `scripts/xpu/smoke-sft-2gpu.sh` for a real-model FSDP run on cards 2,3.
3. **Iterate the MTP finetune** to fix the python regression + prose: broader/balanced
   corpus (Nemotron text) + stronger regularization, re-measured on the disjoint test set.
4. Wire a `sft_trainer`/`ppo_trainer` config for `device=xpu`, 2 GPUs, once the
   transport + all-on-XPU path are in.
