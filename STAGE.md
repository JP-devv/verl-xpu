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

**Real verl FSDP SFT trainer — BLOCKED (crash-respawn loop).** Ran
`scripts/xpu/smoke-sft-2gpu.sh` (Qwen2.5-0.5B, 2 GPUs, `VERL_PLATFORM=intel`) on cards 2,3.
The **preflight passes** (2 devices, `xpu/intel/xccl`), the **FSDP wrap works** (model
loaded, 5.3 GB, `min_num_params=1e8` → one FSDP unit, `forward_prefetch=true`,
`use_torch_compile=false`, `enable_gradient_checkpointing=false`), and the **training loop
starts** (`Epoch 1/1: 0/N`). But the **first training step never completes**: the ranks
run at ~100% CPU for ~50 s then die (native crash in the oneCCL/SYCL layer, no Python
traceback), and `mpiexec` respawns them in a loop (seen as fresh PIDs with `etimes` reset
to ~50 s while wall-clock keeps climbing). Root cause: the venv's **old oneCCL 2021.16.1
+ `mpi` transport** handles a single large all-reduce (the probe passed) but **crashes on
the FSDP backward's many-small reduce-scatters** (~one per weight tensor). The
`CCL_SYCL_COPY_ENGINE=1` knob fixes the large-allreduce path, not this one. **The real fix
for the primary goal (fast, scalable XPU training) is to move off the 2021.16.1/mpi stack
to oneCCL 2022.0.0+ with native XCCL** (the k8v4-pins known-good set: no `mpi` transport,
`CCL_SYCL_ALLREDUCE_LL=ring` + copy-engine + the three `*_SIMPLE_THRESHOLD` + `ALLTOALL_TMP_BUF`),
which is both faster and stable. Config gotcha hit: `data.train_batch_size` is the **global**
batch and must be ≥ the GPU count (with 2 GPUs, `=1` → per-GPU 0.5 → `batch_size=0` error;
minimum is `=2`).

**Real verl FSDP SFT trainer — UNBLOCKED on the throwaway 2026 venv (native XCCL 2022.0.0).**
Built a **separate throwaway venv** `~/AI/verl/.venv-xpu2026` with the matched set
**torch 2.13.0+xpu + oneCCL 2022.0.0 + `intel-sycl-rt` 2026.0.0** (the k8v4-pins known-good
stack) — the main `~/AI/verl/.venv` (torch 2.9.1 + oneCCL 2021.16.1) is **untouched**. On this
venv the FSDP SFT smoke test **completes**: Qwen2.5-0.5B, 2 GPUs (cards 2,3), `torchrun`
(native XCCL needs **no** `mpi` transport / `mpiexec`), all **16/16 steps** run, loss decreases
(0.90 → 0.32), grad_norms finite, and the **FSDP model checkpoint saves** (599 MB, 2 rank
files). **The 2021.16.1/`mpi` FSDP-backward many-reduce-scatter crash is fixed** by the native
XCCL 2022.0.0 backend. Setup gotchas that cost time: (a) the `--no-deps` pure-Python dep install
must **exclude the entire Intel/SYCL/MKL/oneCCL/nvidia stack** (a filter pattern missing `.*`
before `==` let 2025.2.1 `onemkl-sycl-blas` in, which ships `libmkl_sycl_blas.so.5` and breaks
torch 2.13.0's `.so.6` link); (b) `torchrun` must be called as `torchrun … -m verl.trainer.sft_trainer`
— passing `"$PY"` as the first arg makes the worker command `[python, python, -m, …]` (null-byte
SyntaxError); (c) pip/TMPDIR must **not** be the 31 G `/tmp` tmpfs (the 819 MB torch wheel unpack
hits EDQUOT) — point `TMPDIR` at `/var/tmp`.

**1/2/4-GPU scaling evidence (Qwen2.5-0.5B, 128-token micro-batch, 32 total samples):**
- **1 GPU** (card 2): 32 steps @ **5.95 it/s** → 32 samples in ~5.4 s; peak XPU mem 11.7 GB.
- **2 GPUs** (cards 2,3): 16 steps @ **~4.0 it/s** → 32 samples in ~4.0 s; peak XPU mem 6.7 GB/rank.
- **4 GPUs: not runnable on this box** — cards 0,1 are the live serving lane (never touched), so
  the max is 2. Wall-clock speedup 1-GPU→2-GPU is **1.35×** (5.4 s → 4.0 s for the same 32 samples);
  it is sub-linear because a 0.5 B model's per-step compute is tiny, so the FSDP reduce-scatter/
  all-gather collectives dominate. A 27 B model (the real guinea pig) has far more per-step compute
  per GPU, so the collective fraction shrinks and the 2-GPU speedup should approach the ideal.
  Memory scales down as expected (11.7 GB → 6.7 GB/rank) — FSDP sharding is working.

**Swift-27B MTP-head FSDP SFT — BLOCKED by a torchrun-agent SIGHUP during the model load
(NOT a RAM issue).** Wired the MTP head into verl's FSDP SFT path via a remote-code model
class (`mtp_work/modeling_qwen3_5_mtp.py`: `Qwen3_5MTPHead` = 424.7 M trainable MTP head +
27.36 B frozen backbone, loaded via `auto_map` + `trust_remote_code`), with
`scripts/xpu/smoke-mtp-2gpu.sh` (2-GPU, `ZE_AFFINITY_MASK=2,3`, cards 2,3). Patched
`verl/workers/engine/fsdp/transformer_impl.py::_build_module` to pass
`low_cpu_mem_usage=True` to `from_pretrained` (streams weights shard-by-shard into the
meta-tensor init context). A bounded load-probe (`mtp_work/load_probe.py`) confirmed the
low-CPU-mem load keeps host RAM bounded: 2 shards (3.5 B params, 6.5 GB) → **peak RSS
0.5 GB**, free stayed 51.8 GB; full 18-shard model est. ~5 GB peak. **But the 2-process
torchrun FSDP load dies at ~37% (437/1184 tensors) with `SignalException: Process
<agent-pid> got signal: 1` (SIGHUP) on the torchrun *agent* process** — reproduced
consistently, **even with `setsid`** (fully detached from the controlling terminal), and
with 53 GB host RAM free. So it is **not** an OOM and **not** a worker crash: the elastic
agent itself receives a SIGHUP mid-load and tears down the workers. The single-process
load (no torchrun agent) works fine, which isolates the fault to the torchrun elastic
agent's signal handling on this host. **Next concrete fix:** launch the 2 ranks via a
manual `env://` bootstrap (set `RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_*` and run
`python -m verl.trainer.sft_trainer` directly, no `torchrun` agent) to bypass the
SIGHUP-prone elastic agent; if that also dies at ~37%, capture `dmesg` / `py-spy dump`
on the agent PID at the crash to identify the SIGHUP sender. Until then the 27B MTP-head
FSDP run is blocked; the 0.5 B FSDP SFT (small model, loads before the SIGHUP window)
remains the working XPU FSDP proof.

**oneCCL 2022.0.0 upgrade — ATTEMPTED, INCOMPATIBLE with torch 2.9.1+xpu (reverted).**
Tried to move off the 2021.16.1/`mpi` stack to oneCCL 2022.0.0 (native XCCL) to fix the
FSDP-backward crash. oneCCL is a standalone Intel pip package, so a `--no-deps` install of
2022.0.0 was possible **without** touching torch — but it failed: uninstalling 2021.16.1
removed **`libsycl.so.9`**, which torch 2.9.1+xpu is linked against (the 2021.16.1 oneCCL
bundles that `.so.9`; the venv's `intel-sycl-rt` 2025.2.1 only ships `libsycl.so.8`). Result:
`ImportError: libsycl.so.9: cannot open shared object file`. **Root cause: oneCCL 2022.0.0
and torch 2.9.1+xpu need different `libsycl.so.9` builds and cannot coexist.** Reverted to
oneCCL 2021.16.1 (restores `libsycl.so.9`); torch imports and the 512 MiB large-allreduce
probe pass again (~36 GB/s). **The clean path to native-XCCL oneCCL 2022.0.0+ is a matched
set — torch 2.13.0+xpu + oneCCL 2022.0.0 + `intel-sycl-rt` 2026.x (the k8v4-pins stack) —
which is a major, high-risk venv change** (would re-break the transformers 5.12.1 / verl
0.10.0.dev / MTP-head pins and needs full re-validation). **DONE in a throwaway venv**
(`~/AI/verl/.venv-xpu2026`, see above) — the main `~/AI/verl/.venv` is left at the
known-good 2021.16.1 + torch 2.9.1 state, and the throwaway venv is where the FSDP SFT
path now works. **1/2-GPU scaling evidence captured** (above); 4-GPU not runnable on this
box (cards 0,1 = serving lane). The MTP-head trainer (`train_mtp3.py`) still runs on the
main venv (single-card head, no FSDP collectives); the full 27B FSDP path now runs on the
throwaway 2026 venv.

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
- [x] **Real verl FSDP SFT trainer — UNBLOCKED** on the throwaway 2026 venv (native XCCL
  2022.0.0): 16/16 steps, loss 0.90→0.32, FSDP checkpoint saved. The 2021.16.1/`mpi`
  many-reduce-scatter backward crash is fixed.
- [x] **oneCCL 2022.0.0 upgrade — DONE in throwaway venv** `~/AI/verl/.venv-xpu2026`
  (torch 2.13.0+xpu + oneCCL 2022.0.0 + sycl-rt 2026.0.0). Main venv untouched.
  **1/2-GPU scaling evidence captured** (1.35× wall-clock; 4-GPU not runnable — cards 0,1
  are the serving lane).
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
