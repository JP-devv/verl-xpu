---
name: xpu
description: "Intel XPU software stack for the 4x Arc Pro B70 (Battlemage) on haxor: verl + verl-hardware-plugin RL training (VERL_PLATFORM=intel, xccl/oneCCL, plugin registration), the SYCL/DPC++ toolchain (icpx, --gpu-arch targets, Level Zero visibility debugging, compute-runtime env vars), performance kernel libraries (sycl-tla, vllm-xpu-kernels), fine-tuning paths (ipex-llm, unsloth), and community escape hatches. Invoke /xpu for verl XPU setup, SYCL compilation, Level Zero/driver debugging, XPU kernel optimization, or Intel Arc fine-tuning. Hardware/driver/wedge triage: /intel-b70; box inventory: /haxor."
---

# xpu — Intel XPU software stack (verl · SYCL · Level Zero · kernels)

Scope: the **software** side of the 4x Arc Pro B70 (Battlemage/BMG, `8086:e223`) on haxor — verl RL training, the SYCL/DPC++ + Level Zero toolchain, optimized kernel libraries, fine-tuning paths, and community troubleshooting. Hardware/driver/wedge triage (power caps, PCIe, hang diagnosis, monitoring) lives in `/intel-b70`; box inventory in `/haxor`.

Full knowledge base: `/home/haxor/xpu-kb/` (`01-verl-xpu.md` … `09-k8v4-pins.md`, each with source repos + commit SHAs).

## 1. Layer 1 — the verl contract

**What it is:** `verl-hardware-plugin` is the out-of-tree platform + training-engine plugin for verl (RL post-training), jointly developed by the ByteDance verl team and the FlagOS community. The Intel XPU backend registers as platform `intel`, device `xpu`, vendor `intel`, comm backend `xccl` (oneCCL). Status: "✅ Example (requires vendor support)" — a reference implementation, not Intel-maintained. (Note: there is no monolithic `flagos-ai/FlagOS` repo — the org is a 61-repo family; `flagos-ai/docs` is the doc home.)

**Registration (no core patching):** verl auto-discovers the setuptools entry-points group `verl.plugins` (`verl/__init__.py:55-80`), gated by `VERL_USE_EXTERNAL_PLUGINS` (`auto` default / `none` / explicit list). The plugin's pyproject declares `[project.entry-points."verl.plugins"] hardware = "verl_hardware_plugin"`; importing it fires `@PlatformRegistry.register(platform="intel")` then `@EngineRegistry.register(device="xpu", vendor="intel")` (platforms first — upstream engines cache `get_platform()` at import). Engine lookup is 2-level `(device, vendor)` in `verl/workers/engine/base.py`.

**Install:**
```bash
git clone https://github.com/verl-project/verl.git && cd verl && pip install -e .
git clone https://github.com/verl-project/verl-hardware-plugin.git && cd verl-hardware-plugin && pip install -e .
```
Requires `verl>=0.9.0` (tested pin v0.9.1 @ `1876b06`); PyTorch with XPU support; **vLLM with XPU kernels** (`pip install "vllm==<ver>+xpu" --extra-index-url https://wheels.vllm.ai/<ver>/xpu` — the default PyPI wheel does NOT ship XPU kernels); oneAPI env (`source /opt/intel/oneapi/setvars.sh`).

⚠️ **Self-contained venv:** if the venv bundles its own oneAPI (the `+xpu` torch wheel ships `intel_sycl_rt-2025.2.1`), do **NOT** `source` the system `setvars.sh` — a version mismatch (system 2025.3 vs venv 2025.2.1) breaks the torch import (`undefined symbol: urEnqueueCooperativeKernelLaunchExp, LIBUR_LOADER_0.12`). Run the venv python directly.
⚠️ **`pip install -e .` side effect:** installing the plugin pulls in PyPI `verl` (can clobber a local editable `verl` checkout) and can downgrade `transformers`. Re-`pip install -e .` your local verl and restore the `transformers` pin afterward, then re-verify `get_platform()`.

**Verification:**
```bash
python3 -c 'from verl.plugin.platform import get_platform; p=get_platform(); print(p.device_name, p.vendor_name, p.communication_backend_name())'
# expected: xpu intel xccl — if it falls back to nvidia, the plugin was not discovered
```

**Load-bearing env vars:**
| Var | Effect |
|---|---|
| `VERL_PLATFORM=intel` | force platform (skip auto-detect) |
| `ZE_AFFINITY_MASK=0,1` | device visibility (the XPU `CUDA_VISIBLE_DEVICES`) |
| `ONEAPI_DEVICE_SELECTOR` | **must stay unset** — conflicts with `ZE_AFFINITY_MASK` |
| `RAY_EXPERIMENTAL_NOSET_ZE_AFFINITY_MASK` | emitted by the plugin; Ray must not fight verl's manual device assignment |
| `CCL_ATL_TRANSPORT=mpi` | oneCCL transport — the default `ofi` fails without RDMA; see the transport note below |
| `ONEAPI_ROOT=<venv>` | points oneCCL's MPI check at the venv's bundled Intel MPI |
| `I_MPI_FABRICS=shm` | IMPI fabric — `shm` for single-node (no RDMA); see the transport note |

**The 4 monkeypatches** (applied in `PlatformXPU.__init__`, so they only install where XPU is in play):
1. **`reduce_avg_allreduce_patch`** — oneCCL's `ReduceOp.AVG` **aborts on its scheduler path** ("average operation is not supported for the scheduler path"); the FSDP engine forces SUM+divide on every FSDP2 wrapper, plus a process-wide `all_reduce` AVG→SUM patch for xccl groups. Fix tracked for oneCCL 2022.2 / torch 2.15. **AVG on xccl groups is the minefield — user code is unprotected.**
2. **`ray_device_pin_patch`** — torch-xpu drops `ZE_AFFINITY_MASK` after lazy init, so without this every rank lands on `xpu:0` (silent OOM / collectives sharing a device).
3. **`numa_affinity_patch`** — pyzes-based NUMA pinning (needs pyzes ≥0.1.2; replaces the pynvml path).
4. **`register_vtune`** — VTune/ITT profiler markers; needs the `profiler_markers()` hook that landed in verl main 2026-10-08 (silently no-ops on older verl).

**Capabilities:** FSDP/FSDP2 validated (GRPO/PPO/SFT on GSM8K/Qwen2.5), vLLM rollout validated, **colocate mode only**. Megatron = WIP (bare subclass), SGLang = WIP, multi-node = WIP. Validated card: **B60 (24 GB)** with published numbers (1-GPU ~148 tok/s, Qwen2.5-0.5B); **B70 is a listed target with no in-repo validation — 4x B70 is extrapolation for haxor**.

**oneCCL transport reality (the part the docs skip):** the `xccl` backend's default transport is **OFI/RDMA**, which **fails to init on a host with no RDMA fabric** (`fi_getinfo` returns 0 providers → `failed to initialize ATL`). The fix is the **MPI transport**:
- `CCL_ATL_TRANSPORT=mpi` + `ONEAPI_ROOT=<venv>` (satisfies oneCCL's `check_impi_env`, which requires `I_MPI_ROOT` or `ONEAPI_ROOT`) + `I_MPI_FABRICS=shm` (single-node; `tcp` silently falls back to OFI and crashes).
- **Launch with `mpiexec`, not `torchrun`** — oneCCL's MPI transport needs a real MPI bootstrap. IMPI sets `PMI_RANK`/`PMI_SIZE` (not `OMPI_COMM_WORLD_*`), so map them to torch.distributed's `RANK`/`WORLD_SIZE`/`LOCAL_RANK` in a wrapper before `init_process_group("xccl")`.
- **`shm` fabric ceiling:** handles small collectives (a few-MB all-reduce) but **deadlocks on real-model-sized (~1GB) gradient all-reduces** — a full FSDP run of a real model hangs in the backward. A host with RDMA (or a oneCCL with a working software provider) clears this. Verified on haxor: 2-proc `all_reduce` + a small FSDP model (wrap + fwd/bwd/step) work; a Qwen2.5-0.5B FSDP run does not.

**FSDP-on-XPU gotchas:**
- The FSDP wrap moves each param to XPU one-at-a-time (`_move_states_to_device`), which is slow; with one FSDP unit **per layer** (default `min_num_params: 0`) it deadlocks on the `shm` fabric. Set `engine.wrap_policy.min_num_params=<high>` (e.g. `1e8`) to make the whole model **one** FSDP unit.
- `use_torch_compile` and `enable_gradient_checkpointing` both **hang in the backward** on XPU — set both `false` for a working (if slower) run.

KB detail: `/home/haxor/xpu-kb/01-verl-xpu.md` (full registration mechanism, exact 1-GPU GRPO launch command, all patch sources, 10 gotchas).

## 2. Layer 2 — the SYCL / Level Zero toolchain

### Compiling SYCL for Arc (`icpx` / `clang++ -fsycl`)
- **B70 = `bmg_g31`** (IP 20.2.0, Arc Pro B60/B70); B580 = `bmg_g21` (20.1.4); **`bmg` = the family alias** (safest — covers all BMG parts). Verified in `clang/lib/Driver/ToolChains/SYCL.cpp:1175` (`resolveGenDevice`).
- **`dgm` is gone** (2023-era Alchemist-datacenter code, no longer an arch); **`cmc` is NOT an arch code** (it's the compute-matrix-compiler tool).
- `icpx --sycl --gpu-arch=bmg kernel.cpp -o kernel` — icpx is a oneAPI-toolkit wrapper (not in intel/llvm); it expands to the in-repo-verified clang form: `clang++ -fsycl -fsycl-targets=intel_gpu_bmg_g31 kernel.cpp -o kernel`.
- Multi-arch + JIT fallback: `-fsycl-targets=intel_gpu_bmg,spir64` (`spir64` = runtime JIT; `intel_gpu_*` = AOT via `ocloc`; `--ocloc-path=` overrides the compiler). `spir64_gen` is deprecated.
- Per-arch preprocessor macros: `INTEL_GPU_BMG`, `INTEL_GPU_BMG_G21`, `INTEL_GPU_BMG_G31`.
- Toolchain versions: `sycl` branch = SYCL 9.0.0 dev; latest release 7.0.0 (7.1 upcoming; `sycl-rel-7_2` branch exists). oneAPI 2025.3+ is the CI-validated floor for the kernel libs.

### Device selection (runtime)
- **`ONEAPI_DEVICE_SELECTOR`** (SYCL-runtime var — NOT read by the NEO driver): `level_zero:*` = all GPUs; `level_zero:0` = first card; `"level_zero:*;!level_zero:0"` = all but card 0 (discard filters come last and win); **`level_zero:0.*` = all sub-devices (tiles) of card 0 as root devices (B70 = 2 tiles)**; `level_zero:0.0` = tile 0 only. Full BNF in `sycl/doc/EnvironmentVariables.md`; `sycl-ls` shows indices.
- `SYCL_DEVICE_ALLOWLIST` (BackendName/DeviceType/VendorId/Name-regex); `SYCL_LAUNCH_BLOCKING=1` for synchronous submits (debugging); `__SYCL_DEVICES` is a compile-time icpx-wrapper feature, not a runtime var.

### Multi-tile B70
- A B70 is a **2-tile card**. `sycl/doc/MultiTileCardWithLevelZero.md` documents four programming models: per-tile contexts, multi-sub-device context, implicit scaling on the root device, multi-card context. `ZE_FLAT_DEVICE_HIERARCHY=FLAT|COMPOSITE|COMBINED` controls how tiles/CCS map to root devices; `malloc_device` on a root device is visible to all its tiles.

### Level Zero loader (`libze_loader.so`)
- Flow: SYCL/UR → L0 loader → Intel ICD (`libze_intel_gpu.so.1`) → `xe` kernel driver. ICD discovery is **hardcoded to Intel ICD names**; `ZE_ENABLE_ALT_DRIVERS` for custom ICDs (dev/debug only).
- Logging: `ZEL_ENABLE_LOADER_LOGGING=1` + `ZEL_LOADER_LOG_CONSOLE=1` + `ZEL_LOADER_LOGGING_LEVEL=trace` → `$HOME/.oneapi_logs/ze_loader.log` (`ZE_ENABLE_LOADER_DEBUG_TRACE` is deprecated). Layers: `ZE_ENABLE_VALIDATION_LAYER=1`, `ZE_ENABLE_TRACING_LAYER=1`; `ZE_ENABLE_LOADER_INTERCEPT=1` forces handle-translation intercept even with one driver.
- API is at **v1.19** (spec 1.19.12), loader v1.34.0. The normative spec lives in the separate `oneapi-src/level-zero-spec` repo (this repo = headers + loader; headers are mako-generated, don't hand-edit).
- P2P: `zeDeviceCanAccessPeer` + `ze_device_p2p_properties_t` (`ACCESS` | `ATOMICS` flags). Sub-devices: `zeDeviceGetSubDevices`/`zeDeviceGetRootDevice`; subdevices get their own UUIDs.

### NEO driver (compute-runtime) env vars
- **~400 `NEO_*` debug vars are DEAD in the shipped release driver** (debug builds only). Production knobs = 14 release vars + 8 raw vars. Blog-post `NEO_EnableX` knobs may silently do nothing.
- Prefix scan: `NEO_L0_<Name>` > `NEO_<Name>` > `<Name>` (first hit wins). L0-spec vars (`ZE_AFFINITY_MASK`, `ZE_FLAT_DEVICE_HIERARCHY`, `ZES_ENABLE_SYSMAN`, `ZEX_NUMBER_OF_CCS`, `ZET_*`) are **bare-name only** — no prefixes.
- `ONEAPI_DEVICE_SELECTOR`/`__SYCL_DEVICES` are not read by NEO (loader/SYCL vars); **`ZE_ENABLE_L0` does not exist**.
- BMG: L0 1.18 / OCL 3.0; **sysman should use `zesInit`-based init** — legacy `ZES_ENABLE_SYSMAN` mode is only supported up to PVC (`programmers-guide/SYSMAN.md`).
- Compiler cache: `NEO_CACHE_DIR` / `NEO_CACHE_MAX_SIZE` (default 1 GiB) / `NEO_CACHE_PERSISTENT` / `NEO_CACHE_STATS`.
- Packaging: `intel-opencl-icd` (apt) + `intel-level-zero-gpu`; `.deb` on GitHub releases; monthly cadence (`releases/yy.ww`). Apps must link the L0 loader, never the runtime directly.

KB detail: `/home/haxor/xpu-kb/02-sycl-compiler.md` (full arch table, icpx flag mapping, extension headers), `03-level-zero.md` (API surface, loader env-var table, fabric model), `04-compute-runtime.md` (exhaustive NEO var tables).

## 3. Layer 3 — the performance path

**Rule of thumb:** `vllm-xpu-kernels` = the tuned, vLLM-wired product; `sycl-tla` = the template library you compose from. On B70, start with the vLLM wheel; drop to sycl-tla only when you need a kernel the product doesn't ship.

### vllm-xpu-kernels (inference kernels)
- vLLM XPU dispatches here automatically (PyTorch custom-ops; `import vllm_xpu_kernels._C` registers them all).
- Coverage: FMHA chunked-prefill + paged split-K decode, MoE grouped GEMM (INT4/MXFP4/FP8-tensor/FP8-block × BF16/FP16), oneDNN quantized GEMM (W8A8/W8A16/W4A4/W4A16/W4A8), sampling, RoPE (incl. DeepSeek fused paths), activation quant, LoRA shrink/expand.
- **B70 = `bmg-g31-a0` is in the default AOT device list** — a stock PyPI wheel already contains B70-targeted kernels. Install: `pip install vllm-xpu-kernels` (dev: `pip install --no-build-isolation -e .`).
- Build pins: torch `2.14.0+xpu`, oneAPI 2026.x, CMake ≥3.26, Ninja. Trim a B70-only build: `VLLM_XPU_XE2_AOT_DEVICES=bmg-g31-a0`.
- **Attention kernel matrix is compile-time**: `VLLM_CHUNK_PREFILL_CONFIG` / `VLLM_PAGED_DECODE_CONFIG` (default preset ≈ Llama/Qwen/DeepSeek coverage, ~2 min; full = 240/384 variants, ~60 min). An uncompiled (headsize, paged, causal, local, sink, lse) combo fails at runtime with "kernel tuple not compiled" → rebuild with a bigger config (`KERNEL_CONFIGURATION.md`).

### sycl-tla (custom GEMM / attention)
- Header-only fork of CUTLASS 4.2.1 (v0.9.3, BSD-3, active ~monthly). Consumers just add its `include/` to their include path — no library build.
- **B70 target: `intel_gpu_bmg_g31`** (also `intel_gpu_bmg_g21` = B580, `intel_gpu_pvc`, `intel_gpu_cri`). Build: `CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31` (oneAPI ≥2025.1; CI uses 2025.3+).
- API is the 4-layer CUTLASS 3.x stack: `CollectiveMma` (policy `MainloopXeL1Staged`) → `CollectiveEpilogue` (`IntelXeGeneric`, EVT fusion ops) → `GemmUniversal` → `GemmUniversalAdapter` (`can_implement` / `get_workspace_size` / `initialize` / `run`).
- Xe building blocks: `XE_DPAS_TT` MMA atoms (8×16×16 bf16→f32), 2D block copy (`XE_LOAD_2D`/VNNI/transpose) with 64B-aligned surfaces, explicit 2D prefetch, vISA inline asm (`sbarrier`, `lsc_fence`).
- Examples `00`–`13` cover bf16/FP8/grouped-MoE/stream-K GEMM + FMHA (prefill, cached-KV, paged-KV, varlen, LSE) — marked "Not for Benchmarking"; no SYCL profiler exists, so benchmark via examples/tests.
- **No conv** (inherited CUDA-only headers). vllm-xpu-kernels pins an older sycl-tla commit (`87f68506`), so the wheel's attention code lags upstream.

KB detail: `/home/haxor/xpu-kb/05-kernel-libs.md` (kernel inventory table, implementation techniques, build flags).

### GRIMOIRE (native B70 engine — kernel techniques + measured ceilings)
- From-scratch C++/SYCL/L0 inference engine built **exclusively for Arc Pro B70** (no vLLM/PyTorch in the compute path) + a 6-week dev log with 22 hard-won rules (`CLAUDE.md`). GPL-3.0. Paused at v1.8.3 (2026-10-06).
- **Measured B70 ceilings:** Ornith-1.5-35B-A3B: 193–199 tok/s decode, 9.5–10.2k prefill (1 card); c8 batched 464 tok/s (beats vLLM's 332); Qwen3.8-27B GPTQ-Int4+MTP: 54–72 tok/s.
- **PP beats TP on B70s** (TP all-gathers every projection through host memory); `multi` / `nrun.sh` launchers generalize to 4 cards (only 2-card measured in-repo).
- **Battlemage kernel techniques:** `joint_matrix` GEMM at 256 GRF/thread (110 TFLOP/s bf16), ESIMD flash attention, row-parallel DeltaNet, small-M DPAS MoE, **`-fno-fast-math`** (fast-math corrupts FA masking), AOT `-fsycl-targets=intel_gpu_bmg_g31` (g31/g21 images are NOT interchangeable).
- **Driver failure forensics:** killing a container with GPU work in flight drops the card off the bus (forcewake `0xFFFFFFFF` = power cycle only) → `--init --stop-timeout 300`; legacy L0 adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=0`) costs ~1.2 ms/token — never set it; resolve render nodes **by PCI address, never `renderD` number** (renumbered every reboot).
- **Pinned working driver-stack Dockerfile** (compute-runtime 26.27.39122.11 / IGC 2.38.2 / L0 1.32.0 .debs) — a template for haxor's own containers.
- Reading order: `CLAUDE.md` top → newest-dated handoff → README (it's a dev log — heavy superseded content).

KB detail: `/home/haxor/xpu-kb/08-grimoire.md` (full content inventory, version pins, open questions).

## 4. Layer 4 — the community escape hatch

Two repos with an explicit division of labor: **Unofficial-Intel-XPU-Community** = the stable hub (wiki guides + diagnostics; last commit 2026-05-23); **b70-optimization-lab** = the fast research feed + website (actively maintained; 8.5 GB / 187k files — never clone-read it wholesale: use `llms.txt`, `CURRENT.md`, `guides.html`, and named `community/`/`repro/`/`notes/` paths).

**The flagship working recipe (our exact topology — 4x B70):** MiniMax M2.7 INT4 AutoRound, vLLM, 32k context, ~84.1 output tok/s — lab `repro/minimax-m27-b70-110tps-ubuntu24-20260523/`. Two BMG-specific findings inside it:
- `xe.disable_display=1` + ASPEED display offload reclaims B70 VRAM (served context 24k→32k)
- PCIe4 x16 allreduce = 13.79 GB/s vs PCIe5-class 27.88 GB/s (~2x) — **check PCIe generation before blaming the model**

**Symptom → guide map:**
| Symptom | Go to |
|---|---|
| L0 device not visible / framework can't see GPU | community `docs/wiki/Diagnostics.md` (layered lspci → xpu-smi → ze_info → clinfo → torch.xpu) |
| New machine, where to start | `docs/wiki/Getting-Started.md` + `scripts/collect_xpu_snapshot.sh` |
| Version mismatch — which layer is broken | `docs/wiki/Drivers-and-Runtimes.md` |
| vLLM slow on multi-card B70 | lab `repro/minimax-m27-...` + `notes/2026-05-23-current-host-pcie4-prefill-check.md` |
| vLLM OOM / context won't fit | `xe.disable_display=1` + ASPEED offload (community README working recipe) |
| 2/4-GPU scaling doesn't add up | `docs/wiki/Two-B70-Use-Cases.md` (independent services first; no automatic memory pooling) |
| Docker/container XPU pass-through | `docs/wiki/Docker-and-Containers.md` |
| xe GUC hard lockup / GPU wedge | lab `data/2026-10-03-xe-guc-hard-lockup/` (cross-ref `/intel-b70` triage ladder) |
| llama.cpp SYCL / vLLM FP8 TP2 on B70 | lab `community/dominick253-*`, `community/sergiiob-*` |
| What's current / what changed | lab `CURRENT.md` + `CURRENT-history-*.md` |

**Gotchas:** freshness asymmetry (community ~5 months stale; lab current); the lab's "own-runtime" line is explicitly experimental (same-generation only, 128 GiB host RAM); single-machine docker recipes may not port — verify host requirements.

KB detail: `/home/haxor/xpu-kb/07-community.md` (full guide + recipe inventories).

## 5. Version pins (k8v4-xpu = the authority)

`Wrapzii/k8v4-xpu` (@ `04176ae0`, 2026-10-21) is the designated pinning authority: a vLLM 0.30 XPU serving stack (K8/V4 = int8-K/int4-V KV cache + hand-written SYCL decode/prefill libs), measured on 2× B60 and 2× B70 — the B70 community run is **on haxor**.

**The known-good set:**
| Layer | Pin |
|---|---|
| Base image | `vllm/vllm-openai-xpu:v0.30.0@sha256:fc0e112a…` (digest-pinned) |
| In the image | vLLM 0.30.0 · Python 3.12 · torch `2.13.0+xpu` (native XCCL) · oneCCL `2022.0.0` bundled · SYCL runtime `libsycl.so.9` · oneDNN 3.x (torch-bundled) |
| Compile toolchain | oneAPI `2026.0.0` / icpx `2026.0` — **icpx 2025.3 segfaults in dlopen against `libsycl.so.9`** (the sharpest constraint in the repo) |
| Engine config | `--kv-cache-dtype=int8_k_int4_v` · MTP 6 · `FULL_DECODE_ONLY` graphs `[7,14,21,28]` · `--max-num-batched-tokens=4224` · util 0.95 (0.90 on B70) |
| Env | `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE` · `ZE_AFFINITY_MASK=…` · CCL block: `CCL_SYCL_ALLREDUCE_LL=ring` on B70 (`twoshots` on B60), `CCL_SYCL_ALLREDUCE_SIMPLE_READ=1`, `CCL_SYCL_COPY_ENGINE=1`, three `*_SIMPLE_THRESHOLD=4294967296`, `CCL_SYCL_ALLTOALL_TMP_BUF=1` |
| Host | xe driver + L0; documented host Fedora 44 / kernel `7.2.9-201` / `linux-firmware 20260916` (BMG GuC v70.72.1); on Fedora 44 the container needs `--security-opt seccomp=unconfined` + `--cap-add SYS_PTRACE` (Level Zero IPC handles) |
| GPU tuning | `xpu-smi` 400–2400 MHz, 180 W burst |

**Measured (B70):** 120.5 tok/s single-stream / 328.3 tok/s 4-stream aggregate (community recipe); 1,255 tok/s fresh prefill at 128K with W4A8. B70 power ceiling ~250 W vs the 275 W firmware cap.

**Gotchas:** the vLLM patch is string-anchored to 0.30 internals (a vLLM upgrade breaks it until re-anchored); no driver package pins (assumed present on host); the B70 community recipe is "candidate, not qualified" (its own gates fail: attention rel-L2 1.336e-3 vs the 1e-3 screen; prefill aborts `DEVICE_LOST`); do NOT use `CCL_SYCL_ALLREDUCE_ARC=1`, simple threshold 0/8192, `CCL_ALLREDUCE=direct`, `CCL_ATL_TRANSPORT=mpi`.

KB detail: `/home/haxor/xpu-kb/09-k8v4-pins.md` (full pin tables incl. artifact hashes, complete env-var list, setup commands).

## Repo map

| Repo | Role in the stack | KB |
|---|---|---|
| `verl-project/verl` | RL framework core (plugin discovery, registries) | 01 |
| `verl-project/verl-hardware-plugin` | Intel XPU platform + FSDP engine (reference impl) | 01 |
| `flagos-ai` (org, 61 repos) | heterogeneous-chip community stack (no monolithic repo) | 01 |
| `intel/llvm` (sycl branch) | DPC++/icpx compiler + libsycl runtime | 02 |
| `oneapi-src/level-zero` | L0 headers + loader (spec lives in `level-zero-spec`) | 03 |
| `intel/compute-runtime` | NEO: the L0/OpenCL driver (userspace half over `xe`) | 04 |
| `intel/sycl-tla` | CUTLASS 4.2.1 SYCL fork (GEMM/FMHA templates) | 05 |
| `vllm-project/vllm-xpu-kernels` | vLLM XPU kernel layer (oneDNN + sycl-tla) | 05 |
| `intel/ipex-llm` | IPEX-LLM: LoRA/QLoRA training + serving | 06 |
| `unslothai/unsloth` | fast finetuning, XPU first-class (torch ≥2.6) | 06 |
| `steveseguin/Unofficial-Intel-XPU-Community` | stable guide hub | 07 |
| `steveseguin/b70-optimization-lab` | B70 research feed + website | 07 |
| `doopeworld/GRIMOIRE` | native B70 inference engine + dev log | 08 |
| `Wrapzii/k8v4-xpu` | version-pinning authority | 09 |
