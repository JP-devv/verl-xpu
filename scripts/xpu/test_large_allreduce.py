"""Probe: large (>=512MB) all-reduce over the native XCCL SYCL device backend.

This is the test that decides whether real-model FSDP can run on cards 2,3.
The old path (CCL_ATL_TRANSPORT=mpi + I_MPI_FABRICS=shm) deadlocks on large
all-reduces because it shuttles data through host memory. The native `xccl`
backend with the CCL_SYCL_* SYCL-kernel knobs does GPU-to-GPU allreduce and
scales to large messages.

Launch (2 procs on the free cards):
  ZE_AFFINITY_MASK=2,3  (ONEAPI_DEVICE_SELECTOR unset)
  CCL_SYCL_ALLREDUCE_LL=ring  CCL_SYCL_ALLREDUCE_SIMPLE_READ=1
  CCL_SYCL_COPY_ENGINE=1  CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
  CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
  CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296  CCL_SYCL_ALLTOALL_TMP_BUF=1
  torchrun --nproc_per_node=2 test_large_allreduce.py

Exit 0 + "LARGE ALL-REDUCE OK" = the transport is unblocked.
"""
import os, sys, time, torch
import torch.distributed as dist

MB = 1024 * 1024
SIZE_MB = int(os.environ.get("AR_SIZE_MB", "512"))  # tensor size in MiB (bf16)

def main():
    dist.init_process_group("xccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.xpu.set_device(rank)
    dev = f"xpu:{rank}"

    n = SIZE_MB * MB // 2  # bf16 = 2 bytes
    t = torch.ones(n, device=dev, dtype=torch.bfloat16)
    if rank == 0:
        print(f"[rank0] all-reduce {SIZE_MB}MiB bf16 over xccl (world={world})", flush=True)

    # warmup (small) then the large one
    small = torch.ones(1024, device=dev, dtype=torch.bfloat16)
    dist.all_reduce(small)
    dist.barrier()

    t0 = time.time()
    dist.all_reduce(t)  # SUM
    torch.xpu.synchronize()
    dt = time.time() - t0
    gbs = (SIZE_MB * world) / dt / 1024  # rough GB/s (both directions)

    ok = abs(float(t[0].item()) - world) < 1e-3
    if rank == 0:
        print(f"[rank0] large all-reduce done in {dt:.2f}s (~{gbs:.2f} GB/s), "
              f"t[0]={t[0].item()} (expect {world})", flush=True)
        print("LARGE ALL-REDUCE OK" if ok else "LARGE ALL-REDUCE WRONG RESULT", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
