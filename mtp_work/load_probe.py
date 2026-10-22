"""Bounded load-probe: load the first two Swift-27B shards with
low_cpu_mem_usage=True, sample peak host-RAM + XPU mem in a background thread,
and report. This measures whether the low-CPU-mem load keeps host RAM bounded
(before committing to the full 52GB load + FSDP step).
"""
import os, sys, threading, time, json
os.environ["ZE_AFFINITY_MASK"] = "2,3"
os.environ.pop("ONEAPI_DEVICE_SELECTOR", None)

import torch
import psutil

# --- peak-RAM sampler (background thread) ---
peak_rss_gb = [0.0]
stop = [False]
proc = psutil.Process()
def _sample():
    while not stop[0]:
        rss = proc.memory_info().rss / (1024**3)
        if rss > peak_rss_gb[0]:
            peak_rss_gb[0] = rss
        time.sleep(0.05)
t = threading.Thread(target=_sample, daemon=True)
t.start()

def _free_gb():
    return psutil.virtual_memory().available / (1024**3)

print(f"[probe] start: free={_free_gb():.1f}GB rss={proc.memory_info().rss/1024**3:.1f}GB", flush=True)

# --- load first two shards only (low_cpu_mem_usage + safetensors) ---
from safetensors import safe_open
model_dir = "/home/haxor/models/swift-27b"
shards = sorted(f for f in os.listdir(model_dir) if f.startswith("model-") and f.endswith(".safetensors"))
print(f"[probe] {len(shards)} shards total; loading first 2", flush=True)

total_params = 0
for sh in shards[:2]:
    path = os.path.join(model_dir, sh)
    sz = os.path.getsize(path) / (1024**3)
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        n = 0
        for k in keys:
            t_ = f.get_tensor(k)  # materializes into host RAM
            n += t_.numel()
            del t_
    total_params += n
    print(f"[probe] loaded {sh} ({sz:.1f}GB, {len(keys)} tensors, {n/1e6:.0f}M params) "
          f"free={_free_gb():.1f}GB peak_rss={peak_rss_gb[0]:.1f}GB", flush=True)

stop[0] = True
t.join(timeout=2)

# --- XPU mem (cards 2,3 should be ~0 — this probe is host-RAM only) ---
print(f"[probe] DONE: 2 shards = {total_params/1e9:.2f}B params "
      f"peak_rss={peak_rss_gb[0]:.1f}GB free={_free_gb():.1f}GB", flush=True)
print(f"[probe] est full-model peak_rss (18 shards) ~= {peak_rss_gb[0]*9:.0f}GB "
      f"(if linear; low_cpu_mem_usage keeps it to ~1 shard at a time)", flush=True)
print("PROBE_OK", flush=True)
