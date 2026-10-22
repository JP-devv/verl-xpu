"""Build a finetuned Swift-27B model dir: symlink all original shards except the
last, and write a modified last shard with the mtp.* weights replaced by the
trained MTPHead weights (key-mapped: layer.X -> mtp.layers.0.X, etc.)."""
import os, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/home/haxor/models/swift-27b"
DST = "/home/haxor/models/swift-27b-mtp-ft-model"
HEAD = "/home/haxor/models/swift-27b-mtp-ft/mtp_head_final.pt"
LAST = "model-00018-of-00018.safetensors"

# key map: trained-head name -> original mtp.* name
def map_key(k):
    if k == "fc.weight": return "mtp.fc.weight"
    if k == "norm.weight": return "mtp.norm.weight"
    if k.startswith("pre_fc_norm_"): return f"mtp.{k}"
    if k.startswith("layer."): return k.replace("layer.", "mtp.layers.0.", 1)
    raise KeyError(k)

os.makedirs(DST, exist_ok=True)

# 1. symlink everything except the last shard
for f in os.listdir(SRC):
    dst = os.path.join(DST, f)
    if f == LAST:
        continue
    if not os.path.exists(dst):
        os.symlink(os.path.join(SRC, f), dst)
        print(f"symlink {f}")

# 2. load the last shard, replace mtp.* weights, save
print("loading last shard...")
tensors = {}
with safe_open(os.path.join(SRC, LAST), "pt") as sf:
    for k in sf.keys():
        tensors[k] = sf.get_tensor(k)

print(f"last shard: {len(tensors)} tensors, {sum(v.numel() for v in tensors.values())/1e6:.0f}M params")

# load trained head and map keys
head = torch.load(HEAD, map_location="cpu")
print(f"trained head: {len(head)} tensors")
n_replaced = 0
for hk, hv in head.items():
    mk = map_key(hk)
    assert mk in tensors, f"mapped key {mk} not in last shard"
    assert tensors[mk].shape == hv.shape, f"shape mismatch {mk}: {tensors[mk].shape} vs {hv.shape}"
    tensors[mk] = hv.to(torch.bfloat16)
    n_replaced += 1
print(f"replaced {n_replaced} mtp.* tensors with trained head weights")

# save modified last shard
out = os.path.join(DST, LAST)
save_file(tensors, out)
print(f"saved {out} ({os.path.getsize(out)/1e9:.2f} GB)")
print("DONE")
