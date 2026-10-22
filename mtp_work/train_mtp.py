"""Finetune the Swift-27B MTP head to improve next-token acceptance.

- Main 27B model: FROZEN, loaded memory-bounded (meta init + tensor-by-tensor) with a
  MANUAL device_map: 18GiB on card0 (leaves 14GiB free for the MTP head), 30GiB on
  card1, ~3GiB on CPU. No disk offload.
- MTP head: trainable (~425M params: fc + one decoder layer + norms). embed/lm_head shared (frozen).
- Objective: CE of MTP logits vs true token i+2, given (h_i, embed(token i+1)).
- Trains on WikiText-103 TRAIN; acceptance is measured on the held-out TEST split.

Run under the RAM watchdog. ZE_AFFINITY_MASK=2,3; ONEAPI_DEVICE_SELECTOR unset.
"""
import os, sys, time, torch, torch.nn as nn
sys.path.insert(0, "/home/haxor/AI/verl/mtp_work")
from mtp_head import MTPHead
from transformers import AutoTokenizer
from safetensors import safe_open
from accelerate import init_empty_weights, load_checkpoint_in_model, dispatch_model
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

MODEL = "/home/haxor/models/swift-27b"
TRAIN = "/tmp/mtp-corpus/train.txt"
OUT = "/home/haxor/models/swift-27b-mtp-ft"
SEQ_LEN = 128
BATCH = int(os.environ.get("MTP_BATCH", "2"))
N_SEQS = int(os.environ.get("MTP_NSEQS", "512"))
EPOCHS = int(os.environ.get("MTP_EPOCHS", "2"))
LR = float(os.environ.get("MTP_LR", "1e-4"))
SAVE_EVERY = int(os.environ.get("MTP_SAVE_EVERY", "200"))
GiB = 1024**3

def host_ram_free_gb():
    import subprocess
    out = subprocess.check_output(["free", "-g"]).decode()
    return int([l for l in out.splitlines() if l.startswith("Mem")][0].split()[6])

def build_device_map(model, t_c0, t_c1, t_cpu):
    """Greedy whole-UNIT assignment (each decoder layer / embed / lm_head goes whole to one
    device; submodules inherit). Fill card1, then card0, then cpu. No disk, no split layers."""
    targets = {0: t_c0, 1: t_c1, "cpu": t_cpu}
    used = {0: 0, 1: 0, "cpu": 0}
    device_map = {}
    units = [("model.visual", model.model.visual),
             ("model.language_model.embed_tokens", model.model.language_model.embed_tokens)]
    for i, layer in enumerate(model.model.language_model.layers):
        units.append((f"model.language_model.layers.{i}", layer))
    units += [("model.language_model.norm", model.model.language_model.norm),
             ("lm_head", model.lm_head)]
    for name, mod in units:
        s = sum(p.numel() * 2 for p in mod.parameters())  # full unit size (bf16)
        for dev in [1, 0, "cpu"]:
            if used[dev] + s <= targets[dev] * GiB:
                device_map[name] = dev
                used[dev] += s
                break
        else:
            device_map[name] = "cpu"
            used["cpu"] += s
    return device_map

def main():
    dev0 = "xpu:0"
    avail = host_ram_free_gb()
    print(f"host RAM available: {avail}GB (safe threshold 30GB)", flush=True)
    if avail < 30:
        print(f"ABORT: host RAM {avail}GB < 30GB — not touching the GPUs", flush=True)
        return
    print(f"torch {torch.__version__} | xpu devs: {torch.xpu.device_count()}", flush=True)

    # ---- load frozen main model (memory-bounded, manual device_map) ----
    print("loading frozen 27B main model (manual map: c0=18 c1=30 cpu=6 GiB)...", flush=True)
    t0 = time.time()
    cfg_full = Qwen3_5Config.from_pretrained(MODEL)
    with init_empty_weights():
        model = Qwen3_5ForConditionalGeneration(cfg_full)
    device_map = build_device_map(model, 18, 30, 6)
    load_checkpoint_in_model(model, MODEL, device_map=device_map, dtype=torch.bfloat16)
    # safety: materialize any meta (cpu-offloaded) params into real CPU RAM
    import json
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    def _set_param(m, name, tensor):
        parts = name.split("."); mod = m
        for p in parts[:-1]:
            mod = getattr(mod, p)
        setattr(mod, parts[-1], torch.nn.Parameter(tensor, requires_grad=False))
    nmat = 0
    for name, param in list(model.named_parameters()):
        if param.is_meta:
            fname = idx.get(name)
            if fname:
                with safe_open(os.path.join(MODEL, fname), "pt") as s:
                    if name in s.keys():
                        _set_param(model, name, s.get_tensor(name).to(torch.bfloat16).cpu()); nmat += 1
    dispatch_model(model, device_map=device_map)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # forward sanity check before training
    _ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=dev0)
    with torch.no_grad():
        _out = model(input_ids=_ids, attention_mask=torch.ones_like(_ids), output_hidden_states=True, use_cache=False)
    print(f"main model loaded in {time.time()-t0:.0f}s | materialized {nmat} cpu params | forward OK hidden[-1]={tuple(_out.hidden_states[-1].shape)}", flush=True)

    text_model = model.model.language_model
    final_norm = text_model.norm
    embed = text_model.embed_tokens
    lm_head = model.lm_head
    cfg = model.config.text_config

    # ---- MTP head on card 0, trainable ----
    head = MTPHead(cfg, embed, lm_head, device=dev0).to(dev0)
    f = f"{MODEL}/model-00018-of-00018.safetensors"
    mtp_state = {}
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if k.startswith("mtp."):
                mtp_state[k] = s.get_tensor(k)
    rep = head.load_mtp_state(mtp_state)
    head = head.to(torch.bfloat16)
    print(f"MTP head loaded: {rep['loaded']} tensors", flush=True)

    trainable = [p for p in head.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"MTP trainable params: {n_train/1e6:.1f}M", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=LR)

    # ---- training corpus ----
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = open(TRAIN).read()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    n_full = (len(ids) // SEQ_LEN) * SEQ_LEN
    ids = ids[:n_full]
    seqs = [torch.tensor(ids[i:i+SEQ_LEN]) for i in range(0, n_full, SEQ_LEN)][:N_SEQS]
    print(f"train corpus: {len(seqs)} sequences of {SEQ_LEN}", flush=True)

    # ---- train ----
    os.makedirs(OUT, exist_ok=True)
    head.train()
    step = 0
    for epoch in range(EPOCHS):
        for b in range(0, len(seqs), BATCH):
            batch = torch.stack(seqs[b:b+BATCH]).to(dev0)
            am = torch.ones_like(batch)
            with torch.no_grad():
                out = model(input_ids=batch, attention_mask=am, output_hidden_states=True, use_cache=False)
                h = final_norm(out.hidden_states[-1]).to(dev0)
            h_flat = h[:, :-2, :].reshape(-1, cfg.hidden_size)
            tok_flat = batch[:, 1:-1].reshape(-1)
            tgt_flat = batch[:, 2:].reshape(-1)
            logits = head(h_flat, tok_flat)
            loss = nn.functional.cross_entropy(logits.float(), tgt_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} batch {b}/{len(seqs)} loss {loss.item():.4f}", flush=True)
            if step % SAVE_EVERY == 0:
                torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()},
                           f"{OUT}/mtp_head_step{step}.pt")
                print(f"  saved checkpoint step {step}", flush=True)

    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, f"{OUT}/mtp_head_final.pt")
    print(f"TRAINING DONE | final loss {loss.item():.4f} | saved to {OUT}/mtp_head_final.pt", flush=True)

if __name__ == "__main__":
    main()
