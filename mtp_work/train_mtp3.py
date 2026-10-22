"""All-on-XPU single-phase MTP finetune for Swift-27B.

The two-phase approach (train_mtp2.py) is CPU-bound in phase 2: it harvests h_i to
CPU, then shuttles them back to the XPU every step, and a 424M head can't saturate a
B70. This version keeps EVERYTHING on the cards:

  - The frozen 27B main model stays loaded across the two free cards (device_map,
    NO CPU offload -> no meta tensors).
  - The MTP head lives on the LAST card (where final_norm + lm_head live), with a
    local copy of embed_tokens on that same card.
  - Each step: run the main model forward (no grad) to get h_i ON the last card,
    run the MTP head (with grad) on that same card, backprop through the head only.

No CPU<->XPU round-trip for h_i; both cards do real work (main-model forward spans
both, MTP head fwd/bwd on the last). This is the same shape verl's FSDP engine uses.

Run under the RAM watchdog. ZE_AFFINITY_MASK=2,3; ONEAPI_DEVICE_SELECTOR unset.
"""
import os, sys, time, json, random, torch, torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "/home/haxor/AI/verl/mtp_work")
from mtp_head import MTPHead
from transformers import AutoTokenizer
from safetensors import safe_open
from accelerate import init_empty_weights, load_checkpoint_in_model, infer_auto_device_map, dispatch_model
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

MODEL = "/home/haxor/models/swift-27b"
TRAIN = "/tmp/mtp-corpus/diverse_train.jsonl"
VAL   = "/tmp/mtp-corpus/diverse_eval.jsonl"
OUT   = "/home/haxor/models/swift-27b-mtp-ft"
SEQ_LEN = 128
BATCH   = int(os.environ.get("MTP_BATCH", "2"))   # small: keep activations on-card
EPOCHS  = int(os.environ.get("MTP_EPOCHS", "8"))
LR      = float(os.environ.get("MTP_LR", "1e-4"))
MARGIN  = float(os.environ.get("MTP_MARGIN", "1.0"))
LAMBDA  = float(os.environ.get("MTP_LAMBDA", "0.5"))
PATIENCE = int(os.environ.get("MTP_PATIENCE", "3"))
random.seed(0)

def host_ram_free_gb():
    import subprocess
    out = subprocess.check_output(["free", "-g"]).decode()
    return int([l for l in out.splitlines() if l.startswith("Mem")][0].split()[6])

def load_jsonl_seqs(path, tok, pad_id):
    seqs, domains = [], []
    for line in open(path):
        o = json.loads(line)
        ids = tok(o["text"], add_special_tokens=False)["input_ids"]
        if len(ids) < 16:
            continue
        ids = ids[:SEQ_LEN]
        seqs.append(torch.tensor(ids + [pad_id] * (SEQ_LEN - len(ids))))
        domains.append((o["domain"], len(ids)))
    return seqs, domains

def main():
    avail = host_ram_free_gb()
    print(f"host RAM available: {avail}GB (safe threshold 30GB)", flush=True)
    if avail < 30:
        print("ABORT: host RAM < 30GB"); return 1
    cfg_full = Qwen3_5Config.from_pretrained(MODEL)
    cfg = cfg_full.text_config
    H = cfg.hidden_size
    tok = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    # ---- load the frozen 27B across the two free cards (no CPU offload) ----
    print("=== loading frozen 27B on cards 2,3 (device_map, no offload) ===", flush=True)
    t0 = time.time()
    with init_empty_weights():
        model = Qwen3_5ForConditionalGeneration(cfg_full)
    device_map = infer_auto_device_map(model, max_memory={0: "31GiB", 1: "31GiB"},
                                       no_split_module_classes=["Qwen3_5DecoderLayer"])
    load_checkpoint_in_model(model, MODEL, device_map=device_map, dtype=torch.bfloat16)
    dispatch_model(model, device_map=device_map)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"main model loaded in {time.time()-t0:.0f}s", flush=True)

    text_model = model.model.language_model
    final_norm = text_model.norm
    lm_head = model.lm_head
    # the last card = where final_norm + lm_head live (visible index 1 = physical 3)
    last_dev = next(final_norm.parameters()).device
    print(f"final_norm/lm_head on {last_dev}; MTP head will live there", flush=True)

    # ---- local copy of embed_tokens on the last card (main's embed is on card 0) ----
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(MODEL, idx["model.language_model.embed_tokens.weight"]), "pt") as s:
        embed_w = s.get_tensor("model.language_model.embed_tokens.weight").to(torch.bfloat16).to(last_dev)
    embed_copy = nn.Embedding(cfg.vocab_size, H)
    embed_copy.weight = nn.Parameter(embed_w)
    embed_copy.to(last_dev)
    for p in embed_copy.parameters():
        p.requires_grad_(False)

    # ---- build the MTP head on the last card (shares lm_head, local embed) ----
    head = MTPHead(cfg, embed_copy, lm_head, device=str(last_dev)).to(last_dev)
    f = f"{MODEL}/model-00018-of-00018.safetensors"
    mtp_state = {}
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if k.startswith("mtp."): mtp_state[k] = s.get_tensor(k)
    head.load_mtp_state(mtp_state)
    head = head.to(torch.bfloat16)
    for p in lm_head.parameters():
        p.requires_grad_(False)
    trainable = [p for p in head.parameters() if p.requires_grad]
    print(f"MTP trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M (all on {last_dev})", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=LR)

    # ---- data ----
    train_seqs, _ = load_jsonl_seqs(TRAIN, tok, pad_id)
    val_seqs, val_dom = load_jsonl_seqs(VAL, tok, pad_id)

    def acceptance(seqs, doms):
        head.eval()
        a, t = 0, 0
        with torch.no_grad():
            for b in range(0, len(seqs), BATCH):
                batch = torch.stack(seqs[b:b+BATCH]).to("xpu:0")
                am = torch.zeros_like(batch)
                for j in range(batch.size(0)):
                    am[j, :doms[b+j][1]] = 1
                out = model(input_ids=batch, attention_mask=am, output_hidden_states=True, use_cache=False)
                h = final_norm(out.hidden_states[-1]).to(last_dev)
                h_flat = h[:, :-2, :].reshape(-1, H)
                tok_flat = batch[:, 1:-1].reshape(-1)
                tgt_flat = batch[:, 2:].reshape(-1).to(last_dev)
                am1 = am[:, :-2].reshape(-1).to(last_dev); am2 = am[:, 2:].reshape(-1).to(last_dev)
                valid = (am1 == 1) & (am2 == 1)
                logits = head(h_flat, tok_flat)
                pred = logits.argmax(-1)
                a += int(((pred == tgt_flat) & valid).sum()); t += int(valid.sum())
        return a / t if t else 0.0

    base_val = acceptance(val_seqs, val_dom)
    print(f"baseline val acceptance: {base_val:.4f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/baseline_val_acc.txt", "w") as f: f.write(f"{base_val:.6f}\n")

    best_val = base_val
    best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    bad = 0; step = 0; t_start = time.time()
    for epoch in range(EPOCHS):
        random.shuffle(train_seqs)
        head.train()
        for b in range(0, len(train_seqs), BATCH):
            batch = torch.stack(train_seqs[b:b+BATCH]).to("xpu:0")
            am = torch.ones_like(batch)
            # main model forward (frozen, no grad) -> h_i on the last card
            with torch.no_grad():
                out = model(input_ids=batch, attention_mask=am, output_hidden_states=True, use_cache=False)
                h = final_norm(out.hidden_states[-1]).to(last_dev)
            h_flat = h[:, :-2, :].reshape(-1, H)
            tok_flat = batch[:, 1:-1].reshape(-1)
            tgt_flat = batch[:, 2:].reshape(-1).to(last_dev)
            logits = head(h_flat, tok_flat).float()
            ce = F.cross_entropy(logits, tgt_flat)
            z_true = logits.gather(1, tgt_flat.unsqueeze(1)).squeeze(1)
            lm = logits.clone(); lm.scatter_(1, tgt_flat.unsqueeze(1), float("-inf"))
            max_other = lm.max(dim=1).values
            margin_loss = torch.relu(MARGIN - (z_true - max_other)).mean()
            loss = ce + LAMBDA * margin_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            step += 1
        val_acc = acceptance(val_seqs, val_dom)
        ep_s = time.time() - t_start
        t_start = time.time()
        print(f"epoch {epoch:2d} | loss {loss.item():.4f} | val acc {val_acc:.4f} (best {best_val:.4f})", flush=True)
        if val_acc > best_val + 1e-4:
            best_val = val_acc
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    torch.save(best_state, f"{OUT}/mtp_head_final.pt")
    with open(f"{OUT}/best_val_acc.txt", "w") as f: f.write(f"{best_val:.6f}\n")
    print(f"DONE | best val acc {best_val:.4f} (baseline {base_val:.4f}) | all-on-XPU, no CPU h_i round-trip", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
