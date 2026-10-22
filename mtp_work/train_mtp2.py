"""Two-phase MTP finetune for Swift-27B — diverse, acceptance-optimized.

Phase 1: load the FULL frozen main model on the free cards (2,3); run the diverse
         TRAIN and VAL (eval) sets through it; save every (h_i, tok i+1, tok i+2)
         triple (+ validity masks) to disk. h_i = post-final-norm hidden state.
Phase 2: unload the main model; load embed_tokens + lm_head + MTP head on ONE card;
         train the MTP head with CE + margin (acceptance-direct) loss; early-stop on
         VAL acceptance (computed from the saved val h_i, no full model needed).

The margin loss pushes the true token's logit above the runner-up by MARGIN, which
optimizes argmax acceptance directly (the metric we care about), not just CE.

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
HTRAIN = "/tmp/mtp-hstore-train.pt"
HVAL   = "/tmp/mtp-hstore-val.pt"
SEQ_LEN = 128
BATCH   = int(os.environ.get("MTP_BATCH", "4"))
EPOCHS  = int(os.environ.get("MTP_EPOCHS", "12"))
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

def harvest(model, final_norm, path, tok, pad_id, H, tag):
    seqs, domains = load_jsonl_seqs(path, tok, pad_id)
    print(f"[{tag}] {len(seqs)} seqs", flush=True)
    hstore = []
    with torch.no_grad():
        for b in range(0, len(seqs), BATCH):
            batch = torch.stack(seqs[b:b+BATCH]).to("xpu:0")
            am = torch.zeros_like(batch)
            for j in range(batch.size(0)):
                am[j, :domains[b+j][1]] = 1
            out = model(input_ids=batch, attention_mask=am, output_hidden_states=True, use_cache=False)
            h = final_norm(out.hidden_states[-1])  # [B,L,H]
            hstore.append((h[:, :-2, :].reshape(-1, H).cpu(),
                           batch[:, 1:-1].reshape(-1).cpu(),
                           batch[:, 2:].reshape(-1).cpu(),
                           am[:, :-2].reshape(-1).cpu(),
                           am[:, 2:].reshape(-1).cpu()))
    return hstore

def val_acceptance(head, hstore):
    head.eval()
    a, t = 0, 0
    with torch.no_grad():
        for (h, tokf, tgt, am1, am2) in hstore:
            h, tokf, tgt = h.to("xpu:0"), tokf.to("xpu:0"), tgt.to("xpu:0")
            am1, am2 = am1.to("xpu:0"), am2.to("xpu:0")
            valid = (am1 == 1) & (am2 == 1)
            logits = head(h, tokf)
            pred = logits.argmax(-1)
            a += int(((pred == tgt) & valid).sum())
            t += int(valid.sum())
    return a / t if t else 0.0

def main():
    dev0 = "xpu:0"
    avail = host_ram_free_gb()
    print(f"host RAM available: {avail}GB (safe threshold 30GB)", flush=True)
    if avail < 30:
        print("ABORT: host RAM < 30GB"); return
    cfg_full = Qwen3_5Config.from_pretrained(MODEL)
    cfg = cfg_full.text_config
    H = cfg.hidden_size
    tok = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    # ================= PHASE 1: harvest train + val h_i =================
    if os.path.exists(HTRAIN) and os.path.exists(HVAL):
        print(f"=== PHASE 1: skipping (hstores exist) ===", flush=True)
    else:
        print("=== PHASE 1: load full model on 2 cards, harvest train+val h_i ===", flush=True)
        t0 = time.time()
        with init_empty_weights():
            model = Qwen3_5ForConditionalGeneration(cfg_full)
        device_map = infer_auto_device_map(model, max_memory={0: "31GiB", 1: "31GiB"},
                                           no_split_module_classes=["Qwen3_5DecoderLayer"])
        load_checkpoint_in_model(model, MODEL, device_map=device_map, dtype=torch.bfloat16)
        dispatch_model(model, device_map=device_map)
        model.eval()
        print(f"main model loaded in {time.time()-t0:.0f}s", flush=True)
        text_model = model.model.language_model
        final_norm = text_model.norm
        htrain = harvest(model, final_norm, TRAIN, tok, pad_id, H, "train")
        hval   = harvest(model, final_norm, VAL, tok, pad_id, H, "val")
        torch.save(htrain, HTRAIN); torch.save(hval, HVAL)
        nt = sum(x[0].shape[0] for x in htrain); nv = sum(x[0].shape[0] for x in hval)
        print(f"phase1 done: {nt} train triples, {nv} val triples", flush=True)
        del model, text_model, final_norm
        torch.xpu.empty_cache(); import gc; gc.collect()
        print(f"main model unloaded | host RAM now {host_ram_free_gb()}GB free", flush=True)

    # ================= PHASE 2: train MTP head (CE + margin) =================
    print("=== PHASE 2: train MTP head on one card, early-stop on val ===", flush=True)
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    def load_tensor(name, dev):
        with safe_open(os.path.join(MODEL, idx[name]), "pt") as s:
            return s.get_tensor(name).to(torch.bfloat16).to(dev)
    embed = nn.Embedding(cfg.vocab_size, H); embed.weight = nn.Parameter(load_tensor("model.language_model.embed_tokens.weight", dev0))
    lm_head = nn.Linear(H, cfg.vocab_size, bias=False); lm_head.weight = nn.Parameter(load_tensor("lm_head.weight", dev0))
    head = MTPHead(cfg, embed, lm_head, device=dev0).to(dev0)
    f = f"{MODEL}/model-00018-of-00018.safetensors"
    mtp_state = {}
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if k.startswith("mtp."): mtp_state[k] = s.get_tensor(k)
    head.load_mtp_state(mtp_state)
    head = head.to(torch.bfloat16)
    for p in embed.parameters(): p.requires_grad_(False)
    for p in lm_head.parameters(): p.requires_grad_(False)
    trainable = [p for p in head.parameters() if p.requires_grad]
    print(f"MTP trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=LR)

    htrain = torch.load(HTRAIN)
    hval   = torch.load(HVAL)
    base_val = val_acceptance(head, hval)
    print(f"baseline val acceptance: {base_val:.4f}", flush=True)

    best_val = base_val
    best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    bad = 0; step = 0
    for epoch in range(EPOCHS):
        random.shuffle(htrain)
        head.train()
        for (h, tokf, tgt, am1, am2) in htrain:
            h, tokf, tgt = h.to(dev0), tokf.to(dev0), tgt.to(dev0)
            am1, am2 = am1.to(dev0), am2.to(dev0)
            valid = (am1 == 1) & (am2 == 1)
            if int(valid.sum()) == 0:
                continue
            hv, tv, gv = h[valid], tokf[valid], tgt[valid]
            logits = head(hv, tv).float()
            ce = F.cross_entropy(logits, gv)
            z_true = logits.gather(1, gv.unsqueeze(1)).squeeze(1)
            lm = logits.clone(); lm.scatter_(1, gv.unsqueeze(1), float("-inf"))
            max_other = lm.max(dim=1).values
            margin_loss = torch.relu(MARGIN - (z_true - max_other)).mean()
            loss = ce + LAMBDA * margin_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            step += 1
        val_acc = val_acceptance(head, hval)
        print(f"epoch {epoch:2d} | loss {loss.item():.4f} | val acc {val_acc:.4f} (best {best_val:.4f})", flush=True)
        if val_acc > best_val + 1e-4:
            best_val = val_acc
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stop at epoch {epoch} (no val gain for {PATIENCE} epochs)", flush=True)
                break

    os.makedirs(OUT, exist_ok=True)
    torch.save(best_state, f"{OUT}/mtp_head_final.pt")
    print(f"DONE | best val acc {best_val:.4f} (baseline {base_val:.4f}) | saved {OUT}/mtp_head_final.pt", flush=True)

if __name__ == "__main__":
    main()
