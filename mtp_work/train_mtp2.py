"""Two-phase MTP finetune for Swift-27B (avoids all CPU-offload/meta issues).

Phase 1: load the FULL frozen main model on both free cards (no CPU offload -> no meta
         tensors), run the training corpus through it, and save every (h_i, token i+1,
         token i+2) triple to disk. h_i = post-final-norm hidden state.
Phase 2: unload the main model; load just embed_tokens + lm_head + the MTP head onto
         ONE card and train the MTP head (CE vs true token i+2) against the saved h_i.

Result: finetuned MTP head saved to OUT. Acceptance is measured separately on the
held-out TEST split (measure_mtp.py --head <OUT>).

Run under the RAM watchdog. ZE_AFFINITY_MASK=2,3; ONEAPI_DEVICE_SELECTOR unset.
"""
import os, sys, time, json, torch, torch.nn as nn
sys.path.insert(0, "/home/haxor/AI/verl/mtp_work")
from mtp_head import MTPHead
from transformers import AutoTokenizer
from safetensors import safe_open
from accelerate import init_empty_weights, load_checkpoint_in_model, infer_auto_device_map, dispatch_model
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

MODEL = "/home/haxor/models/swift-27b"
TRAIN = "/tmp/mtp-corpus/train.txt"
OUT = "/home/haxor/models/swift-27b-mtp-ft"
HSTORE = "/tmp/mtp-hstore.pt"
SEQ_LEN = 128
BATCH = int(os.environ.get("MTP_BATCH", "4"))
N_SEQS = int(os.environ.get("MTP_NSEQS", "1024"))
EPOCHS = int(os.environ.get("MTP_EPOCHS", "3"))
LR = float(os.environ.get("MTP_LR", "1e-4"))
GiB = 1024**3

def host_ram_free_gb():
    import subprocess
    out = subprocess.check_output(["free", "-g"]).decode()
    return int([l for l in out.splitlines() if l.startswith("Mem")][0].split()[6])

def load_mtp_state_dict():
    f = f"{MODEL}/model-00018-of-00018.safetensors"
    d = {}
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if k.startswith("mtp."):
                d[k] = s.get_tensor(k)
    return d

def main():
    dev0 = "xpu:0"
    avail = host_ram_free_gb()
    print(f"host RAM available: {avail}GB (safe threshold 30GB)", flush=True)
    if avail < 30:
        print(f"ABORT: host RAM {avail}GB < 30GB", flush=True)
        return
    cfg_full = Qwen3_5Config.from_pretrained(MODEL)
    cfg = cfg_full.text_config
    H = cfg.hidden_size

    # ================= PHASE 1: generate + save h_i =================
    if os.path.exists(HSTORE):
        print(f"=== PHASE 1: skipping (hstore exists at {HSTORE}) ===", flush=True)
    else:
        print("=== PHASE 1: load full main model on 2 cards, generate h_i ===", flush=True)
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

        tok = AutoTokenizer.from_pretrained(MODEL)
        text = open(TRAIN).read()
        ids = tok(text, add_special_tokens=False)["input_ids"]
        n_full = (len(ids) // SEQ_LEN) * SEQ_LEN
        ids = ids[:n_full]
        seqs = [torch.tensor(ids[i:i+SEQ_LEN]) for i in range(0, n_full, SEQ_LEN)][:N_SEQS]
        print(f"train corpus: {len(seqs)} sequences of {SEQ_LEN}", flush=True)

        hstore = []  # list of (h_flat [N,H] cpu, tok_flat [N] cpu, tgt_flat [N] cpu)
        with torch.no_grad():
            for b in range(0, len(seqs), BATCH):
                batch = torch.stack(seqs[b:b+BATCH]).to(dev0)
                am = torch.ones_like(batch)
                out = model(input_ids=batch, attention_mask=am, output_hidden_states=True, use_cache=False)
                h = final_norm(out.hidden_states[-1])  # [B,L,H]
                hstore.append((h[:, :-2, :].reshape(-1, H).cpu(),
                               batch[:, 1:-1].reshape(-1).cpu(),
                               batch[:, 2:].reshape(-1).cpu()))
                if b % (BATCH*16) == 0:
                    print(f"  phase1 batch {b}/{len(seqs)}", flush=True)
        torch.save(hstore, HSTORE)
        ntrip = sum(x[0].shape[0] for x in hstore)
        print(f"phase1 done: saved {ntrip} triples to {HSTORE}", flush=True)

        # unload the main model to free the cards
        del model, text_model, final_norm
        torch.xpu.empty_cache()
        del device_map
        import gc; gc.collect()
        print(f"main model unloaded | host RAM now {host_ram_free_gb()}GB free", flush=True)

    # ================= PHASE 2: train MTP head on one card =================
    print("=== PHASE 2: load embed+lm_head+MTP head on card0, train ===", flush=True)
    # load embed_tokens + lm_head from safetensors onto card0
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    def load_tensor(name, dev):
        fname = idx[name]
        with safe_open(os.path.join(MODEL, fname), "pt") as s:
            return s.get_tensor(name).to(torch.bfloat16).to(dev)
    embed_w = load_tensor("model.language_model.embed_tokens.weight", dev0)
    lmhead_w = load_tensor("lm_head.weight", dev0)
    embed = nn.Embedding(cfg.vocab_size, H)
    embed.weight = nn.Parameter(embed_w)
    lm_head = nn.Linear(H, cfg.vocab_size, bias=False)
    lm_head.weight = nn.Parameter(lmhead_w)
    embed.to(dev0); lm_head.to(dev0)

    head = MTPHead(cfg, embed, lm_head, device=dev0).to(dev0)
    rep = head.load_mtp_state(load_mtp_state_dict())
    head = head.to(torch.bfloat16)
    print(f"MTP head loaded: {rep['loaded']} tensors", flush=True)

    # freeze embed + lm_head (only train the mtp-specific params)
    for p in embed.parameters():
        p.requires_grad_(False)
    for p in lm_head.parameters():
        p.requires_grad_(False)
    trainable = [p for p in head.parameters() if p.requires_grad]
    print(f"MTP trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=LR)

    hstore = torch.load(HSTORE)
    head.train()
    step = 0
    for epoch in range(EPOCHS):
        for (h_flat, tok_flat, tgt_flat) in hstore:
            h_flat = h_flat.to(dev0); tok_flat = tok_flat.to(dev0); tgt_flat = tgt_flat.to(dev0)
            logits = head(h_flat, tok_flat)
            loss = nn.functional.cross_entropy(logits.float(), tgt_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{EPOCHS*len(hstore)} loss {loss.item():.4f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, f"{OUT}/mtp_head_final.pt")
    print(f"TRAINING DONE | final loss {loss.item():.4f} | saved to {OUT}/mtp_head_final.pt", flush=True)

if __name__ == "__main__":
    main()
