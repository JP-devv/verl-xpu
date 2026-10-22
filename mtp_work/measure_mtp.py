"""Measure MTP acceptance for Swift-27B on 2 Intel Arc XPU cards.

Acceptance = fraction of positions i where the MTP head's argmax prediction of
token i+2 (from main post-norm hidden state h_i + embedding of token i+1) equals
the true token i+2.

Run with:  ZE_AFFINITY_MASK=2,3  (cards 2,3 = our playground; 0,1 serve the agent)
"""
import os, sys, time, torch
sys.path.insert(0, "/home/haxor/AI/verl/mtp_work")
from mtp_head import MTPHead
from transformers import AutoConfig, AutoTokenizer
from safetensors import safe_open

MODEL = "/home/haxor/models/swift-27b"
CORPUS = "/tmp/mtp-corpus/corpus.txt"
SEQ_LEN = 128
BATCH = int(os.environ.get("MTP_BATCH", "4"))
N_SEQS = int(os.environ.get("MTP_NSEQS", "256"))

def host_ram_free_gb():
    import subprocess
    out = subprocess.check_output(["free", "-g"]).decode()
    line = [l for l in out.splitlines() if l.startswith("Mem")][0]
    return int(line.split()[6])  # 'available' column

def main():
    dev0 = "xpu:0"
    # PRE-FLIGHT GUARD: never risk the serving lane's (cards 0,1) host RAM.
    # Host RAM + the XPU driver are system-wide shared; a big in-RAM build OOMs the host.
    avail = host_ram_free_gb()
    print(f"host RAM available: {avail}GB (safe threshold 30GB)", flush=True)
    if avail < 30:
        print(f"ABORT: host RAM {avail}GB < 30GB — not touching the GPUs", flush=True)
        return
    print(f"torch {torch.__version__} | xpu devs visible: {torch.xpu.device_count()}", flush=True)

    # ---- load main model: MEMORY-BOUNDED (meta init + tensor-by-tensor to XPU) ----
    # Peak host RAM ~25GB (not 54GB) -> safe for the serving lane. No CPU/disk offload -> no meta tensors.
    from accelerate import init_empty_weights, load_checkpoint_in_model, infer_auto_device_map, dispatch_model
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    print("loading 27B model (memory-bounded: meta init + tensor-by-tensor to XPU)", flush=True)
    t0 = time.time()
    cfg_full = Qwen3_5Config.from_pretrained(MODEL)
    with init_empty_weights():
        model = Qwen3_5ForConditionalGeneration(cfg_full)
    device_map = infer_auto_device_map(
        model, max_memory={0: "31GiB", 1: "31GiB"},
        no_split_module_classes=["Qwen3_5DecoderLayer"],  # keep each layer on one device
    )
    load_checkpoint_in_model(model, MODEL, device_map=device_map, dtype=torch.bfloat16)
    dispatch_model(model, device_map=device_map)  # install inter-device activation hooks
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s | device_map: {dict(__import__('collections').Counter(str(v) for v in device_map.values()))}", flush=True)

    # ---- locate text model / norm / embed / lm_head ----
    text_model = model.model.language_model
    final_norm = text_model.norm
    embed = text_model.embed_tokens
    lm_head = model.lm_head
    print("text_model:", type(text_model).__name__, "| layers:", len(text_model.layers))

    # ---- build MTP head; SHARE the device-mapped embed/lm_head (no copy) ----
    cfg = model.config.text_config
    head = MTPHead(cfg, embed, lm_head, device=dev0).to(dev0)

    # ---- load MTP head weights (finetuned checkpoint if MTP_HEAD set, else original) ----
    MTP_HEAD = os.environ.get("MTP_HEAD")
    if MTP_HEAD:
        sd = torch.load(MTP_HEAD, map_location="cpu")
        head.load_state_dict(sd)
        rep = {"loaded": len(sd), "missing": [], "unexpected": []}
        print(f"MTP head loaded from FINETUNED checkpoint: {MTP_HEAD} ({len(sd)} tensors)")
    else:
        f = f"{MODEL}/model-00018-of-00018.safetensors"
        mtp_state = {}
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if k.startswith("mtp."):
                    mtp_state[k] = s.get_tensor(k)
        rep = head.load_mtp_state(mtp_state)
        print(f"MTP weights loaded (original): {rep['loaded']} | missing: {rep['missing']} | unexpected: {rep['unexpected']}")
    head = head.to(torch.bfloat16)  # match the main model's dtype

    # ---- load + chunk corpus ----
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = open(CORPUS).read()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    n_full = (len(ids) // SEQ_LEN) * SEQ_LEN
    ids = ids[:n_full]
    seqs = [torch.tensor(ids[i:i+SEQ_LEN]) for i in range(0, n_full, SEQ_LEN)]
    seqs = seqs[:N_SEQS]
    print(f"corpus: {len(ids)} tokens -> {len(seqs)} sequences of {SEQ_LEN}")

    # ---- measure ----
    accepts = 0
    total = 0
    head.eval()
    with torch.no_grad():
        for b in range(0, len(seqs), BATCH):
            batch = torch.stack(seqs[b:b+BATCH]).to(dev0)
            am = torch.ones_like(batch)
            # call the TOP-LEVEL model so accelerate's device-mapping hooks handle
            # the meta/CPU-offloaded layers; extract the text hidden states from its output
            out = model(input_ids=batch, attention_mask=am, output_hidden_states=True, use_cache=False)
            h = final_norm(out.hidden_states[-1]).to(dev0)          # [B,L,H] post-norm
            h_flat = h[:, :-2, :].reshape(-1, cfg.hidden_size)      # h_i, i in [0,L-3]
            tok_flat = batch[:, 1:-1].reshape(-1)                  # token i+1
            tgt_flat = batch[:, 2:].reshape(-1)                    # token i+2
            logits = head(h_flat, tok_flat)                        # [N,V]
            pred = logits.argmax(-1)
            accepts += int((pred == tgt_flat).sum())
            total += int(tgt_flat.numel())
            if b % (BATCH*8) == 0:
                print(f"  batch {b}/{len(seqs)} | running acceptance {accepts/total:.4f} ({total} positions)")

    acc = accepts / total
    print("\n" + "="*50)
    print(f"MTP ACCEPTANCE (baseline): {acc:.4f}  ({accepts}/{total} positions)")
    print("="*50)

if __name__ == "__main__":
    main()
