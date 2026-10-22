"""Measure MTP acceptance for Swift-27B on the free Intel Arc XPU cards (2,3).

Acceptance = fraction of positions i where the MTP head's argmax prediction of
token i+2 (from main post-norm hidden state h_i + embedding of token i+1) equals
the true token i+2.

Corpus: a single text file (chunked to SEQ_LEN) OR a JSONL of {"domain","text"}
lines (each line = one sequence, padded to SEQ_LEN; acceptance reported per domain
+ overall). The JSONL mode benchmarks across json / tool_call / python / prose.

Loads the main model ONCE and measures the ORIGINAL mtp.* head (baseline) and, if
MTP_HEAD is set, the FINETUNED head too — reporting the delta. This is the fair
comparison: both heads see the identical fixed eval sequences.

Run with:  ZE_AFFINITY_MASK=2,3  (cards 2,3 = playground; 0,1 = the serving model)
"""
import os, sys, time, json, torch
from collections import Counter, defaultdict
sys.path.insert(0, "/home/haxor/AI/verl/mtp_work")
from mtp_head import MTPHead
from transformers import AutoConfig, AutoTokenizer
from safetensors import safe_open

MODEL = "/home/haxor/models/swift-27b"
CORPUS = os.environ.get("MTP_CORPUS", "/tmp/mtp-corpus/diverse_eval.jsonl")
SEQ_LEN = 128
BATCH = int(os.environ.get("MTP_BATCH", "4"))
N_SEQS = int(os.environ.get("MTP_NSEQS", "0"))  # 0 = all

def host_ram_free_gb():
    import subprocess
    out = subprocess.check_output(["free", "-g"]).decode()
    line = [l for l in out.splitlines() if l.startswith("Mem")][0]
    return int(line.split()[6])

def main():
    dev0 = "xpu:0"
    avail = host_ram_free_gb()
    print(f"host RAM available: {avail}GB (safe threshold 30GB)", flush=True)
    if avail < 30:
        print(f"ABORT: host RAM {avail}GB < 30GB — not touching the GPUs", flush=True)
        return
    print(f"torch {torch.__version__} | xpu devs visible: {torch.xpu.device_count()}", flush=True)

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
        no_split_module_classes=["Qwen3_5DecoderLayer"],
    )
    load_checkpoint_in_model(model, MODEL, device_map=device_map, dtype=torch.bfloat16)
    dispatch_model(model, device_map=device_map)
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    text_model = model.model.language_model
    final_norm = text_model.norm
    embed = text_model.embed_tokens
    lm_head = model.lm_head
    cfg = model.config.text_config
    head = MTPHead(cfg, embed, lm_head, device=dev0).to(dev0)

    # ---- load corpus: JSONL (per-domain) or single text (chunked) ----
    tok = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    seqs, domains = [], []
    if CORPUS.endswith(".jsonl"):
        for line in open(CORPUS):
            o = json.loads(line)
            ids = tok(o["text"], add_special_tokens=False)["input_ids"]
            if len(ids) < 16:
                continue
            ids = ids[:SEQ_LEN]
            seqs.append(torch.tensor(ids + [pad_id] * (SEQ_LEN - len(ids))))
            domains.append((o["domain"], len(ids)))
        print(f"jsonl corpus: {len(seqs)} seqs | domains: {dict(Counter(d for d,_ in domains))}", flush=True)
    else:
        text = open(CORPUS).read()
        ids = tok(text, add_special_tokens=False)["input_ids"]
        n_full = (len(ids) // SEQ_LEN) * SEQ_LEN
        for i in range(0, n_full, SEQ_LEN):
            seqs.append(torch.tensor(ids[i:i+SEQ_LEN])); domains.append(("text", SEQ_LEN))
        print(f"text corpus: {len(ids)} tokens -> {len(seqs)} seqs of {SEQ_LEN}", flush=True)
    if N_SEQS > 0:
        seqs = seqs[:N_SEQS]; domains = domains[:N_SEQS]

    # ---- measurement over the (fixed) eval sequences ----
    def measure(tag):
        accepts = defaultdict(int); totals = defaultdict(int)
        head.eval()
        with torch.no_grad():
            for b in range(0, len(seqs), BATCH):
                batch = torch.stack(seqs[b:b+BATCH]).to(dev0)
                am = torch.zeros_like(batch)
                for j in range(batch.size(0)):
                    am[j, :domains[b+j][1]] = 1
                out = model(input_ids=batch, attention_mask=am, output_hidden_states=True, use_cache=False)
                h = final_norm(out.hidden_states[-1]).to(dev0)
                h_flat = h[:, :-2, :].reshape(-1, cfg.hidden_size)
                tok_flat = batch[:, 1:-1].reshape(-1)
                tgt_flat = batch[:, 2:].reshape(-1)
                logits = head(h_flat, tok_flat)
                pred = logits.argmax(-1)
                match2d = (pred == tgt_flat).reshape(batch.size(0), -1)
                valid2d = (am[:, :-2] == 1) & (am[:, 2:] == 1)
                for j in range(batch.size(0)):
                    dom = domains[b+j][0]
                    accepts[dom] += int((match2d[j] & valid2d[j]).sum())
                    totals[dom] += int(valid2d[j].sum())
        print(f"\n=== {tag} ===")
        ga, gt = 0, 0
        for dom in sorted(totals):
            a, t = accepts[dom], totals[dom]
            ga += a; gt += t
            print(f"  {dom:12s} {a/t:.4f}   ({a}/{t})")
        print(f"  {'OVERALL':12s} {ga/gt:.4f}   ({ga}/{gt})")
        return ga / gt

    # baseline: original mtp.* weights
    f = f"{MODEL}/model-00018-of-00018.safetensors"
    mtp_state = {}
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if k.startswith("mtp."):
                mtp_state[k] = s.get_tensor(k)
    head.load_mtp_state(mtp_state)
    head = head.to(torch.bfloat16)
    base = measure("BASELINE (original mtp.*)")

    # finetuned (if MTP_HEAD set)
    MTP_HEAD = os.environ.get("MTP_HEAD")
    if MTP_HEAD:
        head.load_state_dict(torch.load(MTP_HEAD, map_location="cpu"))
        head = head.to(torch.bfloat16)
        ft = measure(f"FINETUNED ({os.path.basename(MTP_HEAD)})")
        print(f"\n*** DELTA: {ft-base:+.4f}   ({base:.4f} -> {ft:.4f}) ***")

if __name__ == "__main__":
    main()
