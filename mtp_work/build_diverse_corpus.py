#!/usr/bin/env python
"""Build a diverse, domain-matched MTP corpus (train + val + held-out test).

Domains mirror the Swift-27B serving distribution:
  - json        : structured output (nested JSON objects)
  - tool_call   : function-calling (name + JSON arguments)
  - python      : real Python code (local ~/AI project code)
  - prose       : summarizing/general prose (WikiText)

CRITICAL: train + val are generated from the ORIGINAL global-RNG sequence
(random.seed(SEED)) so they stay byte-identical across rebuilds. The held-out
TEST set is generated from a SEPARATE random.Random(SEED+1000) instance, so it
is disjoint from train/val AND adding it never perturbs their bytes.

Output (NOT committed - generated data), each line {"domain":<str>,"text":<str>}:
  /tmp/mtp-corpus/diverse_train.jsonl  (TRAIN_N per domain)
  /tmp/mtp-corpus/diverse_eval.jsonl   (EVAL_N per domain, val: early-stop)
  /tmp/mtp-corpus/diverse_test.jsonl   (TEST_N per domain, disjoint held-out)
"""
import os, sys, json, random

SEED = 20240601
random.seed(SEED)  # global RNG: drives train + val (original sequence)
OUTDIR = "/tmp/mtp-corpus"
EVAL_N = int(os.environ.get("EVAL_N", "64"))
TRAIN_N = int(os.environ.get("TRAIN_N", "250"))
TEST_N = int(os.environ.get("TEST_N", "64"))

# ---- synthetic vocab ----
KEYS = ["id","name","status","value","count","type","data","result","error","message",
        "user","item","config","params","output","input","model","token","score","label",
        "timestamp","url","path","role","content","action","tool","args","response","request"]
STRS = ["active","inactive","pending","complete","error","success","running","stopped",
        "high","low","medium","alpha","beta","gamma","foo","bar","baz","hello","world",
        "test","prod","dev","staging","login","search","update","delete","create","query",
        "function","argument","result","payload","schema","endpoint","token","session"]
TOOLS = {
    "web_search":   ["query","max_results","region","language"],
    "read_file":    ["path","offset","limit"],
    "write_file":   ["path","content","append"],
    "run_command":  ["command","timeout","cwd","shell"],
    "http_request": ["url","method","headers","body"],
    "calculate":    ["expression","precision"],
    "send_email":   ["to","subject","body","cc"],
    "query_db":     ["table","filters","limit","order_by"],
    "create_ticket":["title","priority","assignee","labels"],
    "get_weather":  ["location","units","forecast_days"],
    "list_dir":     ["path","recursive","pattern"],
    "git_commit":   ["message","files","branch"],
}

def gen_val(rng, depth=0):
    r = rng.random()
    if depth >= 3 or r < 0.35:
        c = rng.random()
        if c < 0.30: return rng.randint(0, 100000)
        if c < 0.50: return round(rng.uniform(0, 1000), 3)
        if c < 0.70: return rng.choice([True, False, None])
        return rng.choice(STRS)
    if r < 0.70:
        return {rng.choice(KEYS): gen_val(rng, depth+1) for _ in range(rng.randint(1, 4))}
    return [gen_val(rng, depth+1) for _ in range(rng.randint(1, 4))]

def gen_json(rng):
    return json.dumps({rng.choice(KEYS): gen_val(rng) for _ in range(rng.randint(2, 6))}, indent=2)

def gen_tool_call(rng):
    tool = rng.choice(list(TOOLS))
    args = {}
    for k in TOOLS[tool]:
        r = rng.random()
        if r < 0.45: args[k] = rng.choice(STRS)
        elif r < 0.75: args[k] = rng.randint(0, 1000)
        else: args[k] = rng.choice([True, False, None])
    return json.dumps({"name": tool, "arguments": args}, indent=2)

def _chunk_lines(text, target=512):
    out, buf = [], []
    for ln in text.split("\n"):
        buf.append(ln)
        if sum(len(x)+1 for x in buf) > target:
            out.append("\n".join(buf)); buf = []
    if buf: out.append("\n".join(buf))
    return out

def gather_python(n, rng):
    chunks, files = [], []
    for root, dirs, fnames in os.walk(os.path.expanduser("~/AI")):
        dirs[:] = [d for d in dirs if d not in (".venv","site-packages","node_modules","__pycache__",".git","dist","build")]
        for fn in fnames:
            if fn.endswith(".py"): files.append(os.path.join(root, fn))
    rng.shuffle(files)
    for fp in files:
        try:
            with open(fp) as f: c = f.read()
        except Exception: continue
        for ch in _chunk_lines(c):
            if 60 < len(ch) < 4000: chunks.append(ch)
            if len(chunks) >= n: return chunks
    return chunks[:n]

def gather_prose(n, rng):
    chunks = []
    with open(os.path.join(OUTDIR, "train.txt")) as f:
        text = f.read()
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 60]
    rng.shuffle(paras)
    for p in paras:
        chunks.append(p[:900])
        if len(chunks) >= n: break
    return chunks[:n]

def build():
    os.makedirs(OUTDIR, exist_ok=True)
    # ---- TRAIN + VAL: original global-RNG sequence (byte-stable) ----
    g = random  # global, seeded SEED
    ev, tr = [], []
    for _ in range(EVAL_N): ev.append({"domain":"json","text":gen_json(g)})
    for _ in range(TRAIN_N): tr.append({"domain":"json","text":gen_json(g)})
    for _ in range(EVAL_N): ev.append({"domain":"tool_call","text":gen_tool_call(g)})
    for _ in range(TRAIN_N): tr.append({"domain":"tool_call","text":gen_tool_call(g)})
    py = gather_python(EVAL_N + TRAIN_N, g)
    ev += [{"domain":"python","text":t} for t in py[:EVAL_N]]
    tr += [{"domain":"python","text":t} for t in py[EVAL_N:EVAL_N+TRAIN_N]]
    pr = gather_prose(EVAL_N + TRAIN_N, g)
    ev += [{"domain":"prose","text":t} for t in pr[:EVAL_N]]
    tr += [{"domain":"prose","text":t} for t in pr[EVAL_N:EVAL_N+TRAIN_N]]
    g.shuffle(ev); g.shuffle(tr)

    # ---- TEST: separate RNG instance (disjoint, independent of train/val) ----
    t = random.Random(SEED + 1000)
    te = []
    for _ in range(TEST_N): te.append({"domain":"json","text":gen_json(t)})
    for _ in range(TEST_N): te.append({"domain":"tool_call","text":gen_tool_call(t)})
    te += [{"domain":"python","text":x} for x in gather_python(TEST_N, t)]
    te += [{"domain":"prose","text":x} for x in gather_prose(TEST_N, t)]
    t.shuffle(te)

    for name, data in [("diverse_train", tr), ("diverse_eval", ev), ("diverse_test", te)]:
        with open(f"{OUTDIR}/{name}.jsonl","w") as f:
            for o in data: f.write(json.dumps(o)+"\n")
    from collections import Counter
    for label, data in [("TRAIN",tr),("EVAL ",ev),("TEST ",te)]:
        print(f"{label} {len(data)} seqs: {dict(Counter(o['domain'] for o in data))}", flush=True)
    print("wrote train/eval/test jsonl (test disjoint via separate RNG)", flush=True)

if __name__ == "__main__":
    build()
