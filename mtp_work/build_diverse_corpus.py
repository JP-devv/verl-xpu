#!/usr/bin/env python
"""Build a diverse, domain-matched MTP corpus (train + eval) for Swift-27B.

Domains mirror the Swift-27B serving distribution:
  - json        : structured output (nested JSON objects)
  - tool_call   : function-calling (name + JSON arguments)
  - python      : real Python code (local ~/AI project code)
  - prose       : summarizing/general prose (WikiText)

Output (NOT committed - generated data):
  /tmp/mtp-corpus/diverse_eval.jsonl   fixed, seeded (baseline vs finetuned measured on identical seqs)
  /tmp/mtp-corpus/diverse_train.jsonl  larger, for phase-1 h_i harvesting + phase-2 training
Each line: {"domain": <str>, "text": <str>}
"""
import os, sys, json, random

SEED = 20240601
random.seed(SEED)
OUTDIR = "/tmp/mtp-corpus"
EVAL_N = int(os.environ.get("EVAL_N", "64"))     # sequences per domain
TRAIN_N = int(os.environ.get("TRAIN_N", "250")) # sequences per domain

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

def gen_val(depth=0):
    r = random.random()
    if depth >= 3 or r < 0.35:
        c = random.random()
        if c < 0.30: return random.randint(0, 100000)
        if c < 0.50: return round(random.uniform(0, 1000), 3)
        if c < 0.70: return random.choice([True, False, None])
        return random.choice(STRS)
    if r < 0.70:
        return {random.choice(KEYS): gen_val(depth+1) for _ in range(random.randint(1, 4))}
    return [gen_val(depth+1) for _ in range(random.randint(1, 4))]

def gen_json():
    return json.dumps({random.choice(KEYS): gen_val() for _ in range(random.randint(2, 6))}, indent=2)

def gen_tool_call():
    tool = random.choice(list(TOOLS))
    args = {}
    for k in TOOLS[tool]:
        r = random.random()
        if r < 0.45: args[k] = random.choice(STRS)
        elif r < 0.75: args[k] = random.randint(0, 1000)
        else: args[k] = random.choice([True, False, None])
    return json.dumps({"name": tool, "arguments": args}, indent=2)

def _chunk_lines(text, target=512):
    out, buf = [], []
    for ln in text.split("\n"):
        buf.append(ln)
        if sum(len(x)+1 for x in buf) > target:
            out.append("\n".join(buf)); buf = []
    if buf: out.append("\n".join(buf))
    return out

def gather_python(n):
    chunks, files = [], []
    for root, dirs, fnames in os.walk(os.path.expanduser("~/AI")):
        dirs[:] = [d for d in dirs if d not in (".venv","site-packages","node_modules","__pycache__",".git","dist","build")]
        for fn in fnames:
            if fn.endswith(".py"): files.append(os.path.join(root, fn))
    random.shuffle(files)
    for fp in files:
        try:
            with open(fp) as f: c = f.read()
        except Exception: continue
        for ch in _chunk_lines(c):
            if 60 < len(ch) < 4000: chunks.append(ch)
            if len(chunks) >= n: return chunks
    return chunks[:n]

def gather_prose(n):
    chunks = []
    with open(os.path.join(OUTDIR, "train.txt")) as f:
        text = f.read()
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 60]
    random.shuffle(paras)
    for p in paras:
        chunks.append(p[:900])
        if len(chunks) >= n: break
    return chunks[:n]

def build():
    os.makedirs(OUTDIR, exist_ok=True)
    ev, tr = [], []
    # synthetic domains
    for _ in range(EVAL_N): ev.append({"domain":"json","text":gen_json()})
    for _ in range(TRAIN_N): tr.append({"domain":"json","text":gen_json()})
    for _ in range(EVAL_N): ev.append({"domain":"tool_call","text":gen_tool_call()})
    for _ in range(TRAIN_N): tr.append({"domain":"tool_call","text":gen_tool_call()})
    # file-backed domains (draw eval first so it's a fixed subset)
    py = gather_python(EVAL_N + TRAIN_N)
    ev += [{"domain":"python","text":t} for t in py[:EVAL_N]]
    tr += [{"domain":"python","text":t} for t in py[EVAL_N:EVAL_N+TRAIN_N]]
    pr = gather_prose(EVAL_N + TRAIN_N)
    ev += [{"domain":"prose","text":t} for t in pr[:EVAL_N]]
    tr += [{"domain":"prose","text":t} for t in pr[EVAL_N:EVAL_N+TRAIN_N]]
    random.shuffle(ev); random.shuffle(tr)
    with open(f"{OUTDIR}/diverse_eval.jsonl","w") as f:
        for o in ev: f.write(json.dumps(o)+"\n")
    with open(f"{OUTDIR}/diverse_train.jsonl","w") as f:
        for o in tr: f.write(json.dumps(o)+"\n")
    from collections import Counter
    print(f"EVAL  {len(ev)} seqs: {dict(Counter(o['domain'] for o in ev))}", flush=True)
    print(f"TRAIN {len(tr)} seqs: {dict(Counter(o['domain'] for o in tr))}", flush=True)
    print(f"wrote {OUTDIR}/diverse_eval.jsonl + diverse_train.jsonl", flush=True)

if __name__ == "__main__":
    build()
