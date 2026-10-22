"""Quick MTP-acceptance + decode-speed probe for a vLLM endpoint.
Records the spec-decode draft/accepted counters, fires N fixed generate
requests, records the counters again, and reports the acceptance delta +
mean tokens/s. Run against baseline and finetuned servers for a fair A/B.
"""
import sys, time, json, urllib.request

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19622/v1"
MODEL    = sys.argv[2] if len(sys.argv) > 2 else "Swift-Qwen-27B-TP2CP2"
N        = int(sys.argv[3]) if len(sys.argv) > 3 else 10
MAXTOK   = int(sys.argv[4]) if len(sys.argv) > 4 else 64

def get_metrics():
    with urllib.request.urlopen(ENDPOINT.replace("/v1","") + "/metrics", timeout=10) as r:
        txt = r.read().decode()
    draft = accept = 0.0
    for line in txt.splitlines():
        if line.startswith("vllm:spec_decode_num_draft_tokens_total{"):
            draft = float(line.rsplit(" ",1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total{"):
            accept = float(line.rsplit(" ",1)[1])
    return draft, accept

# fixed diverse prompts (json / code / prose / tool-ish)
PROMPTS = [
    "Write a JSON object describing a book with title, author, and year.",
    "def fibonacci(n): return the nth fibonacci number, iteratively.",
    "Explain in one sentence why the sky is blue.",
    "Call the function get_weather(city='Paris') and return the result as JSON.",
    "Write a SQL query to select the top 10 users by spend.",
    "Describe the plot of a short story in three sentences.",
    "Write a Python function to reverse a linked list.",
    "What is the capital of France? Answer in one word.",
    "Generate a CSV row for name,age,city with sample data.",
    "Write a regex to validate an email address.",
]

d0, a0 = get_metrics()
print(f"[start] draft={d0:.0f} accept={a0:.0f}", flush=True)

total_tokens = 0
t_start = time.time()
for i in range(N):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role":"user","content":PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": MAXTOK, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(ENDPOINT + "/chat/completions", data=body,
                                headers={"Content-Type":"application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read().decode())
    dt = time.time() - t0
    usage = out.get("usage", {})
    nt = usage.get("completion_tokens", 0)
    total_tokens += nt
    print(f"  req {i+1}/{N}: {nt} tokens in {dt:.2f}s ({nt/dt:.1f} tok/s)", flush=True)
wall = time.time() - t_start

d1, a1 = get_metrics()
dd = d1 - d0; da = a1 - a0
acc = da/dd if dd > 0 else 0.0
print(f"\n[done] {N} requests, {total_tokens} completion tokens in {wall:.1f}s "
      f"({total_tokens/wall:.1f} tok/s aggregate)", flush=True)
print(f"[MTP] draft delta={dd:.0f} accept delta={da:.0f} -> ACCEPTANCE={acc:.4f} "
      f"({da:.0f}/{dd:.0f})", flush=True)
print(f"PROBE_OK acceptance={acc:.4f} tok_s={total_tokens/wall:.2f}", flush=True)
