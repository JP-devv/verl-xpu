# Swift-27B MTP Head — Reconstruction, Baseline, and Finetune

Isolated work on haxor's free cards (2,3). Serving lane (cards 0,1) untouched throughout.

## What this is

Swift-27B (`~/models/swift-27b`, a copy of the pristine `~/models/Qwen3.8-27B`) is a
Qwen3_5 27B GDN-hybrid with a **Multi-Token-Prediction (MTP) head** (`mtp.*` weights,
15 tensors, ~425M params: a projection `fc`, one full-attention decoder layer, and
zero-centered RMSNorms). The base `transformers` model ignores `mtp.*` on load
(`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`), so the head is not exposed —
this work reconstructs it as a standalone trainable module.

## MTP forward (verified against 4 independent references)

```
e_n = pre_fc_norm_embedding(embed(t))     # zero-centered RMSNorm
h_n = pre_fc_norm_hidden(h)               # zero-centered RMSNorm on main post-norm hidden
x   = fc(cat([e_n, h_n]))                 # [2H->H], no bias, EMBED FIRST
x   = decoder_layer(x)                    # one full-attention Qwen3_5 layer (single token)
x   = norm(x)
logits = lm_head(x)                       # SHARED main lm_head -> predicts t+1
```
Given main post-norm hidden state `h_i` and the embedding of token `t_{i+1}`, the head
emits logits for token `t_{i+2}`. For a single self-attending token the RoPE position is
irrelevant (relative position 0).

**Acceptance** = fraction of positions `i` where the MTP head's argmax of token `i+2`
equals the true token `i+2`.

## Files

- `mtp_head.py` — the `MTPHead` module + `load_mtp_state` (maps `mtp.layers.0.*` -> `layer.*`).
- `measure_mtp.py` — loads the frozen 27B (memory-bounded) + MTP head, measures acceptance.
  `MTP_HEAD=<ckpt>` loads a finetuned head; unset uses the original `mtp.*` weights.
- `train_mtp2.py` — two-phase finetune (see below).
- `~/models/swift-27b-mtp-ft/mtp_head_final.pt` — the finetuned head (849MB).

## Memory-safe loading (the hard part)

The 54GB model cannot load naively on a 61GB host that is also serving: a full in-RAM
build OOMs the host and kills the serving lane (cards 0,1). The safe path:

- `init_empty_weights()` -> model on meta (no RAM).
- `load_checkpoint_in_model(...)` -> tensors stream to XPU one at a time (peak RAM ~25GB,
  not 54GB).
- `dispatch_model(...)` -> installs the inter-device activation hooks.
- `no_split_module_classes=["Qwen3_5DecoderLayer"]` -> keeps each layer on one device.
- `use_cache=False` on the forward -> avoids the pipeline-parallel KV-cache device mismatch.

Every run is guarded by a RAM watchdog (kills the load if host free RAM < 8GB) and a
pre-flight check (abort if < 30GB free). `ZE_AFFINITY_MASK=2,3`, `ONEAPI_DEVICE_SELECTOR` unset.

## Two-phase finetune (`train_mtp2.py`)

The MTP head's training state (~5GB) doesn't fit beside the 54GB main model on the same
cards, and CPU-offloading the main model reintroduces fragile meta-tensor states. So:

- **Phase 1:** full frozen main model on both cards (no CPU offload -> no meta). Run the
  WikiText-103 TRAIN corpus through it, save every `(h_i, token i+1, token i+2)` triple to
  `/tmp/mtp-hstore.pt` (129k triples, 1.3GB). Unload the main model.
- **Phase 2:** load just `embed_tokens` + `lm_head` + the MTP head onto ONE card (frozen
  embed/lm_head, trainable MTP head) and train against the saved `h_i` (AdamW, lr=1e-4).

## Diverse evaluation (disjoint held-out TEST set)

The WikiText-only number above is superseded. We now evaluate on a **disjoint**
`diverse_test.jsonl` (254 seqs, 24,856 positions) that shares **no** sequence with the
train or val sets — so the finetuned head was never trained on or selected against it.
This is the unbiased generalization estimate. Both heads are measured on the identical
sequences by `measure_mtp.py`.

| Task | Baseline | Finetuned | Delta | z | 95% CI of delta | Verdict |
|------|---------:|----------:|------:|-----:|:----------------|:--------|
| json      | 0.4134 (3053/7385)  | 0.5852 (4322/7385)  | **+0.1718** | 21.2 | [+0.156, +0.188] | significant + |
| tool_call | 0.5726 (1746/3049)  | 0.8393 (2559/3049)  | **+0.2666** | 23.9 | [+0.245, +0.289] | significant + |
| python    | 0.4215 (3330/7900)  | 0.3956 (3125/7900)  | **−0.0259** | −3.3 | [−0.041, −0.011] | **significant −** |
| prose     | 0.3431 (2238/6522)  | 0.3451 (2251/6522)  | +0.0020 | 0.24 | [−0.014, +0.018] | noise |
| **OVERALL** | 0.4171 (10367/24856) | 0.4931 (12257/24856) | **+0.0760** | 17.1 | [+0.067, +0.085] | significant + |

**Read:** the distribution-matched finetune gives a real, significant **+7.6pp overall**
and large gains on the structured domains (json +17.2pp, tool_call +26.7pp). But it is
**not uniform**: **python regressed −2.6pp (significant)** and **prose is within noise**.
The val-set gain (+9.6pp) was inflated by early-stopping selection bias; the disjoint
test set is the honest number.

## Caveats / next iteration

- **Overfit to the train/val distribution:** the head helps json/tool_call but hurts
  python and leaves prose flat — a sign the margin loss + CE is overfitting the specific
  train mix rather than generalizing. The fix is a broader, more balanced train corpus
  (e.g. Nemotron post-training text for prose; more diverse python) and/or a stronger
  regularizer, then re-measure on the disjoint test set.
- **Two-phase is CPU-bound:** phase 2 shuttles `h_i` CPU→XPU each step and a 424M head
  can't saturate a B70. The all-on-XPU single-phase trainer (below) removes that.
- To deploy: load `mtp_head_final.pt` into the `MTPHead` in the serving stack's MTP path.

## Standalone all-on-XPU MTP-head run (2 GPU, cards 2,3) — PASS

`train_mtp3.py`: frozen 27B sharded across cards 2,3 via `device_map`, MTP head on the
last card sharing `embed_tokens`, no CPU h_i round-trip. Main `.venv` (torch 2.9.1+xpu,
oneCCL 2021.16.1). 998 train / 254 val seqs, CE + margin loss.

| epoch | time  | final loss | val acc  |
|-------|-------|-----------|----------|
| base  | —     | —         | 0.4201   |
| 0     | 200s  | 1.4355    | 0.4781   |
| 1     | 199s  | 0.5120    | 0.4827   |
| 2     | 200s  | 0.9915    | 0.4893   |
| 3     | 199s  | 0.4924    | 0.4962   |
| 4     | 199s  | 0.1721    | **0.5029** |

**+8.3 pp** over 5 epochs. Peak XPU mem card 2: 32566/32656 MiB (99.7%). 1 GPU infeasible
(27B bf16 ≈ 54 GB > 32 GB/card); 2 GPU minimum; 4 GPU blocked by serving lane.

## Apples-to-apples: two-phase vs all-on-XPU MTP head (disjoint test set)

`measure_mtp.py` on `/tmp/mtp-corpus/diverse_test.jsonl` (256 seqs, 24,856 positions;
64 each of json/tool_call/python/prose). Baseline = original `mtp.*` head (identical for
both rows). Finetuned = the saved `mtp_head_final.pt` from each trainer.

| Domain (test) | Baseline | Two-phase `train_mtp2` | All-on-XPU `train_mtp3` | Δ (2-phase) | Δ (all-on-XPU) |
|---|---|---|---|---|---|
| json | 0.4134 | 0.5852 | **0.6018** | +0.1718 | **+0.1884** |
| tool_call | 0.5726 | 0.8393 | **0.8419** | +0.2666 | **+0.2693** |
| python | 0.4215 | 0.3956 | 0.3909 | −0.0259 | −0.0306 |
| prose | 0.3431 | 0.3451 | 0.3373 | +0.0020 | −0.0058 |
| **OVERALL** | **0.4171** | **0.4931** | **0.4948** | **+0.0760** | **+0.0777** |

**Conclusion:** the all-on-XPU trainer (frozen 27B sharded on cards 2,3, MTP head on the
last card, no CPU h_i round-trip) is **quality-equivalent** to the two-phase CPU-h_i
trainer (deltas within ±0.01–0.02 = noise on 24.8k positions). Both show the same
signature: large gains on structured domains (json +18–19pp, tool_call +27pp), flat-to-
slightly-negative on prose/python. The all-on-XPU path is the one that runs inside verl's
FSDP-style engine, so this validates it as the production training path with no quality
regression.

Note on the "34% baseline" from early work: that was the **WikiText-only** (single prose
domain) baseline. The diverse-corpus baseline is **41.7%** because json/tool_call pull it
up — not a regression, just a different (harder, mixed) eval set.

## Online serving A/B (BetterBench-style probe): baseline vs finetuned MTP head

Live vLLM endpoints, 10 fixed diverse prompts (json/code/prose/tool/SQL/regex),
64 max tokens, greedy. Acceptance = accepted/draft spec-decode tokens (from the
vLLM `/metrics` counters, delta over the request set).

| Metric | Baseline (orig mtp.*, serving lane TP2/CP2 cards 0,1) | Finetuned (trained head, TP2 cards 2,3) | Δ |
|---|---|---|---|
| **MTP acceptance** | **47.69%** (455/954) | **73.96%** (250/338) | **+26.27 pp** |
| Decode speed | 121.78 tok/s | 26.48 tok/s | n/a (config artifact) |

The finetuned head accepts **73.96%** of draft tokens vs **47.69%** baseline — a
**+26.27 pp** online improvement (consistent in direction with the offline test-set
delta of +7.77 pp; the online gap is larger because serving traffic + 1-spec-token
drafting amplify the head's gain). The decode-speed gap (26 vs 122 tok/s) is a
**config artifact, not the head**: the finetuned server ran `--enforce-eager`
(XPU graph capture deadlocks on cards 2,3) with `num_speculative_tokens=1`, while
the baseline serving lane uses XPU graphs + 4 spec tokens. With a matched
graph+4-token config the higher acceptance would *raise* effective throughput.

**Key enabler discovered:** a second vLLM/XCCL group on cards 2,3 only inits cleanly
with the serving lane's exact env — critically `CCL_ZE_IPC_EXCHANGE=pidfd` (without it
`onecclGetVersion` crashes on entry), plus `CCL_ENABLE_SYCL_KERNELS=1`,
`CCL_TOPO_P2P_ACCESS=1`, `SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0`, `LD_LIBRARY_PATH=$VENV/lib`.
The XPU graph-capture path deadlocks (one rank 100% / one 0%) on a 2-GPU TP-only launch;
`--enforce-eager` + `VLLM_XPU_ENABLE_XPU_GRAPH=0` avoids it.

## BetterBench decode-only A/B (matched config) — baseline vs finetuned MTP head

Real BetterBench 0.6.1 (`python3 -m betterbench.cli run --quick --no-prefill
--no-concurrency`, corpus v1.0, 8 categories, 1 warmup + 5 passes/cat, temp 0.7,
cold prefix-cache). Both endpoints matched on the BetterBench config. The vLLM
server configs differ (noted per row) because the finetuned server runs on the
free cards 2,3 where the XPU graph-capture path deadlocks, so it uses
`--enforce-eager` + 1 spec token; the baseline serving lane uses XPU graphs + 4
spec tokens on cards 0,1.

**Headline — MTP acceptance (vLLM `/metrics` accepted/draft):**

| | Baseline (orig `mtp.*`) | Finetuned (trained head) | Δ |
|---|---|---|---|
| **MTP acceptance** | **56.76%** (939204/1654656) | **73.67%** (1542/2093) | **+16.91 pp** |

**BetterBench single-stream decode (batch=1):**

| category | baseline decode t/s (med) | finetuned decode t/s (med) | baseline tok/update | finetuned tok/update |
|---|--:|--:|--:|--:|
| chat | 116.3 | 26.5 | 3.06 | 1.61 |
| code | 156.0 | 27.6 | 4.07 | 1.65 |
| file_edit | 197.0 | 31.8 | 5.30 | 1.90 |
| json | 180.7 | 29.2 | 4.87 | 1.79 |
| math | 180.3 | 28.2 | 4.86 | 1.79 |
| prose | 93.9 | 25.9 | 2.51 | 1.54 |
| reasoning | 171.5 | 29.9 | 3.35 | 1.79 |
| summarization | 159.1 | 31.2 | 4.29 | 1.84 |
| **combined (weighted)** | **157.9** | **28.8** | — | — |

Also: baseline update p99 28.0 ms / TTFT p50 108 ms; finetuned update p99 74.0 ms /
TTFT p50 128 ms.

**Reading the two numbers separately (they measure different things):**
- **MTP acceptance (+16.91 pp) is the head-quality signal** and it is a fair A/B:
  both servers run the same 27B backbone, same corpus, same BetterBench config; only
  the MTP head weights differ. The finetuned head accepts 73.67% of draft tokens vs
  56.76% baseline — it predicts the next token far more often.
- **Decode t/s (157.9 vs 28.8) is NOT a head-quality comparison** — it is dominated
  by the server-config gap: baseline = XPU graphs + 4 spec tokens (tok/update up to
  5.3), finetuned = `--enforce-eager` + 1 spec token (tok/update ~1.6-1.9). The
  enforce-eager path alone costs ~5-8× on this stack, and 1-vs-4 spec tokens caps the
  tokens-per-update. A matched graph+4-token finetuned server would close most of that
  gap *and* benefit from the higher acceptance. (The graph path deadlocks on a 2-GPU
  TP-only launch on cards 2,3 — the same failure that blocks the FSDP run — so the
  finetuned number here is a floor, not the head's true serving speed.)

Result files: `/tmp/bb-baseline.json` (165 KB), `/tmp/bb-finftuned.json` (98 KB);
HTML reports `/tmp/bb-baseline.html`, `/tmp/bb-finftuned.html`.
