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
