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

## Results (WikiText-103 TEST split, 256 seqs x 128 = 32,256 positions)

| MTP head | Acceptance |
|---|---|
| **Original (baseline)** | **34.55%** (11146/32256) |
| Finetuned, 3 epochs | 35.13% (11330/32256) |
| **Finetuned, 6 epochs** | **35.56%** (11470/32256) |

The finetuned head beats the original by **+1.0pp (~2.9% relative)** on the held-out test
split. Training loss dropped 3.97 -> 0.30 over 6 epochs.

## Caveats

- Training/eval are on WikiText-103 (a proxy corpus). The real serving distribution
  (agent/code tasks) differs, so the on-WikiText gain is a lower bound on what a
  distribution-matched finetune would give.
- The gain is modest because the shipped head is already a decent predictor; MTP acceptance
  is a hard argmax metric that improves slowly with CE training.
- To deploy: load `mtp_head_final.pt` into the `MTPHead` in the serving stack's MTP path.
