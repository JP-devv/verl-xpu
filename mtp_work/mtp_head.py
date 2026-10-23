"""Trainable Multi-Token-Prediction (MTP) head for Qwen3_5 / Swift-27B.

Reconstructs the `mtp.*` head that the base transformers model ignores
(`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`). Forward pass matches the
four independent references (vLLM qwen3_next_mtp.py, onnxruntime-genai,
llama.cpp eh_proj, MLX):

    h_n = pre_fc_norm_hidden(h)          # zero-centered RMSNorm on main hidden
    e_n = pre_fc_norm_embedding(embed(t))# zero-centered RMSNorm on token embed
    x   = fc(cat([e_n, h_n]))            # [2H -> H], no bias, EMBED FIRST
    x   = decoder_layer(x)               # one full-attention Qwen3_5 layer
    x   = norm(x)                        # zero-centered RMSNorm
    logits = lm_head(x)                 # SHARED main lm_head -> predicts t+1

Given main post-norm hidden state h_i and the embedding of token t_{i+1}, the
head emits logits for token t_{i+2}. For a single self-attending token the RoPE
position is irrelevant (relative position 0), so we feed position 0.
"""
import copy
import torch
import torch.nn as nn

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm,
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)


class MTPHead(nn.Module):
    def __init__(self, text_config, embed_tokens, lm_head, device="cpu"):
        super().__init__()
        self.hidden_size = text_config.hidden_size
        eps = text_config.rms_norm_eps

        # MTP layer = a single full-attention decoder layer
        mtp_cfg = copy.deepcopy(text_config)
        mtp_cfg.layer_types = ["full_attention"]

        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(self.hidden_size, eps=eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(self.hidden_size, eps=eps)
        self.fc = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.layer = Qwen3_5DecoderLayer(mtp_cfg, layer_idx=0)
        self.norm = Qwen3_5RMSNorm(self.hidden_size, eps=eps)
        self.rotary = Qwen3_5TextRotaryEmbedding(text_config, device=device)

        # Shared with the main model (device-mapped; NOT copied). Stored via
        # object.__setattr__ so they are NOT submodules: .to() and state_dict()
        # skip them, and data is moved to their devices on the fly in forward().
        object.__setattr__(self, "embed_tokens", embed_tokens)
        object.__setattr__(self, "lm_head", lm_head)
        self.embed_dev = next(embed_tokens.parameters()).device
        self.lmhead_dev = next(lm_head.parameters()).device

    def forward_hidden(self, h, token_ids, positions=None, past_key_values=None):
        """Return the recurrent MTP state before the shared vocabulary head.

        With ``past_key_values`` this is the deployed speculative recurrence:
        each accepted draft token reuses the same MTP decoder layer and appends
        one key/value position.  The layer is deliberately still a single layer
        and adds no checkpoint tensors.
        """
        cdev = next(self.fc.parameters()).device  # MTP compute device
        e = self.embed_tokens(token_ids.to(self.embed_dev)).to(cdev)  # [N,H]
        e_n = self.pre_fc_norm_embedding(e)
        h_n = self.pre_fc_norm_hidden(h.to(cdev))
        x = self.fc(torch.cat([e_n, h_n], dim=-1))  # embed first, hidden second
        x = x.unsqueeze(1)  # [N,1,H]
        if positions is None:
            positions = torch.zeros(x.shape[0], 1, dtype=torch.long, device=cdev)
        elif positions.ndim == 1:
            positions = positions.unsqueeze(1).to(cdev)
        cos, sin = self.rotary(x, positions)
        attention_mask = None
        if past_key_values is not None:
            past_len = past_key_values.get_seq_length(0)
            # Every row in a rollout batch has the same six-step cache length.
            # All cached positions are causal predecessors, so an additive zero
            # mask is the exact one-query causal mask for this recurrent step.
            attention_mask = torch.zeros(
                (x.shape[0], 1, 1, past_len + 1), device=cdev, dtype=x.dtype
            )
        out = self.layer(
            hidden_states=x,
            position_embeddings=(cos, sin),
            position_ids=positions,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        return self.norm(out).squeeze(1)

    def forward(self, h, token_ids, positions=None, past_key_values=None):
        """Return vocabulary logits; supports optional recurrent rollout cache."""
        cdev = next(self.fc.parameters()).device
        out = self.forward_hidden(h, token_ids, positions, past_key_values)
        logits = self.lm_head(out.to(self.lmhead_dev))  # [N,V] on lm_head device
        return logits.to(cdev)  # [N, V]

    # ------------------------------------------------------------------
    def load_mtp_state(self, mtp_state: dict) -> dict:
        """Load `mtp.*` checkpoint tensors into this module.

        Maps `mtp.<rest>` -> `<rest>` on this module (with `layers.0.*` -> `layer.*`).
        Returns a report dict.
        """
        remapped = {}
        for k, v in mtp_state.items():
            if not k.startswith("mtp."):
                continue
            rest = k[len("mtp."):]
            if rest.startswith("layers.0."):
                rest = "layer." + rest[len("layers.0."):]
            remapped[rest] = v
        missing, unexpected = [], []
        loaded = 0
        own = self.state_dict()
        for k, v in remapped.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v.to(own[k].dtype))
                loaded += 1
            else:
                unexpected.append(k)
        for k in own:
            if k not in remapped:
                missing.append(k)
        self.load_state_dict(own, strict=False)
        return {"loaded": loaded, "missing": missing, "unexpected": unexpected}
