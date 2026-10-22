"""Swift-27B MTP-head-only model for verl FSDP SFT (remote-code class).

Wraps the frozen Qwen3_5 backbone and exposes the MTP head's logits as the model
output, so verl's `sft_loss` (CE on the next token) trains ONLY the MTP head.

MTP head semantics (matches mtp_work/mtp_head.py): given the backbone post-norm
hidden state h_i and the token t_{i+1}, it emits logits for t_{i+2}. To align with
verl's convention (logits[i] predicts input_ids[i+1]), we feed the head h_{i-1}
and t_i, so its output at position i predicts t_{i+1}.

The backbone is frozen (requires_grad=False); only mtp_head.* params train.
Registered via config.auto_map + trust_remote_code (no verl core changes).
"""
import copy

import torch
import torch.nn as nn

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5RMSNorm,
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5CausalLMOutputWithPast,
)


class MTPHead(nn.Module):
    """Trainable MTP head (1 full-attention layer), shares embed_tokens + lm_head."""

    def __init__(self, text_config, embed_tokens, lm_head):
        super().__init__()
        self.hidden_size = text_config.hidden_size
        eps = getattr(text_config, "rms_norm_eps", 1e-6)
        mtp_cfg = copy.deepcopy(text_config)
        mtp_cfg.layer_types = ["full_attention"]
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(self.hidden_size, eps=eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(self.hidden_size, eps=eps)
        self.fc = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.layer = Qwen3_5DecoderLayer(mtp_cfg, layer_idx=0)
        self.norm = Qwen3_5RMSNorm(self.hidden_size, eps=eps)
        self.rotary = Qwen3_5TextRotaryEmbedding(text_config)
        # shared, NOT submodules (skipped by .to()/state_dict(); moved on the fly)
        object.__setattr__(self, "embed_tokens", embed_tokens)
        object.__setattr__(self, "lm_head", lm_head)

    def forward(self, h, token_ids):
        """h: [N,H] backbone hidden states; token_ids: [N] tokens. -> [N,V] logits."""
        cdev = next(self.fc.parameters()).device
        e = self.embed_tokens(token_ids.to(self.embed_tokens.weight.device)).to(cdev)
        e_n = self.pre_fc_norm_embedding(e)
        h_n = self.pre_fc_norm_hidden(h.to(cdev))
        x = self.fc(torch.cat([e_n, h_n], dim=-1))
        x = x.unsqueeze(1)  # [N,1,H]
        positions = torch.zeros(x.shape[0], 1, dtype=torch.long, device=cdev)
        cos, sin = self.rotary(x, positions)
        out = self.layer(
            hidden_states=x, position_embeddings=(cos, sin),
            position_ids=positions, attention_mask=None,
        )
        out = self.norm(out)
        logits = self.lm_head(out.to(self.lm_head.weight.device))
        return logits.squeeze(1).to(cdev)  # [N,V]


class Qwen3_5MTPHead(Qwen3_5ForConditionalGeneration):
    """Backbone frozen; MTP head trainable. Forward returns MTP-head logits."""

    def __init__(self, config):
        super().__init__(config)
        tc = config.text_config
        self.mtp_head = MTPHead(tc, self.model.language_model.embed_tokens, self.lm_head)
        self._freeze_backbone()

    def _freeze_backbone(self):
        n_train = 0
        for name, p in self.named_parameters():
            if name.startswith("mtp_head."):
                p.requires_grad = True
                n_train += p.numel()
            else:
                p.requires_grad = False
        self._mtp_trainable_params = n_train

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                pixel_values=None, pixel_values_videos=None, image_grid_thw=None,
                video_grid_thw=None, mm_token_type_ids=None, logits_to_keep=0, **kwargs):
        # backbone hidden states (frozen)
        outputs = self.model(
            input_ids=input_ids, pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw, position_ids=position_ids,
            attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, mm_token_type_ids=mm_token_type_ids, **kwargs,
        )
        hidden = outputs[0]  # [B, L, H]
        B, L, H = hidden.shape

        # MTP head at position i uses h_{i-1} + t_i -> predicts t_{i+1}
        # h_{i-1}: hidden shifted right by 1 (position 0 uses h_0 as a no-op pad)
        h_in = torch.roll(hidden, shifts=1, dims=1)      # [B,L,H], h_in[:,i]=h_{i-1}
        h_in[:, 0, :] = hidden[:, 0, :]                  # pad position 0 with h_0
        t_in = input_ids                                  # [B,L], t_in[:,i]=t_i

        h_flat = h_in.reshape(-1, H)
        t_flat = t_in.reshape(-1)
        mtp_logits = self.mtp_head(h_flat, t_flat)        # [B*L, V], predicts t_{i+1}
        logits = mtp_logits.view(B, L, -1)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels,
                                      vocab_size=tc.vocab_size)
        return Qwen3_5CausalLMOutputWithPast(loss=loss, logits=logits)


# HuggingFace remote-code registration
from transformers.models.auto.configuration_auto import CONFIG_MAPPING as _  # noqa
MODEL_FOR_CAUSAL_LM = Qwen3_5MTPHead
