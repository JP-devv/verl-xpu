# SPDX-License-Identifier: Apache-2.0
"""Small CPU checks for the branch adaptation; never full-model CPU training."""
import os
from pathlib import Path
import tempfile

os.environ['HERMES_HELPERS'] = str(Path(__file__).resolve().parents[3] / 'scripts')
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from train_hermes_capture import MTPHead, FrozenProjection, terms, native_key, export_native, load_store, key, tensor_sha, Qwen3_5TextRotaryEmbedding


def main():
    torch.manual_seed(7)
    cfg = Qwen3_5TextConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                           layer_types=['full_attention'], vocab_size=137,
                           rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': .25})
    cfg._attn_implementation = 'sdpa'
    embed = nn.Embedding(137, 64)
    embed.weight.requires_grad_(False)
    projection = FrozenProjection(torch.randn(137, 64).half() * .02)
    head = MTPHead(cfg, embed, projection)
    embed.bfloat16()
    head.bfloat16()
    before = {n: p.detach().clone() for n, p in head.named_parameters()}
    frozen = embed.weight.detach().clone(), projection.weight.detach().clone()
    h, tokens, labels = torch.randn(9, 64).bfloat16(), torch.arange(9), torch.arange(10, 19)
    assert tensor_sha(head.fc.weight) == tensor_sha(head.fc.weight.detach().clone())
    with torch.autocast('cpu', dtype=torch.bfloat16):
        logits = head(h, tokens)
    loss, ce, penalty = terms(logits, labels)
    z = logits.float()
    direct = torch.nn.functional.cross_entropy(z, labels)
    wrong = z.clone(); wrong.scatter_(1, labels[:, None], -torch.inf)
    direct += .5 * torch.relu(1. - (z.gather(1, labels[:, None]).squeeze(1) - wrong.max(1).values)).mean()
    assert torch.equal(loss, direct)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, foreach=False)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    optimizer.step()
    assert any(not torch.equal(before[n], p) for n, p in head.named_parameters())
    assert torch.equal(embed.weight, frozen[0]) and torch.equal(projection.weight, frozen[1])
    assert embed.weight.grad is None and projection.weight.grad is None
    assert len(before) == 15 and not any(n.startswith(('embed', 'lm_head')) for n in before)
    with torch.device('meta'):
        meta_head = MTPHead(cfg, embed, projection, device='meta')
    meta_head.to(dtype=torch.bfloat16)
    assert all(p.is_meta and p.dtype==torch.bfloat16 for p in meta_head.parameters())
    meta_head.to_empty(device='cpu')
    meta_head.rotary = Qwen3_5TextRotaryEmbedding(cfg, device='cpu')
    meta_head.load_state_dict(head.state_dict(), strict=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert torch.equal(meta_head(h,tokens),head(h,tokens))
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        export_native(head, out / 'head.safetensors')
        with safe_open(str(out / 'head.safetensors'), framework='pt') as f:
            assert set(f.keys()) == {native_key(n) for n in head.state_dict()}
            for n, v in head.state_dict().items():
                assert torch.equal(f.get_tensor(native_key(n)), v.detach().bfloat16())
        torch.save({'model': head.state_dict(), 'optimizer': optimizer.state_dict()}, out / 'state.pt')
        saved = torch.load(out / 'state.pt', weights_only=True, mmap=True)
        head.load_state_dict(saved['model']); optimizer.state.clear(); optimizer.load_state_dict(saved['optimizer'])
        assert optimizer.state
        record = {'sid': 'fixture', 'w': 'turn', 'split': 'train', 'input_ids': list(range(10, 18)),
                  'spans': [[3, 5], [7, 8]], 'n_tgt': 3}
        source_hidden = torch.arange(8).half()[:, None].expand(8, 64).contiguous()
        save_file({'hidden': source_hidden, 'input_ids': torch.tensor(record['input_ids'])},
                  str(out / (key(record) + '.safetensors')))
        hidden, tokens, labels, teacher = load_store([record], out, 64, 'cpu', teacher=True)
        assert tokens.tolist() == [12, 13, 16] and labels.tolist() == [13, 14, 17]
        assert hidden[:, 0].tolist() == [1, 2, 5] and teacher[:, 0].tolist() == [2, 3, 6]
    print({'pass': True, 'branch_head_tensors': len(before), 'loss': float(loss.detach()),
           'ce_margin_matches': True, 'shared_frozen': True, 'native_export_equal': True,
           'adamw_checkpoint_reload': True, 'two_token_shift_and_mask': True,
           'meta_bf16_before_allocation_equal': True,
           'scope': 'tiny CPU numerical/integration check'})


if __name__ == '__main__':
    main()
