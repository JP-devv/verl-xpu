# SPDX-License-Identifier: Apache-2.0
"""Train native Swift MTP for sampler-matched depth-6 rollouts.

This is a separate recipe from ``train_hermes_capture.py``. It starts from the
same native parent tensors but trains the deployed recurrent MTP layer to
maximize filtered target/draft distribution overlap across six positions.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from transformers import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

sys.path.insert(0, os.environ.get('HERMES_HELPERS', '/work/scripts'))
from mtp_head import MTPHead
from hermes_rollout import RolloutFeatureStore, verifier_rollout_loss

from real_trial import TargetTensors
from streamed_projection import StreamedGPTQHead
from hermes_streamed_margin import FeatureOutput
from train_hermes_capture import (BRANCH_SHA, SelectedEmbedding, emit, export_native,
                                  native_key, tensor_sha)


def capture_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--windows', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--target', type=Path, default=Path('/model'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--batch-rollouts', type=int, default=8)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--sampler-source', default='vllm-model-generation-config')
    p.add_argument('--ce-weight', type=float, default=0.05)
    p.add_argument('--eval-every', type=int, default=256)
    p.add_argument('--checkpoint-every', type=int, default=256)
    p.add_argument('--budget-s', type=int, default=3600)
    p.add_argument('--max-updates', type=int, default=0,
                   help='Explicit smoke only; never reports complete data training')
    p.add_argument('--resume', type=Path)
    p.add_argument('--feature-cache-mib', type=int, default=64)
    p.add_argument('--vocabulary-chunk', type=int, default=2048)
    a = p.parse_args()
    assert torch.xpu.is_available(), 'XPU required; no CPU/CUDA training fallback'
    if a.depth != 6:
        raise ValueError('This training recipe is defined for deployed draft depth 6')
    device = 'xpu:0'
    torch.xpu.set_device(0)
    torch.manual_seed(61141)
    a.out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in a.windows.open()]
    train, val = [r for r in records if r['split'] == 'train'], [
        r for r in records if r['split'] == 'heldout'
    ]
    if not train or not val or {r['root_id'] for r in train} & {r['root_id'] for r in val}:
        raise ValueError('Training and heldout rollout roots must be nonempty and disjoint')
    dataset_sha = capture_digest(a.windows)
    target = TargetTensors(a.target)
    cfg = Qwen3_5Config.from_pretrained(a.target).text_config
    cfg._attn_implementation = 'sdpa'
    token_ids = sorted({r['input_ids'][i - 1] for r in records
                        for start, end in r['spans']
                        for i in range(max(2, start), end)})
    embed = SelectedEmbedding(token_ids, cfg.hidden_size, cfg.vocab_size, target, device,
                              dynamic_cache_mib=a.feature_cache_mib)
    projection = StreamedGPTQHead(target, device=device, chunk=a.vocabulary_chunk)
    with torch.device('meta'):
        head = MTPHead(cfg, embed, FeatureOutput(device), device='meta')
    head.to(dtype=torch.bfloat16)
    head.to_empty(device=device)
    head.rotary = Qwen3_5TextRotaryEmbedding(cfg, device=device)
    if sum(p.numel() for p in head.parameters()) != 424699392 or len(list(head.parameters())) != 15:
        raise ValueError('Unexpected native MTP parameter set')
    with safe_open(str(a.parent), framework='pt', device='cpu') as f:
        expected = {native_key(name) for name in head.state_dict()}
        if set(f.keys()) != expected:
            raise ValueError('Parent native tensor mapping mismatch')
        with torch.no_grad():
            for name, value in head.state_dict().items():
                parent_value = f.get_tensor(native_key(name))
                if parent_value.shape != value.shape:
                    raise ValueError(f'Parent tensor shape mismatch: {name}')
                value.copy_(parent_value)
    before = {name: tensor_sha(value) for name, value in head.named_parameters()}
    train_store = RolloutFeatureStore(train, a.captures, cfg.hidden_size, depth=a.depth)
    val_store = RolloutFeatureStore(val, a.captures, cfg.hidden_size, depth=a.depth)
    if not len(train_store) or not len(val_store):
        raise ValueError('Captures do not contain complete depth-6 verifier rollouts')
    emit('rollout_data_ready', depth=a.depth, train_rollouts=len(train_store),
         heldout_rollouts=len(val_store), train_selected_labels=train_store.selected_labels,
         train_covered_labels=train_store.covered_labels,
         train_uncovered_labels=train_store.uncovered_labels,
         heldout_covered_labels=val_store.covered_labels,
         feature_cache_limit_bytes=a.feature_cache_mib * 1024**2,
         dataset_sha256=dataset_sha,
         objective='depth6 filtered rejection-sampling overlap / expected accepted prefix',
         sampler={'source': a.sampler_source, 'temperature': a.temperature,
                  'top_k': a.top_k, 'top_p': a.top_p})
    optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, foreach=False)
    evaluations, best, step, cursor = [], None, 0, 0
    order = torch.randperm(len(train_store), generator=torch.Generator().manual_seed(61141)).tolist()

    @torch.no_grad()
    def evaluate(tag):
        head.eval()
        total_rollouts = 0
        nll_sum = expected_sum = accepted_sum = 0.
        depth_match = torch.zeros(a.depth)
        prefix_match = torch.zeros(a.depth)
        depth_overlap = torch.zeros(a.depth)
        for lo in range(0, len(val_store), a.batch_rollouts):
            indices = list(range(lo, min(lo + a.batch_rollouts, len(val_store))))
            initial, first, teacher, positions = val_store.batch(indices, device)
            rollout_tokens = val_store.continuation_batch(indices, device)
            with torch.autocast('xpu', dtype=torch.bfloat16):
                _, metrics = verifier_rollout_loss(
                    head, initial.to(torch.bfloat16), first, teacher,
                    positions, projection, a.ce_weight, rollout_tokens=rollout_tokens,
                    temperature=a.temperature, top_k=a.top_k, top_p=a.top_p
                )
            n = len(indices)
            total_rollouts += n
            nll_sum += float(metrics['nll']) * n
            expected_sum += float(metrics['expected_accepted']) * n
            accepted_sum += float(metrics['hard_accepted_length']) * n
            depth_match += metrics['per_depth_greedy_match'].cpu() * n
            prefix_match += metrics['per_depth_prefix_acceptance'].cpu() * n
            depth_overlap += metrics['per_depth_sampler_overlap'].cpu() * n
        result = {
            'tag': tag, 'step': step, 'rollouts': total_rollouts,
            'verifier_nll': nll_sum / total_rollouts,
            'expected_accepted_length': expected_sum / total_rollouts,
            'hard_greedy_accepted_length': accepted_sum / total_rollouts,
            'per_depth_sampler_overlap': (depth_overlap / total_rollouts).tolist(),
            'per_depth_greedy_match': (depth_match / total_rollouts).tolist(),
            'per_depth_prefix_acceptance': (prefix_match / total_rollouts).tolist(),
            'scope': 'heldout root-disjoint captured contexts; sampler-overlap proxy, not live vLLM acceptance',
        }
        emit('rollout_evaluation', **result)
        head.train()
        return result

    parent_eval = None

    def save_checkpoint():
        path, temp = a.out / 'rollout-latest.pt', a.out / 'rollout-latest.pt.partial'
        torch.save({
            'branch_sha': BRANCH_SHA, 'dataset_sha256': dataset_sha,
            'objective': 'sampler-overlap-depth6-expected-accepted-prefix-v1',
            'sampler': {'source': a.sampler_source, 'temperature': a.temperature,
                        'top_k': a.top_k, 'top_p': a.top_p},
            'depth': a.depth, 'ce_weight': a.ce_weight, 'lr': a.lr,
            'model': head.state_dict(), 'optimizer': optimizer.state_dict(),
            'step': step, 'cursor': cursor, 'order': order, 'best': best,
            'evaluations': evaluations, 'parent_eval': parent_eval,
        }, temp)
        os.replace(temp, path)
        emit('rollout_checkpoint_saved', step=step, rollouts=cursor, bytes=path.stat().st_size)

    if a.resume:
        saved = torch.load(a.resume, map_location='cpu', weights_only=True, mmap=True)
        if (saved['branch_sha'] != BRANCH_SHA or saved['dataset_sha256'] != dataset_sha
                or saved['objective'] != 'sampler-overlap-depth6-expected-accepted-prefix-v1'
                or saved['depth'] != a.depth or saved['ce_weight'] != a.ce_weight
                or saved['lr'] != a.lr or saved['order'] != order
                or saved.get('sampler') != {'source': a.sampler_source, 'temperature': a.temperature,
                                             'top_k': a.top_k, 'top_p': a.top_p}):
            raise ValueError('Resume checkpoint does not match this rollout recipe and dataset')
        head.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        step, cursor = saved['step'], saved['cursor']
        best, evaluations = saved['best'], saved['evaluations']
        parent_eval = saved['parent_eval']
        del saved
        gc.collect()
        torch.xpu.empty_cache()
        emit('rollout_checkpoint_reloaded', step=step, rollouts=cursor)
    else:
        parent_eval = evaluate('untrained_parent')

    started = time.monotonic()
    log = (a.out / 'rollout-updates.jsonl').open('a')
    while cursor < len(order):
        lo, hi = cursor, min(cursor + a.batch_rollouts, len(order))
        indices = order[lo:hi]
        initial, first, teacher, positions = train_store.batch(indices, device)
        rollout_tokens = train_store.continuation_batch(indices, device)
        optimizer.zero_grad(set_to_none=True)
        t0 = time.monotonic()
        with torch.autocast('xpu', dtype=torch.bfloat16):
            loss, metrics = verifier_rollout_loss(
                head, initial.to(torch.bfloat16), first, teacher,
                positions, projection, a.ce_weight, rollout_tokens=rollout_tokens,
                temperature=a.temperature, top_k=a.top_k, top_p=a.top_p
            )
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite verifier-rollout objective')
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):
            raise FloatingPointError('Missing or non-finite MTP gradient')
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        torch.xpu.empty_cache()
        optimizer.step()
        torch.xpu.synchronize(0)
        step += 1
        cursor = hi
        row = {
            'step': step, 'rollouts_seen': cursor, 'rollouts': len(indices),
            'labels_covered': len(indices) * a.depth, 'loss': float(loss.detach()),
            'verifier_nll': float(metrics['nll']),
            'expected_accepted_length': float(metrics['expected_accepted']),
            'hard_greedy_accepted_length': float(metrics['hard_accepted_length']),
            'per_depth_prefix_acceptance': metrics['per_depth_prefix_acceptance'].tolist(),
            'per_depth_sampler_overlap': metrics['per_depth_sampler_overlap'].tolist(),
            'grad_norm': float(grad_norm), 'seconds': time.monotonic() - t0,
        }
        log.write(json.dumps(row) + '\n'); log.flush()
        if step == 1 or step % 10 == 0:
            emit('rollout_update', **row)
        del initial, first, teacher, positions, rollout_tokens, loss
        torch.xpu.empty_cache()
        budget_stop = time.monotonic() - started >= a.budget_s
        smoke_stop = bool(a.max_updates and step >= a.max_updates)
        final = cursor == len(order)
        if step % a.eval_every == 0 or final or budget_stop or smoke_stop:
            result = evaluate('trained')
            evaluations.append(result)
            if best is None or result['expected_accepted_length'] > best['expected_accepted_length']:
                best = result
                export_native(head, a.out / 'mtp-rollout-best-bf16.safetensors')
        if step % a.checkpoint_every == 0 or final or budget_stop or smoke_stop:
            save_checkpoint()
        if budget_stop or smoke_stop:
            break

    changed = [native_key(name) for name, value in head.named_parameters()
               if tensor_sha(value) != before[name]]
    if not changed:
        raise AssertionError('Rollout training changed no native head tensors')
    result = {
        'status': 'complete' if cursor == len(order) else 'budget_incomplete',
        'branch_sha': BRANCH_SHA, 'objective': 'sampler-overlap-depth6-expected-accepted-prefix-v1',
        'sampler': {'source': a.sampler_source, 'temperature': a.temperature,
                    'top_k': a.top_k, 'top_p': a.top_p},
        'depth': a.depth, 'train_rollouts_seen': cursor,
        'train_rollouts_selected': len(train_store),
        'train_labels_covered': cursor * a.depth,
        'train_labels_selected': train_store.covered_labels,
        'uncovered_capture_labels': train_store.uncovered_labels,
        'dataset_sha256': dataset_sha, 'parent': parent_eval, 'best': best,
        'evaluations': evaluations, 'changed_native_parameters': changed,
        'optimizer': 'AdamW', 'lr': a.lr, 'ce_weight': a.ce_weight,
        'batch_rollouts': a.batch_rollouts,
        'elapsed_seconds_including_evaluation_and_checkpoints': time.monotonic() - started,
        'peak_xpu_bytes': torch.xpu.max_memory_allocated(0),
        'limitations': [
            'continuation tokens come from captured text; exact live sampled trajectories are not regenerated',
            'filtered top-k/top-p overlap is a rejection-sampling acceptance proxy; live acceptance is still required',
            'capture is a bounded replay, not a full live vLLM rollout',
            'live vLLM acceptance and decode throughput still require matched serving evaluation',
        ],
    }
    tmp = a.out / 'rollout-result.json.partial'
    tmp.write_text(json.dumps(result, indent=2)); os.replace(tmp, a.out / 'rollout-result.json')
    emit('rollout_complete', **result)


if __name__ == '__main__':
    main()
