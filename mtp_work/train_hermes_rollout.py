# SPDX-License-Identifier: Apache-2.0
"""Train native Swift MTP for sampler-matched depth-6 rollouts.

This is a separate recipe from ``train_hermes_capture.py``. It starts from the
same native parent tensors but trains the deployed recurrent MTP layer to
maximize filtered target/draft distribution overlap across six positions.
"""
import argparse
from array import array
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
from transformers.optimization import Adafactor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

sys.path.insert(0, os.environ.get('HERMES_HELPERS', '/work/scripts'))
from mtp_head import MTPHead
from hermes_rollout import RolloutFeatureStore, verifier_rollout_loss
from checkpoint_io import SyncingWriter

from real_trial import TargetTensors
from streamed_projection import StreamedGPTQHead
from hermes_streamed_margin import FeatureOutput
from train_hermes_capture import (BRANCH_SHA, SelectedEmbedding, emit, export_native,
                                  native_key, tensor_sha)


ADAMW_NAME = 'AdamW'
ADAFACTOR_NAME = 'Adafactor-factored-no-momentum'


def validate_optimizer_resume(saved_name, current_name, reset_state):
    """Permit only an explicit AdamW -> factored-Adafactor state reset."""
    saved_name = saved_name or ADAMW_NAME  # pre-metadata rollout checkpoints used AdamW
    if reset_state:
        if saved_name == current_name:
            raise ValueError('Optimizer reset requested without changing optimizers')
        if not (saved_name == ADAMW_NAME and current_name == ADAFACTOR_NAME):
            raise ValueError('Only the validated AdamW-to-Adafactor migration is supported')
        return True
    if saved_name != current_name:
        raise ValueError('Checkpoint optimizer differs; use explicit optimizer reset to migrate')
    return False


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
    p.add_argument('--microbatch-rollouts', type=int, default=0,
                   help=('Forward/backward slice size for the effective batch. '
                         '0 uses the full batch; smaller values accumulate exact mean-loss gradients.'))
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
    p.add_argument('--skip-smoke-eval', action='store_true',
                   help='Skip full heldout sweep for a bounded optimizer smoke only')
    p.add_argument('--resume', type=Path)
    p.add_argument('--feature-cache-mib', type=int, default=64)
    p.add_argument('--vocabulary-chunk', type=int, default=2048)
    p.add_argument('--checkpoint-sync-mib', type=int, default=64,
                   help='Bound dirty page-cache growth while saving to slow storage')
    p.add_argument('--empty-xpu-cache-every', type=int, default=0,
                   help='Release cached XPU allocations every N updates; 0 keeps allocator reuse enabled')
    p.add_argument('--optimizer', choices=('adamw', 'adafactor'), default='adamw')
    p.add_argument('--reset-optimizer-on-resume', action='store_true',
                   help=('Load model/data cursor but start fresh optimizer state; required '
                         'when migrating a validated checkpoint to another optimizer'))
    a = p.parse_args()
    if a.reset_optimizer_on_resume and not a.resume:
        raise ValueError('--reset-optimizer-on-resume requires --resume')
    if a.skip_smoke_eval and not a.max_updates:
        raise ValueError('--skip-smoke-eval requires an explicit --max-updates bound')
    if a.microbatch_rollouts < 0 or (a.microbatch_rollouts and
                                     a.microbatch_rollouts > a.batch_rollouts):
        raise ValueError('--microbatch-rollouts must be zero or no larger than --batch-rollouts')
    if a.vocabulary_chunk <= 0:
        raise ValueError('--vocabulary-chunk must be positive')
    assert torch.xpu.is_available(), 'XPU required; no CPU/CUDA training fallback'
    if a.depth != 6:
        raise ValueError('This training recipe is defined for deployed draft depth 6')
    device = 'xpu:0'
    torch.xpu.set_device(0)
    torch.manual_seed(61141)
    a.out.mkdir(parents=True, exist_ok=True)
    # Keep the corpus in compact 32-bit token arrays. The JSON parser briefly
    # creates one Python-int list at a time, but retaining 4.3M Python integers
    # costs roughly 150 MiB versus 16.4 MiB for uint32 storage.
    records = []
    with a.windows.open() as source:
        for line in source:
            record = json.loads(line)
            ids = record['input_ids']
            if any(token < 0 or token > 0xFFFFFFFF for token in ids):
                raise ValueError('Input token ID does not fit uint32 storage')
            record['input_ids'] = array('I', ids)
            if record['input_ids'].itemsize != 4:
                raise RuntimeError('Native unsigned int is not 32 bits')
            records.append(record)
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
    if a.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, foreach=False)
        optimizer_name = ADAMW_NAME
    else:
        # Factored second moments avoid AdamW's two full-size moment tensors.
        # Fixed-step mode is intentional: no hidden relative-LR schedule.
        optimizer = Adafactor(head.parameters(), lr=a.lr, scale_parameter=False,
                              relative_step=False, warmup_init=False, beta1=None,
                              weight_decay=0.)
        optimizer_name = ADAFACTOR_NAME
    evaluations, best, step, cursor = [], None, 0, 0
    optimizer_reset_at_step = None
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
        checkpoint = {
            'branch_sha': BRANCH_SHA, 'dataset_sha256': dataset_sha,
            'objective': 'sampler-overlap-depth6-expected-accepted-prefix-v1',
            'sampler': {'source': a.sampler_source, 'temperature': a.temperature,
                        'top_k': a.top_k, 'top_p': a.top_p},
            'depth': a.depth, 'ce_weight': a.ce_weight, 'lr': a.lr,
            'batch_rollouts': a.batch_rollouts,
            'microbatch_rollouts': a.microbatch_rollouts or a.batch_rollouts,
            'optimizer_name': optimizer_name,
            'optimizer_reset_at_step': optimizer_reset_at_step,
            'model': head.state_dict(), 'optimizer': optimizer.state_dict(),
            'step': step, 'cursor': cursor, 'order': order, 'best': best,
            'evaluations': evaluations, 'parent_eval': parent_eval,
        }
        started_save = time.monotonic()
        with temp.open('wb', buffering=0) as raw:
            writer = SyncingWriter(raw, a.checkpoint_sync_mib * 1024**2)
            torch.save(checkpoint, writer)
            writer.flush()
            write_bytes, sync_count = writer.total_bytes, writer.sync_count
        os.replace(temp, path)
        del checkpoint
        emit('rollout_checkpoint_saved', step=step, rollouts=cursor,
             bytes=path.stat().st_size, write_bytes=write_bytes,
             sync_count=sync_count, checkpoint_seconds=time.monotonic() - started_save)

    if a.resume:
        saved = torch.load(a.resume, map_location='cpu', weights_only=True, mmap=True)
        saved_optimizer_name = saved.get('optimizer_name', 'AdamW')
        if (saved['branch_sha'] != BRANCH_SHA or saved['dataset_sha256'] != dataset_sha
                or saved['objective'] != 'sampler-overlap-depth6-expected-accepted-prefix-v1'
                or saved['depth'] != a.depth or saved['ce_weight'] != a.ce_weight
                or saved['lr'] != a.lr or saved['order'] != order
                or saved.get('sampler') != {'source': a.sampler_source, 'temperature': a.temperature,
                                             'top_k': a.top_k, 'top_p': a.top_p}):
            raise ValueError('Resume checkpoint does not match this rollout recipe and dataset')
        optimizer_migration = validate_optimizer_resume(
            saved_optimizer_name, optimizer_name, a.reset_optimizer_on_resume)
        head.load_state_dict(saved['model'], strict=True)
        if not optimizer_migration:
            optimizer.load_state_dict(saved['optimizer'])
        step, cursor = saved['step'], saved['cursor']
        optimizer_reset_at_step = saved.get('optimizer_reset_at_step')
        if optimizer_migration:
            optimizer_reset_at_step = step
        best, evaluations = saved['best'], saved['evaluations']
        parent_eval = saved['parent_eval']
        del saved
        gc.collect()
        torch.xpu.empty_cache()
        emit('rollout_checkpoint_reloaded', step=step, rollouts=cursor,
             optimizer=optimizer_name,
             optimizer_state='reset' if optimizer_migration else 'restored')
        if optimizer_migration:
            emit('optimizer_migrated', from_optimizer=saved_optimizer_name,
                 to_optimizer=optimizer_name, at_global_step=step,
                 optimizer_internal_step=0,
                 rationale='factored optimizer state reduces memory; optimizer history explicitly reset')
    else:
        parent_eval = evaluate('untrained_parent')

    started = time.monotonic()
    log = (a.out / 'rollout-updates.jsonl').open('a')
    while cursor < len(order):
        update_t0 = time.monotonic()
        update_cpu_t0 = time.process_time()
        lo, hi = cursor, min(cursor + a.batch_rollouts, len(order))
        indices = order[lo:hi]
        # Load and transfer each effective batch once, then retain only one
        # microbatch's recurrent graph for backward at a time. Losses are
        # means, so weighting each microbatch by its share gives the exact
        # full-batch gradient while bounding saved recurrent activations.
        initial, first, teacher, positions = train_store.batch(indices, device)
        rollout_tokens = train_store.continuation_batch(indices, device)
        data_seconds = time.monotonic() - update_t0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.monotonic()
        microbatch_size = a.microbatch_rollouts or len(indices)
        metric_sums = None
        loss_sum = None
        for micro_lo in range(0, len(indices), microbatch_size):
            micro_hi = min(micro_lo + microbatch_size, len(indices))
            weight = (micro_hi - micro_lo) / len(indices)
            sl = slice(micro_lo, micro_hi)
            initial_xpu = initial[sl].to(dtype=torch.bfloat16)
            first_xpu = first[sl]
            teacher_xpu = teacher[sl]
            positions_xpu = positions[sl]
            rollout_xpu = rollout_tokens[sl]
            with torch.autocast('xpu', dtype=torch.bfloat16):
                micro_loss, micro_metrics = verifier_rollout_loss(
                    head, initial_xpu, first_xpu, teacher_xpu,
                    positions_xpu, projection, a.ce_weight, rollout_tokens=rollout_xpu,
                    temperature=a.temperature, top_k=a.top_k, top_p=a.top_p
                )
            if not torch.isfinite(micro_loss):
                raise FloatingPointError('Non-finite verifier-rollout objective')
            (micro_loss * weight).backward()
            loss_sum = micro_loss.detach() * weight if loss_sum is None else (
                loss_sum + micro_loss.detach() * weight)
            if metric_sums is None:
                metric_sums = {key: value.detach() * weight
                               for key, value in micro_metrics.items()}
            else:
                for key, value in micro_metrics.items():
                    metric_sums[key].add_(value.detach(), alpha=weight)
            del initial_xpu, first_xpu, teacher_xpu, positions_xpu, rollout_xpu
            del micro_loss, micro_metrics
        loss, metrics = loss_sum, metric_sums
        if any(p.grad is None for p in head.parameters()):
            raise FloatingPointError('Missing MTP gradient')
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError('Non-finite aggregate MTP gradient norm')
        optimizer.step()
        step += 1
        cursor = hi
        # Pull scalar and per-depth metrics to CPU in one transfer. Several
        # separate float()/tolist() calls each synchronize the XPU stream.
        packed_metrics = torch.cat((
            torch.stack((loss.detach(), metrics['nll'],
                         metrics['expected_accepted'], metrics['hard_accepted_length'],
                         grad_norm)).reshape(-1),
            metrics['per_depth_prefix_acceptance'].reshape(-1),
            metrics['per_depth_sampler_overlap'].reshape(-1),
        )).cpu().tolist()
        loss_value, nll_value, expected_value, hard_value, grad_norm_value = packed_metrics[:5]
        prefix_values = packed_metrics[5:5 + a.depth]
        overlap_values = packed_metrics[5 + a.depth:5 + 2 * a.depth]
        row = {
            'step': step, 'rollouts_seen': cursor, 'rollouts': len(indices),
            'microbatch_rollouts': microbatch_size,
            'vocabulary_chunk': a.vocabulary_chunk,
            'labels_covered': len(indices) * a.depth, 'loss': loss_value,
            'verifier_nll': nll_value,
            'expected_accepted_length': expected_value,
            'hard_greedy_accepted_length': hard_value,
            'per_depth_prefix_acceptance': prefix_values,
            'per_depth_sampler_overlap': overlap_values,
            'grad_norm': grad_norm_value,
            'data_seconds': data_seconds,
            'compute_seconds': time.monotonic() - t0,
            'cpu_seconds': time.process_time() - update_cpu_t0,
            'seconds': time.monotonic() - update_t0,
        }
        log.write(json.dumps(row) + '\n'); log.flush()
        if step == 1 or step % 10 == 0:
            emit('rollout_update', **row)
        del initial, first, teacher, positions, rollout_tokens, loss
        if a.empty_xpu_cache_every and step % a.empty_xpu_cache_every == 0:
            torch.xpu.empty_cache()
        budget_stop = time.monotonic() - started >= a.budget_s
        smoke_stop = bool(a.max_updates and step >= a.max_updates)
        final = cursor == len(order)
        skip_eval = smoke_stop and a.skip_smoke_eval and not final
        if skip_eval:
            emit('rollout_smoke_evaluation_skipped', step=step, rollouts=cursor,
                 reason='bounded one-update optimizer smoke; full heldout sweep remains pending')
        elif step % a.eval_every == 0 or final or budget_stop or smoke_stop:
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
        'optimizer': optimizer_name, 'lr': a.lr, 'ce_weight': a.ce_weight,
        'microbatch_rollouts': a.microbatch_rollouts or a.batch_rollouts,
        'optimizer_reset_at_global_step': optimizer_reset_at_step,
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
