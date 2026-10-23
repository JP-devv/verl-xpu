# SPDX-License-Identifier: Apache-2.0
"""Adapt train_mtp2/train_mtp3 to pre-captured Hermes assistant-label features.

Local adaptation of JP-devv/verl-xpu 31dc9fa80374cfaff7fe53f396f11147d620b22f.
Uses this checkout's unchanged MTPHead, AdamW and CE + margin objective. Changes:
read our causal captures/masks instead of 128-token text; cache frozen GPTQ
projection on compute XPU and stream selected features through a bounded host cache;
loading, native export, checkpoint/resume, timing and immutable-target checks.
The branch's independent one-token MTP attention semantics are preserved.
"""
import argparse
import gc
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from mtp_head import MTPHead
from hermes_feature_store import BoundedFeatureStore, SCHEDULE
from hermes_streamed_margin import FeatureOutput, streamed_terms
from check_hermes_streamed_margin import run_reference as check_streamed_margin

sys.path.insert(0, os.environ.get('HERMES_HELPERS', '/work/scripts'))
from real_trial import TargetTensors
from streamed_projection import StreamedGPTQHead
from check_projection_cache import run_reference as check_cache

BRANCH_SHA = '31dc9fa80374cfaff7fe53f396f11147d620b22f'


def emit(event, **fields):
    print(json.dumps({'event': event, 'time': time.time(), **fields}), flush=True)


def tensor_sha(value):
    # NumPy has no native BF16 scalar dtype. Hash the original bytes without
    # converting precision, staging only a bounded byte chunk on the host.
    flat = value.detach().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for start in range(0, flat.numel(), 2 * 1024 * 1024):
        digest.update(flat[start:start + 2 * 1024 * 1024].cpu().numpy().tobytes())
    return digest.hexdigest()


def key(record):
    return hashlib.sha1(('%s:%s' % (record['sid'], record['w'])).encode()).hexdigest()[:20]


def native_key(name):
    return 'mtp.' + ('layers.0.' + name[6:] if name.startswith('layer.') else name)


def export_native(head, path):
    import struct
    tensors = {native_key(k): v for k, v in head.state_dict().items()}
    header, offset = {}, 0
    for name, value in tensors.items():
        size = value.numel() * 2
        header[name] = {'dtype': 'BF16', 'shape': list(value.shape), 'data_offsets': [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    temp = Path(str(path) + '.partial')
    with temp.open('wb') as f:
        f.write(struct.pack('<Q', len(raw))); f.write(raw)
        for value in tensors.values():
            flat = value.detach().flatten()
            for lo in range(0, flat.numel(), 1024 * 1024):
                chunk = flat[lo:lo + 1024 * 1024].to(device='cpu', dtype=torch.bfloat16)
                f.write(chunk.view(torch.uint16).numpy().tobytes())
    os.replace(temp, path)


class SelectedEmbedding(nn.Module):
    def __init__(self, ids, hidden, vocab, target, device, dynamic_cache_mib=64):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(len(ids), hidden, device=device, dtype=torch.float16), requires_grad=False)
        self.register_buffer('lookup', torch.full((vocab,), -1, device=device, dtype=torch.long))
        self.lookup[torch.tensor(ids, device=device)] = torch.arange(len(ids), device=device)
        name = 'model.language_model.embed_tokens.weight'
        self._target, self._embed_key = target, name
        self._dynamic_cache_limit = dynamic_cache_mib * 1024**2
        self._dynamic_rows = OrderedDict()
        self._dynamic_bytes = 0
        source = target.handle(name).get_slice(name)
        with torch.no_grad():
            for i, token in enumerate(ids):
                self.weight[i].copy_(source[token:token + 1].squeeze(0))

    def forward(self, ids):
        rows = self.lookup[ids]
        result = torch.empty((*ids.shape, self.weight.shape[1]), device=self.weight.device,
                             dtype=torch.bfloat16)
        known = rows >= 0
        if bool(known.any()):
            result[known] = F.embedding(rows[known], self.weight).bfloat16()
        if bool((~known).any()):
            missing_ids = ids[~known].detach().cpu().tolist()
            unique_missing = list(dict.fromkeys(int(token) for token in missing_ids))
            source = self._target.handle(self._embed_key).get_slice(self._embed_key)
            for token in unique_missing:
                if token in self._dynamic_rows:
                    self._dynamic_rows.move_to_end(token)
                    continue
                row = source[token:token + 1].squeeze(0).contiguous()
                row_bytes = row.numel() * row.element_size()
                if row_bytes > self._dynamic_cache_limit:
                    raise ValueError('Dynamic token embedding exceeds its bounded cache')
                while self._dynamic_rows and self._dynamic_bytes + row_bytes > self._dynamic_cache_limit:
                    _, old = self._dynamic_rows.popitem(last=False)
                    self._dynamic_bytes -= old.numel() * old.element_size()
                self._dynamic_rows[token] = row
                self._dynamic_bytes += row_bytes
            dynamic_rows = torch.stack([self._dynamic_rows[token] for token in unique_missing]).to(
                device=self.weight.device, dtype=torch.bfloat16
            )
            dynamic_lookup = {token: index for index, token in enumerate(unique_missing)}
            dynamic_index = torch.tensor([dynamic_lookup[int(token)] for token in missing_ids],
                                        dtype=torch.long, device=ids.device)
            result[~known] = F.embedding(dynamic_index, dynamic_rows)
        return result


class FrozenProjection(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, hidden):
        # Preserve the deployed dequantized FP16 projection, including under
        # the BF16 head's autocast. The cast remains differentiable for the head.
        with torch.autocast(hidden.device.type, enabled=False):
            return F.linear(hidden.half(), self.weight)


def terms(logits, labels, margin=1., coefficient=.5):
    z = logits.float()
    ce = F.cross_entropy(z, labels)
    true = z.gather(1, labels[:, None]).squeeze(1)
    other = z.clone()
    other.scatter_(1, labels[:, None], float('-inf'))
    penalty = torch.relu(margin - (true - other.max(1).values)).mean()
    return ce + coefficient * penalty, ce, penalty


def load_store(records, captures, hidden, device, teacher=False):
    count = sum(r['n_tgt'] for r in records)
    features = torch.empty((count, hidden), device=device, dtype=torch.float16)
    tokens = torch.empty(count, device=device, dtype=torch.long)
    labels = torch.empty_like(tokens)
    teachers = torch.empty_like(features) if teacher else None
    cursor = 0
    for index, r in enumerate(records):
        with safe_open(str(captures / (key(r) + '.safetensors')), framework='pt', device='cpu') as f:
            ids = f.get_tensor('input_ids').long()
            assert ids.tolist() == r['input_ids'], 'Capture token alignment mismatch'
            h = f.get_slice('hidden')
            assert h.get_shape() == [len(ids), hidden]
            before = cursor
            for start, end in r['spans']:
                start = max(2, start)
                if end <= start:
                    continue
                size = end - start
                features[cursor:cursor + size].copy_(h[start - 2:end - 2])
                tokens[cursor:cursor + size].copy_(ids[start - 1:end - 1])
                labels[cursor:cursor + size].copy_(ids[start:end])
                if teacher:
                    teachers[cursor:cursor + size].copy_(h[start - 1:end - 1])
                cursor += size
            assert cursor - before == r['n_tgt']
        if index % 128 == 0 or index + 1 == len(records):
            emit('features_loaded', split=r['split'], windows=index + 1, windows_total=len(records), labels=cursor)
    assert cursor == count
    return features, tokens, labels, teachers


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--windows', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--target', type=Path, default=Path('/model'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--margin', type=float, default=1.)
    p.add_argument('--margin-weight', type=float, default=.5)
    p.add_argument('--token-batch', type=int, default=256)
    p.add_argument('--eval-every', type=int, default=1024)
    p.add_argument('--checkpoint-every', type=int, default=1024)
    p.add_argument('--budget-s', type=int, default=3600)
    p.add_argument('--resume', type=Path)
    p.add_argument('--max-updates', type=int, default=0, help='Explicit smoke only; never reports complete data training')
    p.add_argument('--feature-cache-mib', type=int, default=64)
    p.add_argument('--projection-mode', choices=['streamed','dense'], default='streamed')
    p.add_argument('--vocabulary-chunk', type=int, default=2048)
    a = p.parse_args()
    assert torch.xpu.is_available(), 'XPU required; no CPU/CUDA training fallback'
    device = 'xpu:0'
    torch.xpu.set_device(0)
    torch.manual_seed(61141)
    a.out.mkdir(parents=True, exist_ok=True)
    emit('projection_cache_reference', **check_cache(device))
    emit('streamed_margin_reference', **check_streamed_margin(device))
    dataset_sha = hashlib.sha256(a.windows.read_bytes()).hexdigest()
    records = [json.loads(line) for line in a.windows.open()]
    train = [r for r in records if r['split'] == 'train']
    val = [r for r in records if r['split'] == 'heldout']
    assert train and val and not ({r['root_id'] for r in train} & {r['root_id'] for r in val})
    assert len({key(r) for r in records}) == len(records)
    target = TargetTensors(a.target)
    cfg = Qwen3_5Config.from_pretrained(a.target).text_config
    cfg._attn_implementation = 'sdpa'
    token_ids = sorted({r['input_ids'][i - 1] for r in records for start, end in r['spans'] for i in range(max(2, start), end)})
    emit('initializing', branch_sha=BRANCH_SHA, torch=torch.__version__, train_labels=sum(r['n_tgt'] for r in train),
         heldout_labels=sum(r['n_tgt'] for r in val), token_batch=a.token_batch, head_attention='independent one-token pairs, unchanged branch MTPHead')
    embed = SelectedEmbedding(token_ids, cfg.hidden_size, cfg.vocab_size, target, device)
    projection = StreamedGPTQHead(target, device=device, chunk=a.vocabulary_chunk)
    t0 = time.monotonic()
    if a.projection_mode=='dense':
        cache_bytes = projection.cache_on_device()
        lm_head = FrozenProjection(projection._device_weight)
        emit('projection_cached', bytes=cache_bytes, seconds=time.monotonic() - t0)
    else:
        lm_head = FeatureOutput(device)
        emit('projection_streamed', chunk=a.vocabulary_chunk, dense_cache_bytes=0)
    with torch.device('meta'):
        head = MTPHead(cfg, embed, lm_head, device='meta')
    # Set the dtype on meta tensors before allocating device storage. Allocating
    # FP32 first leaves a needless ~1.70GB temporary/cached head on the device.
    head.to(dtype=torch.bfloat16)
    head.to_empty(device=device)
    head.rotary = Qwen3_5TextRotaryEmbedding(cfg, device=device)
    emit('head_allocated_bf16', allocated_bytes=torch.xpu.memory_allocated(0),
         reserved_bytes=torch.xpu.memory_reserved(0))
    assert sum(p.numel() for p in head.parameters()) == 424699392
    assert len(list(head.parameters())) == 15
    with safe_open(str(a.parent), framework='pt', device='cpu') as f:
        expected = {native_key(name) for name in head.state_dict()}
        assert set(f.keys()) == expected, 'Parent native tensor mapping mismatch'
        with torch.no_grad():
            for name, value in head.state_dict().items():
                parent = f.get_tensor(native_key(name))
                assert parent.shape == value.shape
                value.copy_(parent)
    before = {name: tensor_sha(value) for name, value in head.named_parameters()}
    frozen_versions = (embed.weight._version, lm_head.weight._version)
    frozen_digest = projection.digest()
    emit('head_loaded', trainable_parameters=424699392, native_tensors=15)
    train_store = BoundedFeatureStore(train, a.captures, cfg.hidden_size,
                                     max_cache_bytes=a.feature_cache_mib*1024**2)
    # Validation features remain on the host; only each evaluation batch moves.
    val_store = load_store(val, a.captures, cfg.hidden_size, 'cpu', teacher=True)
    emit('bounded_feature_store_ready', labels=len(train_store),
         host_cache_limit_bytes=train_store.max_cache_bytes, feature_schedule=SCHEDULE,
         corpus_feature_cache_on_xpu=False)
    optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, foreach=False)
    count = len(train_store)
    evaluations, best = [], None
    cursor, step = 0, 0

    @torch.no_grad()
    def evaluate(tag):
        head.eval()
        h, t, y, teacher = val_store
        ce_sum = correct = agree = 0
        for start in range(0, len(t), a.token_batch):
            end = min(len(t), start + a.token_batch)
            with torch.autocast('xpu', dtype=torch.bfloat16):
                logits = head(h[start:end].to(device=device, dtype=torch.bfloat16), t[start:end].to(device))
            labels = y[start:end].to(device)
            if a.projection_mode=='streamed':
                _,ce,_,pred = streamed_terms(logits,labels,projection,a.margin,a.margin_weight)
                truth = projection.topk(teacher[start:end].to(device),1)[1].squeeze(-1)
                ce_sum += float(ce)*(end-start)
            else:
                pred = logits.argmax(-1)
                truth = lm_head(teacher[start:end].to(device)).argmax(-1)
                ce_sum += float(F.cross_entropy(logits.float(), labels, reduction='sum'))
            correct += int((pred == labels).sum()); agree += int((pred == truth).sum())
        result = {'tag': tag, 'step': step, 'tokens': len(t), 'label_ce': ce_sum / len(t),
                  'label_top1': correct / len(t), 'teacher_top1_agreement': agree / len(t),
                  'scope': 'branch one-token teacher-forced proxy; not live speculative acceptance'}
        emit('evaluation', **result)
        head.train()
        return result

    parent = evaluate('untrained_parent')
    torch.xpu.empty_cache()
    reload_checked = False

    def save_checkpoint():
        path = a.out / 'latest.pt'
        temp = a.out / 'latest.pt.partial'
        state = {'branch_sha': BRANCH_SHA, 'dataset_sha256': dataset_sha, 'feature_schedule': SCHEDULE,
                 'projection_mode':a.projection_mode, 'model': head.state_dict(),
                 'optimizer': optimizer.state_dict(), 'optimizer_name': 'AdamW', 'lr': a.lr,
                 'margin': a.margin, 'margin_weight': a.margin_weight, 'cursor': cursor, 'step': step,
                 'supervised_tokens': cursor, 'train_keys': [key(r) for r in train], 'best': best,
                 'evaluations': evaluations, 'parent': parent, 'seed': 61141}
        torch.save(state, temp)
        os.replace(temp, path)
        emit('checkpoint_saved', step=step, labels=cursor, bytes=path.stat().st_size)

    def reload_checkpoint(path):
        nonlocal cursor, step, evaluations, best, parent
        # Map CPU checkpoint storages lazily and release prior optimizer states
        # before rebuilding them, avoiding a second full AdamW state on-device.
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert saved['dataset_sha256'] == dataset_sha and saved['branch_sha'] == BRANCH_SHA
        assert saved['feature_schedule'] == SCHEDULE, 'Feature ordering changed across resume'
        assert saved['projection_mode'] == a.projection_mode, 'Projection backend changed across resume'
        assert saved['train_keys'] == [key(r) for r in train]
        assert (saved['lr'], saved['margin'], saved['margin_weight']) == (a.lr, a.margin, a.margin_weight)
        head.load_state_dict(saved['model'], strict=True)
        optimizer.state.clear()
        torch.xpu.empty_cache()
        optimizer.load_state_dict(saved['optimizer'])
        cursor, step = saved['cursor'], saved['step']
        evaluations, best, parent = saved['evaluations'], saved['best'], saved['parent']
        del saved
        gc.collect()
        emit('checkpoint_reloaded', step=step, labels=cursor)

    if a.resume:
        reload_checkpoint(a.resume)
        reload_checked = True
    started = time.monotonic()
    log = (a.out / 'updates.jsonl').open('a')
    while cursor < count:
        t0 = time.monotonic()
        end = min(count, cursor + a.token_batch)
        h, t, y = train_store.batch(cursor, end)
        hidden = h.to(device=device, dtype=torch.bfloat16)
        tokens, labels = t.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('xpu', dtype=torch.bfloat16):
            logits = head(hidden, tokens)
        if a.projection_mode=='streamed':
            loss,ce,penalty,_ = streamed_terms(logits,labels,projection,a.margin,a.margin_weight)
        else:
            loss, ce, penalty = terms(logits, labels, a.margin, a.margin_weight)
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        assert bool(torch.isfinite(norm))
        # Backward has released its saved activations and vocabulary workspaces.
        # Return those inactive allocations before AdamW creates/updates states.
        torch.xpu.empty_cache()
        optimizer.step()
        torch.xpu.synchronize(0)
        cursor, step = end, step + 1
        seconds = time.monotonic() - t0
        row = {'step': step, 'supervised_tokens': cursor, 'batch_tokens': len(labels), 'loss': float(loss.detach()),
               'ce': float(ce.detach()), 'margin_loss': float(penalty.detach()), 'grad_norm': float(norm),
               'seconds': seconds, 'labels_per_second': len(labels) / seconds}
        log.write(json.dumps(row) + '\n'); log.flush()
        if step == 1 or step % 25 == 0:
            emit('update', **row)
        optimizer.zero_grad(set_to_none=True)
        del logits, loss, ce, penalty, hidden, tokens, labels
        torch.xpu.empty_cache()
        final = cursor == count
        smoke_stop = bool(a.max_updates and step >= a.max_updates)
        budget_stop = time.monotonic() - started > a.budget_s
        if step % a.eval_every == 0 or final or smoke_stop or budget_stop:
            result = evaluate('trained')
            evaluations.append(result)
            if best is None or (result['label_top1'], -result['label_ce']) > (best['label_top1'], -best['label_ce']):
                best = result
                export_native(head, a.out / 'mtp-best-bf16.safetensors')
        if step == 1 or step % a.checkpoint_every == 0 or final or smoke_stop or budget_stop:
            save_checkpoint()
            if not reload_checked:
                checksum = tensor_sha(head.fc.weight)
                reload_checkpoint(a.out / 'latest.pt')
                assert tensor_sha(head.fc.weight) == checksum
                reload_checked = True
        if smoke_stop or budget_stop:
            break
    assert frozen_versions == (embed.weight._version, lm_head.weight._version)
    assert frozen_digest == projection.digest()
    assert embed.weight.grad is None and lm_head.weight.grad is None
    changed = [native_key(name) for name, value in head.named_parameters() if tensor_sha(value) != before[name]]
    assert changed and set(changed).issubset({native_key(name) for name in before})
    result = {'status': 'complete' if cursor == count else 'budget_incomplete', 'branch_sha': BRANCH_SHA,
              'supervised_tokens': cursor, 'selected_supervised_tokens': count, 'full_goal_tokens': 1000000,
              'step': step, 'dataset_sha256': dataset_sha, 'parent': parent, 'best': best, 'evaluations': evaluations,
              'changed_native_parameters': changed, 'shared_tensors_unchanged': True, 'checkpoint_reload_verified': reload_checked,
              'optimizer': 'AdamW', 'lr': a.lr, 'margin': a.margin, 'margin_weight': a.margin_weight,
              'token_batch': a.token_batch, 'elapsed_training_seconds_including_eval_and_checkpoints': time.monotonic() - started,
              'peak_xpu_bytes': {'0': torch.xpu.max_memory_allocated(0)},
              'feature_schedule': SCHEDULE, 'peak_host_feature_cache_bytes': train_store.peak_cached_bytes,
              'projection_mode':a.projection_mode, 'vocabulary_chunk':a.vocabulary_chunk,
              'limitations': ['partial 1M capture selection', 'causal replay approximations, not exact original requests',
                              'branch one-token MTP attention; contextual native serving requires live A/B',
                              'offline proxy does not prove speculative acceptance or speed']}
    temp = a.out / 'result.json.partial'
    temp.write_text(json.dumps(result, indent=2)); os.replace(temp, a.out / 'result.json')
    emit('complete', **result)


if __name__ == '__main__':
    main()
