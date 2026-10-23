# SPDX-License-Identifier: Apache-2.0
"""Depth-k sampler-matched verifier objective for the native Qwen3.5 MTP head.

The head is unrolled with its serving KV cache over captured continuation
tokens. Target and draft distributions receive the production temperature,
top-k, and top-p filters; their overlap is the one-step rejection-sampling
acceptance probability used to maximize expected accepted-prefix length.
"""
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.cache_utils import DynamicCache

from hermes_feature_store import capture_key


class StreamedLabelNLL(torch.autograd.Function):
    """Per-row NLL with vocabulary-blocked first-order gradients."""
    @staticmethod
    def forward(ctx, hidden, labels, projection):
        import torch.nn.functional as F
        with torch.autocast(hidden.device.type, enabled=False):
            h = hidden.half()
            lse = torch.full((len(h),), float('-inf'), device=h.device, dtype=torch.float32)
            target_logit = torch.zeros_like(lse)
            rows = torch.arange(len(h), device=h.device)
            for start, end in projection.blocks():
                logits = F.linear(h, projection.weight(start, end)).float()
                lse = torch.logaddexp(lse, logits.logsumexp(-1))
                valid = (labels >= start) & (labels < end)
                local = (labels - start).clamp(0, end - start - 1)
                target_logit += torch.where(valid, logits[rows, local], 0.)
            ctx.save_for_backward(h, labels, lse)
            ctx.projection = projection
            ctx.input_dtype = hidden.dtype
            return lse - target_logit

    @staticmethod
    def backward(ctx, grad_output):
        import torch.nn.functional as F
        h, labels, lse = ctx.saved_tensors
        with torch.autocast(h.device.type, enabled=False):
            gradient = torch.zeros_like(h, dtype=torch.float32)
            for start, end in ctx.projection.blocks():
                weight = ctx.projection.weight(start, end)
                logits = F.linear(h, weight).float()
                prob = (logits - lse[:, None]).exp()
                valid = (labels >= start) & (labels < end)
                local = (labels - start).clamp(0, end - start - 1)
                prob.scatter_add_(1, local[:, None], -valid.float()[:, None])
                prob.mul_(grad_output[:, None])
                gradient.add_(F.linear(prob.half(), weight.t()).float())
            return gradient.to(ctx.input_dtype), None, None


def label_nll(hidden, labels, projection):
    return StreamedLabelNLL.apply(hidden, labels, projection)


def verifier_rollout_loss(head, initial_hidden, first_tokens, verifier_hidden,
                          positions, projection, ce_weight=0.05,
                          rollout_tokens=None, temperature=1.0,
                          top_k=20, top_p=0.95):
    """Unroll one shared MTP layer and optimize sampler acceptance.

    Args are batched: initial_hidden ``[B,H]``, first_tokens ``[B]``, verifier
    hidden states ``[B,D,H]``, absolute RoPE positions ``[B,D]``. The aligned
    continuation tokens are fed at subsequent depths. ``top_k`` and ``top_p``
    must match the production verifier sampler.
    """
    batch, depth, _ = verifier_hidden.shape
    if rollout_tokens is None or rollout_tokens.shape != (batch, depth):
        raise ValueError('Sampler-matched rollout requires aligned continuation tokens')
    if temperature <= 0 or top_k <= 0 or not 0 < top_p <= 1:
        raise ValueError('Invalid serving sampler settings')
    cache = DynamicCache()
    recurrent_hidden, token_ids = initial_hidden, first_tokens
    nlls, overlaps, hard_matches = [], [], []
    for step in range(depth):
        state = head.forward_hidden(
            recurrent_hidden, token_ids, positions[:, step], past_key_values=cache
        )
        with torch.no_grad():
            target_logits, target_ids = projection.topk(verifier_hidden[:, step], top_k)
            target_probs, target_keep = filtered_topk_probs(target_logits, temperature, top_p)
        draft_logits, draft_ids = topk_with_grad(state, projection, top_k)
        draft_probs, draft_keep = filtered_topk_probs(draft_logits, temperature, top_p)
        overlaps.append(distribution_overlap(
            target_ids, target_probs, target_keep,
            draft_ids, draft_probs, draft_keep,
        ))
        target = target_ids[:, 0]
        nll = label_nll(state, target, projection)
        with torch.no_grad():
            draft_top1 = projection.topk(state, 1)[1].squeeze(-1)
            hard_matches.append(draft_top1.eq(target))
        nlls.append(nll)
        recurrent_hidden = state
        token_ids = rollout_tokens[:, step]

    nlls = torch.stack(nlls, dim=1)
    matches = torch.stack(hard_matches, dim=1)
    overlaps = torch.stack(overlaps, dim=1)
    survival = overlaps.cumprod(dim=1)
    expected_accepted = survival.sum(dim=1)
    # The overlap term optimizes the actual filtered sampler. The NLL anchor
    # retains a gradient if target and draft top-k supports initially differ.
    loss = ce_weight * nlls.mean() - expected_accepted.mean()
    prefix_matches = matches.to(torch.int32).cumprod(dim=1).bool()
    accepted_length = prefix_matches.sum(dim=1)
    per_depth = matches.float().mean(dim=0)
    prefix_acceptance = prefix_matches.float().mean(dim=0)
    return loss, {
        'loss': loss.detach(),
        'nll': nlls.detach().mean(),
        'expected_accepted': expected_accepted.detach().mean(),
        'hard_accepted_length': accepted_length.float().mean(),
        'per_depth_sampler_overlap': overlaps.detach().mean(dim=0),
        'per_depth_greedy_match': per_depth.detach(),
        'per_depth_prefix_acceptance': prefix_acceptance.detach(),
    }


def filtered_topk_probs(logits, temperature=1.0, top_p=0.95):
    """Apply temperature and nucleus filtering to descending top-k logits."""
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    keep = probs.cumsum(dim=-1) - probs < top_p
    keep[:, 0] = True
    probs = probs * keep
    return probs / probs.sum(dim=-1, keepdim=True), keep


def distribution_overlap(target_ids, target_probs, target_keep,
                          draft_ids, draft_probs, draft_keep):
    """Exact sum(min(p_target, p_draft)) over both filtered top-k supports."""
    same = target_ids[:, :, None].eq(draft_ids[:, None, :])
    both = target_keep[:, :, None] & draft_keep[:, None, :] & same
    minimum = torch.minimum(target_probs[:, :, None], draft_probs[:, None, :])
    return (minimum * both.to(minimum.dtype)).sum(dim=(1, 2))


class StreamedTopK(torch.autograd.Function):
    """Vocabulary-blocked top-k with recomputed hidden-only backward.

    Saving every block's dequantized vocabulary matrix in an ordinary linear
    autograd graph costs multiple dense vocabulary copies per rollout depth.
    This function saves only hidden states and selected token IDs, then reloads
    the needed blocks during backward.
    """
    @staticmethod
    def forward(ctx, hidden, projection, k):
        import torch.nn.functional as F
        h = hidden.half()
        best = torch.full((len(h), k), float('-inf'), device=h.device)
        tokens = torch.full((len(h), k), -1, dtype=torch.long, device=h.device)
        with torch.no_grad(), torch.autocast(h.device.type, enabled=False):
            for start, end in projection.blocks():
                logits = F.linear(h, projection.weight(start, end)).float()
                values, ids = logits.topk(min(k, end - start), dim=-1)
                ids = ids + start
                merged, choose = torch.cat((best, values), dim=-1).topk(k, dim=-1)
                tokens = torch.cat((tokens, ids), dim=-1).gather(1, choose)
                best = merged
        ctx.save_for_backward(h, tokens)
        ctx.projection = projection
        ctx.input_dtype = hidden.dtype
        return best, tokens

    @staticmethod
    def backward(ctx, grad_values, _grad_tokens):
        h, tokens = ctx.saved_tensors
        gradient = torch.zeros_like(h)
        for start, end in ctx.projection.blocks():
            in_block = (tokens >= start) & (tokens < end)
            if not in_block.any():
                continue
            weight = ctx.projection.weight(start, end)
            for rank in range(tokens.shape[1]):
                mask = in_block[:, rank]
                if mask.any():
                    local = (tokens[:, rank] - start).clamp(0, end - start - 1)
                    selected = weight[local]
                    gradient.add_(selected * grad_values[:, rank].to(selected.dtype)[:, None]
                                  * mask.to(selected.dtype)[:, None])
        return gradient.to(ctx.input_dtype), None, None


def topk_with_grad(hidden, projection, k):
    """Return exact streamed top-k logits with a bounded-memory custom gradient."""
    return StreamedTopK.apply(hidden, projection, k)


class RolloutFeatureStore:
    """On-demand six-position contiguous rollouts from root-disjoint captures."""
    def __init__(self, records, captures, hidden, depth=6, stride=None):
        self.records, self.captures, self.hidden = records, Path(captures), hidden
        self.depth, self.stride = depth, stride or depth
        if self.depth <= 0 or self.stride <= 0:
            raise ValueError('depth and stride must be positive')
        self.rows = []
        self.selected_labels = self.covered_labels = 0
        for record_index, record in enumerate(records):
            self.selected_labels += record['n_tgt']
            for lo, hi in record['spans']:
                if lo < 2 or hi <= lo or hi > len(record['input_ids']):
                    raise ValueError('Invalid rollout span')
                for start in range(lo, hi - self.depth + 1, self.stride):
                    self.rows.append((record_index, start))
                    self.covered_labels += self.depth
        self.uncovered_labels = self.selected_labels - self.covered_labels
        self.by_record = defaultdict(list)
        for row_index, (record_index, _) in enumerate(self.rows):
            self.by_record[record_index].append(row_index)

    def __len__(self):
        return len(self.rows)

    def batch(self, row_indices, device='cpu'):
        count, depth = len(row_indices), self.depth
        initial = torch.empty((count, self.hidden), dtype=torch.float16)
        teacher = torch.empty((count, depth, self.hidden), dtype=torch.float16)
        first_tokens = torch.empty(count, dtype=torch.long)
        positions = torch.empty((count, depth), dtype=torch.long)
        groups = defaultdict(list)
        for output_index, row_index in enumerate(row_indices):
            record_index, start = self.rows[row_index]
            groups[record_index].append((output_index, start))
        for record_index, items in groups.items():
            record = self.records[record_index]
            with safe_open(str(self.captures / (capture_key(record) + '.safetensors')),
                           framework='pt', device='cpu') as source:
                ids = source.get_tensor('input_ids').long()
                if ids.tolist() != record['input_ids']:
                    raise ValueError('Capture token alignment mismatch')
                h = source.get_slice('hidden')
                if h.get_shape() != [len(ids), self.hidden]:
                    raise ValueError('Capture hidden shape mismatch')
                crop = record.get('capture_crop_start', 0)
                for output_index, start in items:
                    initial[output_index].copy_(h[start - 2:start - 1].squeeze(0))
                    teacher[output_index].copy_(h[start - 1:start - 1 + depth])
                    first_tokens[output_index] = ids[start - 1]
                    positions[output_index] = torch.arange(
                        crop + start - 1, crop + start - 1 + depth, dtype=torch.long
                    )
        return (initial.to(device), first_tokens.to(device), teacher.to(device),
                positions.to(device))

    def continuation_batch(self, row_indices, device='cpu'):
        """Observed continuation tokens, aligned to the captured verifier states."""
        tokens = torch.empty((len(row_indices), self.depth), dtype=torch.long)
        for out_index, row_index in enumerate(row_indices):
            record_index, start = self.rows[row_index]
            tokens[out_index] = torch.tensor(
                self.records[record_index]['input_ids'][start:start + self.depth],
                dtype=torch.long,
            )
        return tokens.to(device)
