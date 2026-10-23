# SPDX-License-Identifier: Apache-2.0
"""Vocabulary-blocked frozen projection for the branch's exact CE+margin loss.

Uses original GPTQ->FP16 weights and recomputes vocabulary blocks in backward;
never stores a dense vocabulary matrix or full-vocabulary gradient. First order.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FeatureOutput(nn.Module):
    """Shared-head adapter: let unchanged MTPHead return its normalized features."""
    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(0, device=device), requires_grad=False)

    def forward(self, hidden):
        return hidden


class StreamedMargin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, labels, projection, margin, coefficient):
        with torch.autocast(hidden.device.type, enabled=False):
            h = hidden.half()
            n = len(h)
            rows = torch.arange(n, device=h.device)
            lse = torch.full((n,), -torch.inf, device=h.device)
            true = torch.zeros(n, device=h.device)
            wrong = torch.full_like(lse, -torch.inf)
            wrong_ids = torch.zeros(n, device=h.device, dtype=torch.long)
            best, pred = wrong.clone(), wrong_ids.clone()
            for start, end in projection.blocks():
                z = F.linear(h, projection.weight(start, end)).float()
                lse = torch.logaddexp(lse, z.logsumexp(-1))
                value, index = z.max(-1)
                take = value > best
                best = torch.maximum(best, value)
                pred = torch.where(take, index + start, pred)
                valid = (labels >= start) & (labels < end)
                local = (labels-start).clamp(0, end-start-1)
                true += torch.where(valid, z[rows,local], 0.)
                z[rows[valid],local[valid]] = -torch.inf
                value, index = z.max(-1)
                take = value > wrong
                wrong = torch.maximum(wrong,value)
                wrong_ids = torch.where(take,index+start,wrong_ids)
            active = margin - true + wrong > 0
            ce = (lse-true).mean()
            penalty = torch.relu(margin-true+wrong).mean()
            loss = ce + coefficient*penalty
            ctx.save_for_backward(h,labels,lse,wrong_ids,active)
            ctx.projection, ctx.coefficient, ctx.input_dtype = projection,coefficient,hidden.dtype
            ctx.mark_non_differentiable(ce,penalty,pred)
            return loss,ce,penalty,pred

    @staticmethod
    def backward(ctx, grad_loss, grad_ce, grad_penalty, grad_pred):
        h,labels,lse,wrong_ids,active = ctx.saved_tensors
        with torch.autocast(h.device.type, enabled=False):
            dh = torch.zeros_like(h,dtype=torch.float32)
            scale = grad_loss / len(h)
            for start,end in ctx.projection.blocks():
                weight = ctx.projection.weight(start,end)
                z = F.linear(h,weight).float()
                dz = (z-lse[:,None]).exp()
                valid = (labels>=start)&(labels<end)
                local = (labels-start).clamp(0,end-start-1)
                truth_gradient = -(1.+ctx.coefficient*active.float())*valid.float()
                dz.scatter_add_(1,local[:,None],truth_gradient[:,None])
                wrong_valid = (wrong_ids>=start)&(wrong_ids<end)&active
                local_wrong = (wrong_ids-start).clamp(0,end-start-1)
                dz.scatter_add_(1,local_wrong[:,None],(ctx.coefficient*wrong_valid.float())[:,None])
                dz.mul_(scale)
                dh.add_(F.linear(dz.half(),weight.t()).float())
            return dh.to(ctx.input_dtype),None,None,None,None


def streamed_terms(hidden,labels,projection,margin=1.,coefficient=.5):
    return StreamedMargin.apply(hidden,labels,projection,margin,coefficient)
