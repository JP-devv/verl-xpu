# SPDX-License-Identifier: Apache-2.0
"""Dense independent reference for vocabulary-blocked CE+margin and gradients."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

helpers = os.environ.get('HERMES_HELPERS','/work/scripts')
if not Path(helpers).is_dir():
    helpers = str(Path(__file__).resolve().parents[3]/'scripts')
sys.path.insert(0,helpers)
from check_projection_cache import TargetFixture
from streamed_projection import StreamedGPTQHead
from hermes_streamed_margin import streamed_terms


def run_reference(device='cpu'):
    torch.manual_seed(73)
    projection = StreamedGPTQHead(TargetFixture(),device=device,chunk=31)
    original = projection.digest()
    weight = torch.cat([projection.weight(a,b) for a,b in projection.blocks()])
    reports = []
    for dtype in [torch.float32,torch.bfloat16]:
        for margin in [0.,1.,100.]:
            h = torch.randn(9,256,device=device,dtype=dtype).requires_grad_(True)
            reference = h.detach().clone().requires_grad_(True)
            labels = torch.tensor([0,30,31,61,62,93,123,124,136],device=device)
            loss,ce,penalty,pred = streamed_terms(h,labels,projection,margin,.5)
            z = F.linear(reference.half(),weight).float()
            dense_ce = F.cross_entropy(z,labels)
            wrong = z.clone(); wrong.scatter_(1,labels[:,None],-torch.inf)
            dense_penalty = torch.relu(margin-z.gather(1,labels[:,None]).squeeze(1)+wrong.max(-1).values).mean()
            dense_loss = dense_ce+.5*dense_penalty
            loss.backward(); dense_loss.backward()
            errors = [abs(float((loss-dense_loss).detach())),abs(float((ce-dense_ce).detach())),abs(float((penalty-dense_penalty).detach()))]
            grad_error = float((h.grad.float()-reference.grad.float()).abs().max())
            assert max(errors)<2e-5 and grad_error<.002, (dtype,margin,errors,grad_error)
            assert torch.equal(pred,z.argmax(-1))
            reports.append({'dtype':str(dtype),'margin':margin,'loss_error':errors[0],
                            'gradient_max_error':grad_error})
    # Equal logits with unequal weight rows check first-index max ties and the
    # zero derivative at the exact ReLU boundary, including true token 0.
    for margin in [0.,1.]:
        h=torch.zeros(3,256,device=device,requires_grad=True)
        other=h.detach().clone().requires_grad_(True)
        labels=torch.tensor([0,31,136],device=device)
        loss,_,_,pred=streamed_terms(h,labels,projection,margin,.5)
        z=F.linear(other.half(),weight).float()
        wrong=z.clone();wrong.scatter_(1,labels[:,None],-torch.inf)
        expected=F.cross_entropy(z,labels)+.5*torch.relu(margin-z.gather(1,labels[:,None]).squeeze(1)+wrong.max(-1).values).mean()
        loss.backward();expected.backward()
        assert torch.equal(pred,z.argmax(-1)) and float((h.grad-other.grad).abs().max())<.002
    assert projection.digest()==original and projection._device_weight is None
    return {'device':device,'passed':True,'checks':reports,'same_top1':True,
            'packed_source_unchanged':True,'dense_vocabulary_cache':False,
            'scope':'tiny dense-vs-blocked numerical reference, not training'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu')
    print(json.dumps(run_reference(p.parse_args().device)))
