# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU checkpoint verification before explicit interrupted-run resume."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def finite(value):
    flat=value.reshape(-1)
    return all(bool(torch.isfinite(flat[i:i+1024**2]).all()) for i in range(0,flat.numel(),1024**2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    saved=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=True)
    assert saved['branch_sha']=='31dc9fa80374cfaff7fe53f396f11147d620b22f'
    assert saved['dataset_sha256']=='a867ae41f9bdb557662e67be0f652ea29978634890633cbaf0287ed39f382fc3'
    assert saved['feature_schedule']=='shuffled-capture-and-within-capture-v1'
    assert saved['projection_mode']=='streamed' and saved['optimizer_name']=='AdamW'
    model=saved['model']; assert len(model)==15 and sum(t.numel() for t in model.values())==424699392
    assert len(saved['train_keys'])==2088 and saved['cursor']==saved['supervised_tokens']>0
    assert all(t.dtype==torch.bfloat16 and finite(t) for t in model.values())
    opt=saved['optimizer'];ids=opt['param_groups'][0]['params'];assert len(ids)==15
    for identity,t in zip(ids,model.values()):
        state=opt['state'][identity]
        assert int(state['step'])==saved['step']
        for name in ['exp_avg','exp_avg_sq']:
            assert state[name].shape==t.shape and finite(state[name])
    changed=[]
    with safe_open(str(a.parent),framework='pt',device='cpu') as parent:
        expected={'mtp.'+('layers.0.'+k[6:] if k.startswith('layer.') else k) for k in model}
        assert set(parent.keys())==expected
        for name,t in model.items():
            native='mtp.'+('layers.0.'+name[6:] if name.startswith('layer.') else name)
            source=parent.get_tensor(native)
            assert source.shape==t.shape
            if not torch.equal(t,source):changed.append(native)
    assert changed
    digest=hashlib.sha256()
    with a.checkpoint.open('rb') as source:
        while block:=source.read(2*1024**2):digest.update(block)
    result={'passed':True,'step':saved['step'],'supervised_tokens':saved['cursor'],
            'bytes':a.checkpoint.stat().st_size,'sha256':digest.hexdigest(),
            'dataset_sha256':saved['dataset_sha256'],'model_and_optimizer_finite':True,
            'native_head_only':True,'changed_native_tensors':changed,'cpu_mmap_reload_verified':True,
            'scope':'CPU artifact validation; continued XPU updates must still be verified'}
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
