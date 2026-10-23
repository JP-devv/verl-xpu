# SPDX-License-Identifier: Apache-2.0
"""Independent CPU fixture for bounded feature coverage, resume and eviction."""
import json
import os
from pathlib import Path
import tempfile

import torch
from safetensors.torch import save_file

from hermes_feature_store import BoundedFeatureStore, capture_key


def main():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        records, expected = [], []
        for i in range(7):
            ids = list(range(i*20, i*20+12))
            record = dict(sid='fixture', w=str(i), input_ids=ids,
                          spans=[[3, 6], [8, 11]], n_tgt=6)
            hidden = torch.tensor([i*100+j for j in range(12)], dtype=torch.float16)[:,None].expand(12,4).contiguous()
            save_file({'input_ids':torch.tensor(ids), 'hidden':hidden},
                      str(directory/(capture_key(record)+'.safetensors')))
            for lo, hi in record['spans']:
                expected.extend((float(hidden[j-2,0]), ids[j-1], ids[j]) for j in range(lo, hi))
            records.append(record)
        # One capture fits; two exceed the bound and must trigger eviction.
        bound = 6 * (4*2+16)
        store = BoundedFeatureStore(records,directory,4,max_cache_bytes=bound)
        got = []
        for start in range(0,len(store),5):
            h,t,y = store.batch(start,min(len(store),start+5))
            got.extend(zip(h[:,0].tolist(),t.tolist(),y.tolist()))
        assert sorted(got)==sorted(expected) and len(got)==len(set(got))==42
        assert store.peak_cached_bytes==bound and len(store.cache)==1
        if hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_DONTNEED'):
            assert store.last_cache_advice_succeeded
        resumed = BoundedFeatureStore(records,directory,4,max_cache_bytes=bound)
        h,t,y = resumed.batch(17,31)
        assert list(zip(h[:,0].tolist(),t.tolist(),y.tolist()))==got[17:31]
        other = BoundedFeatureStore(records,directory,4,seed=42,max_cache_bytes=bound)
        h,t,y = other.batch(0,42)
        changed = list(zip(h[:,0].tolist(),t.tolist(),y.tolist()))
        assert changed!=got and sorted(changed)==sorted(expected)
        try:
            BoundedFeatureStore(records,directory,4,max_cache_bytes=bound-1)
        except ValueError:
            pass
        else:
            raise AssertionError('Oversized capture should be rejected before allocation')
        bad = dict(records[0],input_ids=[999]+records[0]['input_ids'][1:])
        try:
            BoundedFeatureStore([bad],directory,4).batch(0,6)
        except ValueError:
            pass
        else:
            raise AssertionError('Token mismatch should be rejected')
        overlap = dict(records[0],spans=[[3,6],[5,8]])
        try:
            BoundedFeatureStore([overlap],directory,4).batch(0,6)
        except ValueError:
            pass
        else:
            raise AssertionError('Overlapping labels should be rejected')
        print(json.dumps(dict(passed=True, all_labels_once=True, two_token_shift=True,
                              deterministic_resume=True, seed_changes_order_only=True,
                              exact_cache_budget_bytes=bound, eviction=True,
                              malformed_captures_rejected=True, scope='tiny CPU fixture')))


if __name__ == '__main__':
    main()
