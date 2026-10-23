# SPDX-License-Identifier: Apache-2.0
"""Bounded host feature batches from existing Hermes capture shards.

Shuffle captures and their selected rows deterministically. Each label appears
once per pass; cursor-based resume reproduces the same schedule. No corpus-sized
allocation or accelerator feature cache is constructed.
"""
from bisect import bisect_right
from collections import OrderedDict
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open

SCHEDULE = 'shuffled-capture-and-within-capture-v1'


def capture_key(record):
    return hashlib.sha1(('%s:%s' % (record['sid'], record['w'])).encode()).hexdigest()[:20]


class BoundedFeatureStore:
    def __init__(self, records, captures, hidden, seed=61141, max_cache_bytes=64*1024**2):
        self.records, self.captures, self.hidden = records, Path(captures), hidden
        self.seed, self.max_cache_bytes = seed, max_cache_bytes
        generator = torch.Generator().manual_seed(seed)
        self.record_order = torch.randperm(len(records), generator=generator).tolist()
        self.offsets = [0]
        for index in self.record_order:
            count = records[index]['n_tgt']
            if count <= 0:
                raise ValueError('Every capture must contain selected labels')
            required = count * (hidden * 2 + 16)
            if required > max_cache_bytes:
                raise ValueError('A single capture exceeds the bounded host cache')
            self.offsets.append(self.offsets[-1] + count)
        self.cache = OrderedDict()
        self.cached_bytes = self.peak_cached_bytes = self.loads = 0

    def __len__(self):
        return self.offsets[-1]

    def _load(self, record_index):
        if record_index in self.cache:
            self.cache.move_to_end(record_index)
            return self.cache[record_index]
        record = self.records[record_index]
        count = record['n_tgt']
        required = count * (self.hidden * 2 + 16)
        # Evict before allocation, preserving the bound even at load boundaries.
        while self.cache and self.cached_bytes + required > self.max_cache_bytes:
            _, old = self.cache.popitem(last=False)
            self.cached_bytes -= sum(t.numel() * t.element_size() for t in old)
        features = torch.empty((count, self.hidden), dtype=torch.float16)
        tokens = torch.empty(count, dtype=torch.long)
        labels = torch.empty_like(tokens)
        with safe_open(str(self.captures / (capture_key(record) + '.safetensors')),
                       framework='pt', device='cpu') as source:
            ids = source.get_tensor('input_ids').long()
            if ids.tolist() != record['input_ids']:
                raise ValueError('Capture token alignment mismatch')
            h = source.get_slice('hidden')
            if h.get_shape() != [len(ids), self.hidden]:
                raise ValueError('Capture hidden shape mismatch')
            cursor, previous_end = 0, 2
            for lo, hi in record['spans']:
                if lo < previous_end or hi <= lo or hi > len(ids):
                    raise ValueError('Invalid MTP input/label span')
                n = hi - lo
                features[cursor:cursor+n].copy_(h[lo-2:hi-2])
                tokens[cursor:cursor+n].copy_(ids[lo-1:hi-1])
                labels[cursor:cursor+n].copy_(ids[lo:hi])
                cursor += n
                previous_end = hi
            if cursor != count:
                raise ValueError('Selected label count differs from capture mask')
        # Keep cached rows in source order. Permuting only requested batches
        # avoids a second capture-sized feature allocation during shuffling.
        value = (features, tokens, labels)
        self.cache[record_index] = value
        self.cached_bytes += required
        self.peak_cached_bytes = max(self.peak_cached_bytes, self.cached_bytes)
        self.loads += 1
        return value

    def batch(self, start, end):
        if not 0 <= start < end <= len(self):
            raise ValueError('Batch cursor outside selected label pass')
        h = torch.empty((end-start, self.hidden), dtype=torch.float16)
        t = torch.empty(end-start, dtype=torch.long)
        y = torch.empty_like(t)
        out = 0
        while start < end:
            schedule_index = bisect_right(self.offsets, start) - 1
            record_index = self.record_order[schedule_index]
            record_start = self.offsets[schedule_index]
            take = min(end, self.offsets[schedule_index+1]) - start
            features, tokens, labels = self._load(record_index)
            generator = torch.Generator().manual_seed(self.seed + 1 + record_index)
            order = torch.randperm(len(tokens), generator=generator)
            local = start - record_start
            rows = order[local:local+take]
            h[out:out+take] = features[rows]
            t[out:out+take] = tokens[rows]
            y[out:out+take] = labels[rows]
            out += take
            start += take
        return h, t, y
