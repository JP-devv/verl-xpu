# SPDX-License-Identifier: Apache-2.0
"""CPU check that streamed torch checkpoints sync and reload exactly."""
import tempfile
from pathlib import Path

import torch

from checkpoint_io import SyncingWriter


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'fixture.pt'
        expected = {
            'weights': torch.arange(20_000, dtype=torch.float32).reshape(100, 200),
            'cursor': 1280,
            'metadata': {'sampler': {'temperature': 1.0, 'top_k': 20, 'top_p': .95}},
        }
        with path.open('wb', buffering=0) as raw:
            writer = SyncingWriter(raw, sync_interval_bytes=2048)
            torch.save(expected, writer)
            writer.flush()
            assert writer.sync_count > 1
            assert writer.total_bytes == path.stat().st_size
        restored = torch.load(path, map_location='cpu', weights_only=True)
        assert torch.equal(restored['weights'], expected['weights'])
        assert restored['cursor'] == expected['cursor']
        assert restored['metadata'] == expected['metadata']
        print({'pass': True, 'bytes': path.stat().st_size,
               'bounded_sync_interval': 2048, 'sync_count': writer.sync_count,
               'roundtrip': True})


if __name__ == '__main__':
    main()
