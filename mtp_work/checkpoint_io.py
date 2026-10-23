# SPDX-License-Identifier: Apache-2.0
"""Bound dirty page-cache growth while streaming large checkpoints to HDD."""
import os


class SyncingWriter:
    """File-like writer that syncs at bounded byte intervals.

    torch.save streams tensor storages through ``write``. Splitting large
    writes into memoryview slices lets slow disks flush checkpoints
    incrementally instead of accumulating gigabytes of dirty page cache.
    """

    def __init__(self, fileobj, sync_interval_bytes=64 * 1024**2):
        if sync_interval_bytes <= 0:
            raise ValueError('sync interval must be positive')
        self.fileobj = fileobj
        self.sync_interval_bytes = sync_interval_bytes
        self.pending_bytes = 0
        self.total_bytes = 0
        self.sync_count = 0

    def write(self, data):
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            room = self.sync_interval_bytes - self.pending_bytes
            end = min(offset + room, len(view))
            written = self.fileobj.write(view[offset:end])
            if not written:
                raise OSError('checkpoint write made no progress')
            offset += written
            self.pending_bytes += written
            self.total_bytes += written
            if self.pending_bytes >= self.sync_interval_bytes:
                self.sync()
        return offset

    def sync(self):
        self.fileobj.flush()
        sync = getattr(os, 'fdatasync', os.fsync)
        sync(self.fileobj.fileno())
        self.pending_bytes = 0
        self.sync_count += 1

    def flush(self):
        self.sync()

    def tell(self):
        return self.fileobj.tell()

    def seek(self, *args):
        return self.fileobj.seek(*args)

    def fileno(self):
        return self.fileobj.fileno()

    def seekable(self):
        return self.fileobj.seekable()

    def writable(self):
        return self.fileobj.writable()
