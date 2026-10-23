"""Stub replacement for the deleted Cython ``_streams`` extension.

Provides stream utilities used by ``scipy.io.matlab``.
This is a stub.
"""

from __future__ import annotations

import io
import rsnumpy as np

__all__ = ['ZlibInputStream']


class ZlibInputStream:
    """Stub zlib input stream reader."""

    def __init__(self, stream):
        self.stream = stream
        self._pos = 0

    def read(self, n=-1):
        return self.stream.read(n)

    def readline(self):
        return self.stream.readline()

    def __iter__(self):
        return self

    def __next__(self):
        line = self.stream.readline()
        if not line:
            raise StopIteration
        return line
