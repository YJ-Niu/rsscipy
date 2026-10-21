"""Stub replacement for the deleted Cython ``_mio5_utils`` extension.

Provides MATLAB v5 file reader utilities used by ``scipy.io.matlab._mio5``.
This is a stub.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['VarReader5']


class VarReader5:
    """Stub MATLAB v5 variable reader."""

    def __init__(self, file_handle, *args, **kwargs):
        self.file_handle = file_handle

    def read(self):
        return None

    def get_vars(self):
        return {}
