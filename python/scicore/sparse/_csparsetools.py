"""Stub replacement for the deleted C ``_csparsetools`` extension.

Provides low-level sparse-matrix routines used by ``scipy.sparse._lil`` and
other modules, implemented on top of :mod:`rsnumpy`.  This is a stub.
"""

from __future__ import annotations

import rsnumpy as np


def lil_flatten_to_array(data, out):
    raise NotImplementedError("lil_flatten_to_array not yet ported")


def lil_get1(M, N, rows, data, row, col, out):
    raise NotImplementedError("lil_get1 not yet ported")


def lil_fancy_get(M, N, rows, data, row_idx, col_idx, out):
    raise NotImplementedError("lil_fancy_get not yet ported")


def lil_get_row_ranges(M, N, rows, indptr):
    raise NotImplementedError("lil_get_row_ranges not yet ported")


def lil_insert(M, N, rows, data, row, col, val):
    raise NotImplementedError("lil_insert not yet ported")


def lil_fancy_set(M, N, rows, data, row_idx, col_idx, val):
    raise NotImplementedError("lil_fancy_set not yet ported")


def lil_get_lengths(rows, lengths):
    raise NotImplementedError("lil_get_lengths not yet ported")
