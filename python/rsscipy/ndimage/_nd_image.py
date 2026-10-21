"""Stub replacement for the deleted C ``_nd_image`` extension.

Provides the n-dimensional image processing routines used by
``scipy.ndimage``, implemented on top of :mod:`rsnumpy`.  This is a stub:
performance-critical routines should later be ported to Rust under
``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np


def correlate1d(input, weights, axis, output, mode, cval, origin=0):
    raise NotImplementedError("correlate1d not yet ported to Rust/rsnumpy")


def correlate(input, weights, output, mode, cval, origin=0):
    raise NotImplementedError("correlate not yet ported to Rust/rsnumpy")


def uniform_filter1d(input, size, axis, output, mode, cval, origin=0):
    raise NotImplementedError("uniform_filter1d not yet ported to Rust/rsnumpy")


def min_or_max_filter1d(input, size, axis, output, mode, cval, origin=0):
    raise NotImplementedError("min_or_max_filter1d not yet ported to Rust/rsnumpy")


def min_or_max_filter(input, footprint, structure, output, mode, cval, origin=0):
    raise NotImplementedError("min_or_max_filter not yet ported to Rust/rsnumpy")


def rank_filter(input, rank, footprint, output, mode, cval, origin=0):
    raise NotImplementedError("rank_filter not yet ported to Rust/rsnumpy")


def generic_filter1d(input, function, filter_size, axis, output, mode, cval, origin=0):
    raise NotImplementedError("generic_filter1d not yet ported to Rust/rsnumpy")


def generic_filter(input, function, footprint, output, mode, cval, origin=0):
    raise NotImplementedError("generic_filter not yet ported to Rust/rsnumpy")


def fourier_filter(input, sigmas, n, axis, output, type):
    raise NotImplementedError("fourier_filter not yet ported to Rust/rsnumpy")


def fourier_shift(input, shifts, n, axis, output):
    raise NotImplementedError("fourier_shift not yet ported to Rust/rsnumpy")


def find_objects(input, max_label):
    return []


def value_indices(arr, ignoreIsNone, ignore_value_arr):
    return np.array([]), np.array([])


def watershed_ift(input, markers, structure, output):
    raise NotImplementedError("watershed_ift not yet ported to Rust/rsnumpy")


# Aliases commonly expected by ndimage submodules
convolve1d = correlate1d
convolve = correlate
