"""Low-level callback helpers.

Replaces the Cython module ``scipy/_lib/_ccallback_c.pyx``: the PyCapsule
machinery and the C-ABI test callables now live in Rust
(``rust/ccallback``), while the ``ctypes`` declarations stay in Python because
they are pure glue.
"""

import ctypes

from rsscipy._rsscipy_rs._lib import (
    check_capsule,
    get_capsule_signature,
    get_raw_capsule,
    plus1_addr,
    plus1b_addr,
    plus1bc_addr,
    sine_addr,
    sigs,
    test_call_cython,
)

__all__ = [
    "check_capsule",
    "get_capsule_signature",
    "get_raw_capsule",
    "sigs",
    "test_call_cython",
    "plus1_ctypes",
    "plus1b_ctypes",
    "plus1bc_ctypes",
    "sine_ctypes",
]

_plus1_t = ctypes.CFUNCTYPE(
    ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int), ctypes.c_void_p
)
_plus1b_t = ctypes.CFUNCTYPE(
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
)
_plus1bc_t = ctypes.CFUNCTYPE(
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
)
_sine_t = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_void_p)

plus1_ctypes = ctypes.cast(plus1_addr(), _plus1_t)
plus1b_ctypes = ctypes.cast(plus1b_addr(), _plus1b_t)
plus1bc_ctypes = ctypes.cast(plus1bc_addr(), _plus1bc_t)
sine_ctypes = ctypes.cast(sine_addr(), _sine_t)

# `LowLevelCallable.from_cython()` looks functions up in `__pyx_capi__`.  The
# Rust port is not a Cython module, but exposing the same mapping keeps the
# public API (and the Cython-based tests) working unchanged.
__pyx_capi__ = {
    "plus1_cython": get_raw_capsule(
        plus1_addr(), "double (double, int *, void *)", None),
    "plus1b_cython": get_raw_capsule(
        plus1b_addr(), "double (double, double, int *, void *)", None),
    "plus1bc_cython": get_raw_capsule(
        plus1bc_addr(), "double (double, double, double, int *, void *)", None),
}
