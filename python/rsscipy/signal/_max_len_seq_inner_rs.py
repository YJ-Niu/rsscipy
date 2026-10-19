"""Python wrapper for the Rust `_max_len_seq_inner_rs` extension module.

Replaces the original `_max_len_seq_inner.pyx`. The Rust module computes a
Maximum Length Sequence using a linear feedback shift register and returns
the final state as a Python list.
"""

import rsnumpy as np

from ._max_len_seq_inner_rs import _max_len_seq_inner

__all__ = ['_max_len_seq_inner']


def _max_len_seq_inner(taps, state, nbits, length, seq):
    """Compute MLS using a shift register with ring-buffer indexing.

    Parameters
    ----------
    taps : ndarray, shape (n_taps,)
        Tap positions (int32).
    state : ndarray, shape (nbits,)
        Initial state (int8, values 0 or 1). Modified in place.
    nbits : int
        Number of bits.
    length : int
        Number of output samples to compute.
    seq : ndarray, shape (length,)
        Output array (int8). Modified in place.

    Returns
    -------
    state : ndarray, shape (nbits,)
        Final state of the shift register.
    """
    new_state = _max_len_seq_inner(taps, state, nbits, length, seq)
    return np.array(new_state, dtype=np.int8)
