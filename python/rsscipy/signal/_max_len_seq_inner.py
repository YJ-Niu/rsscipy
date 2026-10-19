"""Python adapter for the Rust maximum-length-sequence kernel."""

import rsnumpy as np

from ._max_len_seq_inner_rs import _max_len_seq_inner as _max_len_seq_inner_rust

__all__ = ['_max_len_seq_inner']


def _max_len_seq_inner(taps, state, nbits, length, seq):
    sequence, final_state = _max_len_seq_inner_rust(
        taps, state, nbits, length, seq
    )
    seq[:len(sequence)] = np.asarray(sequence, dtype=np.int8)
    return np.array(final_state, dtype=np.int8)
