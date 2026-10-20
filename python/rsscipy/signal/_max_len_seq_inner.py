"""Python adapter for the Rust maximum-length-sequence kernel."""

import rsnumpy as np

from rsscipy._rsscipy_rs import signal as _rust_signal
_max_len_seq_inner_rust = _rust_signal._max_len_seq_inner

__all__ = ['_max_len_seq_inner']


def _max_len_seq_inner(taps, state, nbits, length, seq):
    sequence, final_state = _max_len_seq_inner_rust(
        taps, state, nbits, length, seq
    )
    seq[:len(sequence)] = np.asarray(sequence, dtype=np.int8)
    return np.array(final_state, dtype=np.int8)
