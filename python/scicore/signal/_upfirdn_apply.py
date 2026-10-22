"""Python adapter for the Rust polyphase upfirdn kernel."""

import rsnumpy as np

from rsscipy._rsscipy_rs import signal as _rust_signal

_apply_rust = _rust_signal._apply
_output_len = _rust_signal._output_len
_pad_test_rust = _rust_signal._pad_test
_mode_enum_rust = _rust_signal.mode_enum

__all__ = ['_output_len', '_apply', '_pad_test', 'mode_enum']

_modes = {
    'constant': 0,
    'symmetric': 1,
    'edge': 2,
    'smooth': 3,
    'wrap': 4,
    'reflect': 5,
    'antisymmetric': 6,
    'antireflect': 7,
    'line': 8,
}


def mode_enum(mode):
    return _mode_enum_rust(mode)


def _as_array(real, imag, dtype, shape):
    values = np.asarray(real, dtype=dtype)
    if dtype.kind == 'c':
        values = values + 1j * np.asarray(imag, dtype=dtype)
    return values.reshape(shape)


def _pad_test(data, npre=0, npost=0, mode=0):
    data = np.asarray(data)
    mode = _modes.get(mode, mode) if isinstance(mode, str) else mode
    real, imag = _pad_test_rust(
        np.reshape(data, (-1,)).tolist(), npre, npost, mode
    )
    return _as_array(real, imag, data.dtype, (npre + data.size + npost,))


def _apply(data, h_trans_flip, out, up, down, axis, mode, cval):
    data = np.asarray(data)
    h_trans_flip = np.asarray(h_trans_flip)
    mode = _modes.get(mode, mode) if isinstance(mode, str) else mode
    real, imag = _apply_rust(
        np.reshape(data, (-1,)).tolist(),
        np.reshape(h_trans_flip, (-1,)).tolist(),
        list(data.shape),
        list(out.shape),
        up,
        down,
        axis,
        mode,
        cval,
    )
    out[...] = _as_array(real, imag, out.dtype, out.shape)
    return out
