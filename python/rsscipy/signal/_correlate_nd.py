"""Python adapter for the Rust N-dimensional direct correlation kernel."""

import rsnumpy as np

from rsscipy._rsscipy_rs import signal as _rust_signal

_correlate_float = _rust_signal._correlate_nd_float
_correlate_signed = _rust_signal._correlate_nd_signed
_correlate_unsigned = _rust_signal._correlate_nd_unsigned


def _correlate_nd(in1, in2, mode):
    dtype = np.result_type(in1, in2)
    in1 = np.asarray(in1).astype(dtype)
    in2 = np.asarray(in2).astype(dtype)
    shape1 = list(in1.shape)
    shape2 = list(in2.shape)

    if dtype.kind in ('i', 'u'):
        bits = int(dtype.itemsize * 8)
        kernel = _correlate_signed if dtype.kind == 'i' else _correlate_unsigned
        result, shape = kernel(in1.tobytes(), in2.tobytes(), shape1, shape2, mode, bits)
        return np.asarray(result).astype(dtype).reshape(tuple(shape))

    if dtype.kind not in ('f', 'c'):
        raise ValueError(f"Unsupported type {dtype}")

    if dtype.kind == 'c':
        in1_real = np.real(in1).tobytes()
        in1_imag = np.imag(in1).tobytes()
        in2_real = np.real(in2).tobytes()
        in2_imag = np.imag(in2).tobytes()
    else:
        in1_real, in2_real = in1.tobytes(), in2.tobytes()
        in1_imag, in2_imag = bytes(len(in1_real)), bytes(len(in2_real))

    real, imag, shape = _correlate_float(
        in1_real,
        in1_imag,
        in2_real,
        in2_imag,
        int(dtype.itemsize // 2) if dtype.kind == 'c' else dtype.itemsize,
        (
            shape1,
            shape2,
            mode,
            dtype.itemsize == (8 if dtype.kind == 'c' else 4),
        ),
    )
    result = np.asarray(real).astype(dtype)
    if dtype.kind == 'c':
        result = result + 1j * np.asarray(imag).astype(dtype)
    return result.astype(dtype).reshape(tuple(shape))
