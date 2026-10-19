"""Python adapter for the Rust linear-filter kernel."""

from math import prod

import rsnumpy as np

from rsscipy._rsscipy_rs.signal import _linear_filter


def _filter_values(values):
    flat = np.reshape(values, (-1,)).tolist()
    return flat


def _restore(real, imag, shape, dtype):
    output = np.asarray(real).reshape(shape).astype(dtype)
    if dtype.kind == 'c':
        output = output + 1j * np.asarray(imag).reshape(shape)
    return output.astype(dtype)


def _linear_filter_adapter(b, a, x, axis=-1, zi=None):
    axis = axis + x.ndim if axis < 0 else axis
    dtype_inputs = [b, a, x] + ([] if zi is None else [zi])
    dtype = np.result_type(*dtype_inputs)
    if dtype.kind not in ('f', 'c'):
        raise NotImplementedError(f"Parameter's dtype {dtype} is not supported")

    b = np.asarray(b).astype(dtype)
    a = np.asarray(a).astype(dtype)
    x = np.asarray(x).astype(dtype)
    if b.ndim != 1 or b.size == 0 or a.ndim != 1 or a.size == 0:
        raise ValueError("filter coefficients must be non-empty 1-D arrays")
    if axis < 0 or axis >= x.ndim:
        raise ValueError("selected axis is out of range")
    if a[0] == 0:
        raise ValueError("BUG: filter coefficient a[0] == 0 not supported yet")

    order = max(b.size, a.size) - 1
    moved_x = np.moveaxis(x, axis, -1)
    samples = moved_x.shape[-1]
    rows = prod(moved_x.shape[:-1])
    flat_x = np.reshape(moved_x, (-1,)).tolist()
    moved_shape = moved_x.shape

    if zi is None:
        flat_zi = []
        zi_shape = (*moved_shape[:-1], order)
    else:
        zi = np.asarray(zi).astype(dtype)
        if zi.ndim != x.ndim:
            raise ValueError(
                f"Dimensions of parameters x and zi must match, but "
                f"{x.ndim=}, {zi.ndim=}!"
            )
        expected = list(x.shape)
        expected[axis] = order
        for dim, (actual, required) in enumerate(zip(zi.shape, expected)):
            if dim == axis:
                if actual != required:
                    raise ValueError(f"Unexpected shape for parameter zi: expected {tuple(expected)}, found {zi.shape}.")
            elif actual not in (1, required):
                raise ValueError(f"Unexpected shape for parameter zi: expected {tuple(expected)}, found {zi.shape}.")
        zi = np.broadcast_to(zi, expected)
        zi = np.moveaxis(zi, axis, -1)
        zi_shape = zi.shape
        flat_zi = np.reshape(zi, (-1,)).tolist()

    b_values = _filter_values(b)
    a_values = _filter_values(a)
    out_re, out_im, state_re, state_im = _linear_filter(
        b_values, a_values, flat_x, flat_zi, rows, samples
    )
    output = _restore(out_re, out_im, moved_shape, dtype)
    output = np.moveaxis(output, -1, axis)
    if zi is None:
        return output
    final_state = _restore(state_re, state_im, zi_shape, dtype)
    final_state = np.moveaxis(final_state, -1, axis)
    return output, final_state
