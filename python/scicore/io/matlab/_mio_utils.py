import rsnumpy as np

from scicore._scicore_rs import io as _rust_io


def squeeze_element(arr):
    if not arr.size:
        return np.array([], dtype=arr.dtype)
    arr2 = np.squeeze(arr)
    if arr2.ndim == 0 and arr2.dtype.kind != "V":
        return arr2.item()
    return arr2


def chars_to_strings(in_arr):
    shape = tuple(in_arr.shape)
    if not shape:
        raise ValueError("character array must have at least one dimension")
    last_dim = shape[-1]
    if in_arr.dtype.kind != "U" or in_arr.dtype.itemsize != 4:
        raise TypeError("character array must have dtype 'U1'")

    if last_dim == 0:
        out_shape = (0,) if len(shape) == 2 else shape[:-2] + (0,)
        return np.empty(out_shape, dtype=in_arr.dtype)

    strings = _rust_io.chars_to_strings(
        in_arr.tobytes(),
        list(shape),
        int(in_arr.dtype.itemsize),
    )
    out_shape = shape[:-1]
    result = np.asarray(strings, dtype=f"<U{last_dim}")
    return result.reshape(out_shape)
