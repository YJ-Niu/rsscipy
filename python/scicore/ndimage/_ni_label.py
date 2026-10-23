"""Python compatibility layer for Rust connected-component labeling."""

import rsnumpy as np

from scicore._scicore_rs import ndimage as _rust_ndimage


class NeedMoreBits(Exception):
    """Raised when the output dtype cannot represent all component labels."""


def _label(input, structure, output):
    input = np.asarray(input)
    output = np.asarray(output)
    structure = np.asarray(structure, dtype=bool)
    assert input.shape == output.shape, (
        "Shapes must match for input and output,"
        f"{input.shape} != {output.shape}"
    )
    assert input.ndim == structure.ndim, (
        "Structuring element must have same # of dimensions as input,"
        f"{input.ndim} != {structure.ndim}"
    )
    assert all(size == 3 for size in structure.shape), (
        "Structuring element must be size 3 in every dimension,"
        f"was {structure.shape}"
    )

    foreground = (input != 0).reshape((-1,)).tolist()
    labels, count = _rust_ndimage._label(
        foreground,
        structure.reshape((-1,)).tolist(),
        list(input.shape),
    )

    if output.dtype.kind not in ("b", "i", "u", "f"):
        raise TypeError(f"unsupported output dtype {output.dtype}")
    if output.dtype.kind in ("b", "i", "u"):
        max_value = 1 if output.dtype.kind == "b" else np.iinfo(output.dtype).max
        if count > max_value:
            raise NeedMoreBits
    converted = np.asarray(labels, dtype=output.dtype)
    if any(int(actual) != expected for actual, expected in zip(
        converted.tolist(), labels
    )):
        raise NeedMoreBits
    output[...] = converted.reshape(output.shape)
    return count
