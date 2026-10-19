"""Symmetry / Hermitian tests for square 2-D arrays.

Replaces the Cython module ``scipy/linalg/_cythonized_array_utils.pyx``; the
element walks now run in Rust (``rust/array_utils``).

Three data paths are used, because rsnumpy only exposes a zero-copy buffer for
float64 arrays:

* float64  -> ``is_sym_f64``          (``memoryview``, zero copy)
* complex  -> ``is_complex_f64``      (``.real`` / ``.imag`` views, zero copy)
* other    -> ``is_sym_her_bytes``    (``__array_interface__`` bytes, exact for
  integer dtypes)
"""

import rsnumpy as np

from rsscipy._lib._util import _apply_over_batch

from ._array_utils_rs import is_complex_f64, is_sym_f64, is_sym_her_bytes

__all__ = ['issymmetric', 'ishermitian']


def _check_shape(a):
    if a.ndim != 2:
        raise ValueError('Input array must be a 2D NumPy array.')
    if a.shape[0] != a.shape[1]:
        raise ValueError('Input array must be square.')


def _exact(a, hermitian):
    """Dispatch a square contiguous array to the Rust kernels."""
    n = a.shape[0]
    if np.iscomplexobj(a):
        re = np.ascontiguousarray(a.real, dtype=np.float64)
        im = np.ascontiguousarray(a.imag, dtype=np.float64)
        return is_complex_f64(memoryview(re), memoryview(im), n, hermitian)

    if a.dtype == np.float64:
        return is_sym_f64(memoryview(a), n)

    ai = a.__array_interface__
    return is_sym_her_bytes(ai['data'], n, ai['typestr'], hermitian)


@_apply_over_batch(('a', 2), signature="(i,i)->bool()", zero_size_fill=True)
def issymmetric(a, atol=None, rtol=None):
    """Check if a square 2D array is symmetric.

    Parameters
    ----------
    a : ndarray
        Input array of size (N, N).

    atol : float, optional
        Absolute error bound

    rtol : float, optional
        Relative error bound

    Returns
    -------
    sym : bool
        True if the array is symmetric.

    Raises
    ------
    TypeError
        If the dtype of the array is not supported, in particular, NumPy
        float16, float128 and complex256 dtypes for exact comparisons.

    See Also
    --------
    ishermitian : Check if a square 2D array is Hermitian

    Notes
    -----
    For square empty arrays the result is True by convention.  Complex valued
    arrays are tested for symmetricity and *not* for being Hermitian.

    The diagonal of the array is not scanned.  Thus if there are infs, NaNs or
    similar problematic entries on the diagonal, they will be ignored.  However,
    `numpy.inf` will be treated as a number, that is to say ``[[1, inf],
    [inf, 2]]`` will return ``True``.  On the other hand `numpy.nan` is never
    symmetric, say, ``[[1, nan], [nan, 2]]`` will return ``False``.

    When ``atol`` and/or ``rtol`` are given, the comparison is performed by
    `numpy.allclose` and the tolerance values are passed to it.  Otherwise an
    exact comparison against zero is performed by the Rust kernels.
    If one of ``atol`` or ``rtol`` is given the other is automatically set to
    zero.

    Examples
    --------
    >>> import numpy as np
    >>> from rsscipy.linalg import issymmetric
    >>> A = np.arange(9).reshape(3, 3)
    >>> A = A + A.T
    >>> issymmetric(A)
    True
    >>> Ac = np.array([[1. + 1.j, 3.j], [3.j, 2.]])
    >>> issymmetric(Ac)  # not Hermitian but symmetric
    True

    """
    _check_shape(a)
    if a.size == 0:
        return True
    a = np.ascontiguousarray(a)

    # An element-by-element scan is pointless for an inexact comparison, and
    # integers have no tolerance semantics.
    if (atol or rtol) and not np.issubdtype(a.dtype, np.integer):
        return bool(np.allclose(a, a.T,
                                atol=atol if atol else 0.,
                                rtol=rtol if rtol else 0.))

    return _exact(a, False)


@_apply_over_batch(('a', 2), signature="(i,i)->bool()", zero_size_fill=True)
def ishermitian(a, atol=None, rtol=None):
    """Check if a square 2D array is Hermitian.

    Parameters
    ----------
    a : ndarray
        Input array of size (N, N)

    atol : float, optional
        Absolute error bound

    rtol : float, optional
        Relative error bound

    Returns
    -------
    her : bool
        True if the array is Hermitian.

    Raises
    ------
    TypeError
        If the dtype of the array is not supported, in particular, NumPy
        float16, float128 and complex256 dtypes.

    See Also
    --------
    issymmetric : Check if a square 2D array is symmetric

    Notes
    -----
    For square empty arrays the result is True by convention.

    `numpy.inf` will be treated as a number, that is to say ``[[1, inf],
    [inf, 2]]`` will return ``True``.  On the other hand `numpy.nan` is never
    symmetric, say, ``[[1, nan], [nan, 2]]`` will return ``False``.

    When ``atol`` and/or ``rtol`` are given, the comparison is performed by
    `numpy.allclose` and the tolerance values are passed to it.  Otherwise an
    exact comparison against zero is performed by the Rust kernels.

    Examples
    --------
    >>> import numpy as np
    >>> from rsscipy.linalg import ishermitian
    >>> A = np.arange(9).reshape(3, 3)
    >>> A = A + A.T
    >>> ishermitian(A)
    True
    >>> A = np.array([[1., 2. + 3.j], [2. - 3.j, 4.]])
    >>> ishermitian(A)
    True
    >>> Ac = np.array([[1. + 1.j, 3.j], [3.j, 2.]])
    >>> ishermitian(Ac)  # not Hermitian but symmetric
    False
    >>> Af = np.array([[0, 1 + 1j], [1 - (1+1e-12)*1j, 0]])
    >>> ishermitian(Af)
    False
    >>> ishermitian(Af, atol=5e-11) # almost hermitian with atol
    True

    """
    _check_shape(a)
    if a.size == 0:
        return True
    a = np.ascontiguousarray(a)

    if (atol or rtol) and not np.issubdtype(a.dtype, np.integer):
        return bool(np.allclose(a, a.conj().T,
                                atol=atol if atol else 0.,
                                rtol=rtol if rtol else 0.))

    return _exact(a, True)
