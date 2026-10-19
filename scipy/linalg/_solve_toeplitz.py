"""Levinson-Durbin recursion for Toeplitz systems.

Replaces ``_solve_toeplitz.pyx``.  That module was a single fused-type Cython
function doing no floating point beyond plain arithmetic, so there is nothing
here that benefits from Rust: the recursion is inherently sequential
(``O(n^2)`` scalar updates with a dependency on the previous iteration) and a
loop in Python straight off rsnumpy scalars is both faithful to the reference
implementation and simpler than marshalling the workspace vectors back and
forth across FFI.

The algorithm is adapted from ``toeplitz.f90`` by Alan Miller (public domain).
"""

import rsnumpy as np
from rsnumpy.linalg import LinAlgError

__all__ = ['levinson']


def levinson(a, b):
    """Solve a linear Toeplitz system using Levinson recursion.

    Parameters
    ----------
    a : array_like, shape (2 * n - 1,)
        The first column of the matrix in reverse order (without the diagonal)
        followed by the first row, as described in the notes below.
    b : array_like, shape (n,)
        Right-hand side vector.

    Returns
    -------
    x : ndarray, shape (n,)
        The solution vector.
    reflection_coeff : ndarray, shape (n + 1,)
        Toeplitz reflection coefficients.  When `a` is symmetric Toeplitz and
        ``b`` is ``a[n:]``, as in the solution of autoregressive systems, then
        `reflection_coeff` also corresponds to the partial autocorrelation
        function.

    Notes
    -----
    The layout below is derived from the recursion itself and verified against
    `scipy.linalg.solve_toeplitz`; note that it is **not** the one the original
    Cython docstring claimed.  A length-`2n-1` array `a` encodes

    .. code-block:: text

        M[i, j] = a[n - 1 + i - j]

    so the diagonal sits at ``a[n - 1]``, the *sub*-diagonals run upwards in
    index and the *super*-diagonals run downwards.  Equivalently

    .. code-block:: python

        a = np.concatenate((r[-1:0:-1], c))

    where ``c`` is the first column and ``r`` the first row of the Toeplitz
    matrix -- exactly what `scipy.linalg._basic._solve_toeplitz` passes.  For
    n = 5 that reads ``[r4, r3, r2, r1, c0, c1, c2, c3, c4]`` with ``c0 == r0``.

    The original Cython docstring described the transpose of this arrangement
    (``c[::-1] + r[1:]``); following it produces wrong answers.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if b.ndim != 1:
        raise ValueError(f'expected a 1-D array for \'b\', got {b.ndim}-D')
    if a.ndim != 1:
        raise ValueError(f'expected a 1-D array for \'a\', got {a.ndim}-D')

    n = b.shape[0]
    if a.shape[0] != 2 * n - 1:
        raise ValueError(
            f'expected {2 * n - 1} elements in \'a\' for a {n}-long '
            f'\'b\', got {a.shape[0]}'
        )

    # Integers and bools cannot hold the recursion (which divides), and the
    # Cython original only accepted the four inexact types anyway.
    dtype = np.result_type(a.dtype, b.dtype)
    if dtype not in (np.float32, np.float64, np.complex64, np.complex128):
        dtype = np.float64
    a = np.asarray(a, dtype=dtype)
    b = np.asarray(b, dtype=dtype)

    x = np.zeros(n, dtype=dtype)                  # result
    g = np.zeros(n, dtype=dtype)                  # workspace
    h = np.zeros(n, dtype=dtype)                  # workspace
    reflection_coeff = np.zeros(n + 1, dtype=dtype)  # history

    if a[n - 1] == 0:
        raise LinAlgError('Singular principal minor')

    x[0] = b[0] / a[n - 1]
    reflection_coeff[0] = 1
    reflection_coeff[1] = x[0]

    if n == 1:
        return x, reflection_coeff

    g[0] = a[n - 2] / a[n - 1]
    h[0] = a[n] / a[n - 1]

    for m in range(1, n):
        # Compute numerator and denominator of x[m].
        x_num = -b[m]
        x_den = -a[n - 1]
        for j in range(m):
            nmj = n + m - (j + 1)
            x_num = x_num + a[nmj] * x[j]
            x_den = x_den + a[nmj] * g[m - j - 1]
        if x_den == 0:
            raise LinAlgError('Singular principal minor')
        x[m] = x_num / x_den
        reflection_coeff[m + 1] = x[m]

        # Compute x.
        for j in range(m):
            x[j] = x[j] - x[m] * g[m - j - 1]
        if m == n - 1:
            return x, reflection_coeff

        # Compute the numerator and denominator of g[m] and h[m].
        g_num = -a[n - m - 2]
        h_num = -a[n + m]
        g_den = -a[n - 1]
        for j in range(m):
            g_num = g_num + a[n + j - m - 1] * g[j]
            h_num = h_num + a[n + m - j - 1] * h[j]
            g_den = g_den + a[n + j - m - 1] * h[m - j - 1]

        if g_den == 0:
            raise LinAlgError('Singular principal minor')

        # Compute g and h.
        g[m] = g_num / g_den
        h[m] = h_num / x_den
        k = m - 1
        m2 = (m + 1) >> 1
        c1 = g[m]
        c2 = h[m]
        for j in range(m2):
            gj = g[j]
            gk = g[k]
            hj = h[j]
            hk = h[k]
            g[j] = gj - (c1 * hk)
            g[k] = gk - (c1 * hj)
            h[j] = hj - (c2 * gk)
            h[k] = hk - (c2 * gj)
            k -= 1
