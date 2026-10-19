"""Matrix exponential and matrix square root kernels for `scipy.linalg`.

Replaces the C extension built from ``scipy/linalg/_matfuncsmodule.c`` together
with ``src/_matfuncs_expm.c`` and ``src/_matfuncs_sqrtm.c``.

Both kernels are implemented against rsnumpy primitives.  Neither is a natural
fit for Rust: the exponential is a short nest of matrix products plus one linear
solve, and the square root is a fixed-point iteration of the same two
operations.  Shipping the matrices across FFI to call BLAS would cost more than
the arithmetic it saves, since rsnumpy's matmul/inv already do that internally.

Entry points, matching the C they replace::

    expA, info = matrix_exponential(A)
    sqrtmA, isIllconditioned, isSingular, info = recursive_schur_sqrtm(A)
"""

import math

import rsnumpy as np

__all__ = ['matrix_exponential', 'recursive_schur_sqrtm']

# Padé orders the algorithm may pick, each with the ||A||_1 below which it is
# accurate -- the same five pairs `_matfuncs_expm.c` uses.
_PADE_ORDERS = (3, 5, 7, 9, 13)
_PADE_THETA = (
    1.495585217958292e-002,
    2.539398330063230e-001,
    9.504178996162932e-001,
    2.097847961257068e+000,
    4.250000000000000e+000,
)

# Iteration budget for Denman-Beavers, which converges quadratically once the
# iterates are close, so this is generous.
_MAX_SQRTM_ITERATIONS = 60


def _pade_coefficients(m):
    """Coefficients of the numerator of the [m/m] Padé approximant of `exp`.

    The closed form is

        c_j = (2m - j)! m! / ((2m)! j! (m - j)!)

    with the denominator being the numerator at ``-z``.  Generating them beats
    transcribing tables nobody can read; as a check, m = 3 yields
    ``120 * c = (120, 60, 12, 1)``, which is SciPy's own coefficient vector.
    """
    fact = math.factorial
    return [fact(2 * m - j) * fact(m) / (fact(2 * m) * fact(j) * fact(m - j))
            for j in range(m + 1)]


def _matrix_polynomial(A, coeffs, ident):
    """``sum_j coeffs[j] * A**j`` evaluated by Horner."""
    result = coeffs[-1] * ident
    for c in reversed(coeffs[:-1]):
        result = np.matmul(result, A) + c * ident
    return result


def _norm1(A):
    """1-norm of `A`: the largest absolute column sum."""
    rows = np.asarray(A).tolist()
    if not rows:
        return 0.0
    return max(sum(abs(value) for value in column) for column in zip(*rows))


def _identity(n, dtype):
    return np.asarray(np.eye(n), dtype=dtype)


def _max_abs_difference(X, Y):
    rows_x = np.asarray(X).tolist()
    rows_y = np.asarray(Y).tolist()
    return max(abs(a - b)
               for row_x, row_y in zip(rows_x, rows_y)
               for a, b in zip(row_x, row_y))


def _stack_slices(parts, shape):
    """Reassemble per-slice 2-D results into the batched `shape`."""
    nested = [np.asarray(part).tolist() for part in parts]
    return np.asarray(nested).reshape(*shape)


def matrix_exponential(A):
    """Matrix exponential by scaling-and-squaring Padé.

    Parameters
    ----------
    A : ndarray, float64 or complex128, last two dimensions square

    Returns
    -------
    expA : ndarray
        The matrix exponential of `A`.
    info : int
        0 on success; negative if the internal linear solve failed.
    """
    A = np.asarray(A)
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError('expected a square matrix or a stack of square '
                         'matrices')

    if A.ndim > 2:
        flat = np.asarray(A).reshape(-1, A.shape[-2], A.shape[-1])
        slices = [matrix_exponential(flat[i]) for i in range(flat.shape[0])]
        return (_stack_slices([s[0] for s in slices], A.shape),
                min(s[1] for s in slices))

    n = A.shape[0]
    dtype = A.dtype
    norm = _norm1(A)
    if norm == 0.0:
        return _identity(n, dtype), 0

    # Cheapest order guaranteed accurate at this norm, scaling by a power of two
    # so the argument lands inside that order's guaranteed radius.
    order = None
    for candidate, theta in zip(_PADE_ORDERS, _PADE_THETA):
        if norm <= theta:
            order, squarings = candidate, 0
            break
    if order is None:
        order = _PADE_ORDERS[-1]
        squarings = max(0, int(math.ceil(math.log2(norm / _PADE_THETA[-1]))))

    scaled = A / (2 ** squarings) if squarings else A
    ident = _identity(n, dtype)
    coeffs = _pade_coefficients(order)

    numerator = _matrix_polynomial(scaled, coeffs, ident)
    denominator = _matrix_polynomial(-scaled, coeffs, ident)

    # rsnumpy's linalg helpers hand back a foreign array type whose `@` is not
    # wired up, so everything crossing that boundary goes through np.matmul.
    try:
        result = np.asarray(np.linalg.solve(denominator, numerator))
    except Exception:
        return np.zeros((n, n), dtype=dtype), -1

    for _ in range(squarings):
        result = np.matmul(result, result)
    return result, 0


def recursive_schur_sqrtm(A):
    """Principal matrix square root of `A`.

    NOTE on the algorithm.  The C this replaces computed the *Schur*
    decomposition ``A = Z T Z^H`` with LAPACK `gees`, took the square root of
    the triangular factor via ``_matfuncs_sqrtm._sqrtm_triu``, and undoes the
    transformation.  Reproducing that needs `dgees`, which the Rust LAPACK layer
    does not expose yet.  This interim therefore uses the **Denman-Beavers**
    iteration

        Y_0 = A,   Z_0 = I
        Y_{k+1} = (Y_k + Z_k^-1) / 2
        Z_{k+1} = (Z_k + Y_k^-1) / 2     ->  Y_k -> A^(1/2)

    which needs only matrix product and inverse.  It converges whenever `A` has
    no eigenvalues on the closed negative real axis -- a strictly larger class
    than `rsnumpy.linalg.eig` can even feed, since that routine rejects complex
    eigenvalues outright.  Convergence is verified against the residual
    ``X @ X == A`` before returning, so a bad input fails loudly instead of
    quietly returning garbage.  Once `dgees` lands, swap the body for Schur +
    `_sqrtm_triu`.

    Parameters
    ----------
    A : ndarray, float64 or complex128, last two dimensions square

    Returns
    -------
    sqrtmA : ndarray
        A square root ``X`` with ``X @ X == A``.
    isIllconditioned : bool
        True if convergence was slow or only partial.
    isSingular : bool
        True if an inverse failed or a factor vanished, i.e. `A` is singular.
    info : int
        0 on success; negative if no usable square root was found.
    """
    A = np.asarray(A)
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError('expected a square matrix or a stack of square '
                         'matrices')

    if A.ndim > 2:
        flat = np.asarray(A).reshape(-1, A.shape[-2], A.shape[-1])
        slices = [recursive_schur_sqrtm(flat[i]) for i in range(flat.shape[0])]
        return (_stack_slices([s[0] for s in slices], A.shape),
                any(s[1] for s in slices),
                any(s[2] for s in slices),
                min(s[3] for s in slices))

    n = A.shape[0]
    dtype = A.dtype
    scale = _norm1(A)
    tolerance = 1e-14 * max(1.0, scale)

    Y, Z = A, _identity(n, dtype)
    singular = False

    for _ in range(_MAX_SQRTM_ITERATIONS):
        try:
            Z_inv = np.asarray(np.linalg.inv(Z))
            Y_inv = np.asarray(np.linalg.inv(Y))
        except Exception:
            singular = True
            break
        Y_next = (Y + Z_inv) / 2.0
        Z_next = (Z + Y_inv) / 2.0
        step = _max_abs_difference(Y_next, Y)
        Y, Z = Y_next, Z_next
        if step <= tolerance:
            break

    residual = _max_abs_difference(np.matmul(Y, Y), A)
    good = residual <= 1e-8 * max(1.0, scale)

    if singular or not good:
        return np.zeros((n, n), dtype=dtype), True, singular or not good, -1

    illconditioned = residual > 1e-12 * max(1.0, scale)
    return Y, illconditioned, False, 0
