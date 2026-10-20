"""FFT-based convolution helpers backed by Rust packed-spectrum kernels."""

import rsnumpy as np

from rsscipy._rsscipy_rs import fftpack as _rust_fftpack
from rsscipy.fft._duccfft.pyduccfft import r2r_fftpack

__all__ = [
    'destroy_convolve_cache',
    'convolve',
    'convolve_z',
    'init_convolution_kernel',
]


def destroy_convolve_cache():
    """Compatibility no-op; this implementation does not cache transforms."""


def _working_array(inout, overwrite_x):
    array = np.asarray(inout)
    if array.ndim != 1:
        raise ValueError("inout and omega must be 1-dimensional arrays of the same length")
    if array.dtype != np.float64:
        if overwrite_x:
            raise ValueError("overwrite_x requires a float64 input array")
        array = array.astype(np.float64)
    elif not overwrite_x:
        array = array.copy()
    return array


def convolve(inout, omega, swap_real_imag=False, overwrite_x=False):
    """Convolve packed FFTPACK data with a real spectrum multiplier."""
    array = _working_array(inout, overwrite_x)
    kernel = np.asarray(omega, dtype=np.float64)
    if kernel.ndim != 1 or kernel.shape[0] != array.shape[0]:
        raise ValueError(
            "inout and omega must be 1-dimensional arrays of the same length"
        )
    r2r_fftpack(array, None, True, True, out=array)
    values = _rust_fftpack._convolve_packed(
        array.tolist(), kernel.tolist(), bool(swap_real_imag)
    )
    array[...] = np.asarray(values, dtype=np.float64)
    r2r_fftpack(array, None, False, False, out=array)
    return array


def convolve_z(inout, omega_real, omega_imag, overwrite_x=False):
    """Convolve packed FFTPACK data with a complex spectrum multiplier."""
    array = _working_array(inout, overwrite_x)
    real = np.asarray(omega_real, dtype=np.float64)
    imag = np.asarray(omega_imag, dtype=np.float64)
    if (
        real.ndim != 1
        or imag.ndim != 1
        or real.shape[0] != array.shape[0]
        or imag.shape[0] != array.shape[0]
    ):
        raise ValueError(
            "inout and omega must be 1-dimensional arrays of the same length"
        )
    r2r_fftpack(array, None, True, True, out=array)
    values = _rust_fftpack._convolve_complex_packed(
        array.tolist(), real.tolist(), imag.tolist()
    )
    array[...] = np.asarray(values, dtype=np.float64)
    r2r_fftpack(array, None, False, False, out=array)
    return array


def init_convolution_kernel(
    n,
    kernel_func,
    d=0,
    zero_nyquist=None,
    kernel_func_extra_args=(),
):
    """Build the packed real spectrum for a periodic convolution kernel."""
    values = _rust_fftpack._init_convolution_kernel(
        n, kernel_func, d, zero_nyquist, tuple(kernel_func_extra_args)
    )
    return np.asarray(values, dtype=np.float64)
