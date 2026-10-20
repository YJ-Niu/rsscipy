import pytest
import rsnumpy as np
from rsnumpy.testing import assert_allclose, assert_equal

from rsscipy.fftpack import convolve


def test_convolve_identity_kernel():
    values = np.arange(8, dtype=np.float64)
    result = convolve.convolve(values, np.ones(values.shape))
    assert_allclose(result, values, atol=1e-12)
    assert_equal(values, np.arange(8, dtype=np.float64))


def test_convolve_overwrite_returns_mutated_array():
    values = np.arange(8, dtype=np.float64)
    result = convolve.convolve(values, np.ones(values.shape), overwrite_x=True)
    assert result is values
    assert_allclose(values, np.arange(8, dtype=np.float64), atol=1e-12)


def test_convolve_complex_identity_kernel():
    values = np.arange(8, dtype=np.float64)
    result = convolve.convolve_z(
        values,
        np.ones(values.shape),
        np.zeros(values.shape),
    )
    assert_allclose(result, values, atol=1e-12)


def test_init_convolution_kernel_passes_extra_callback_args():
    result = convolve.init_convolution_kernel(
        5,
        lambda frequency, scale: scale * (frequency + 1),
        d=1,
        kernel_func_extra_args=(2.0,),
    )
    assert_allclose(result, [0.4, -0.4, 0.0, 0.0, 0.0])


def test_convolve_requires_matching_vector_shapes():
    with pytest.raises(ValueError, match="same length"):
        convolve.convolve(np.ones(4), np.ones(3))
