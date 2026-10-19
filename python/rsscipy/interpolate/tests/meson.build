py3.install_sources([
    '__init__.py',
    '_rbfinterp_kernel_pythran.py',
    'test_bary_rational.py',
    'test_bsplines.py',
    'test_fitpack.py',
    'test_fitpack2.py',
    'test_gil.py',
    'test_interpnd.py',
    'test_interpolate.py',
    'test_ndgriddata.py',
    'test_pade.py',
    'test_polyint.py',
    'test_rbf.py',
    'test_rbfinterp.py',
    'test_rgi.py'
  ],
  subdir: 'rsscipy/interpolate/tests',
  install_tag: 'tests'
)

py3.install_sources([
    'data/bug-1310.npz',
    'data/estimate_gradients_hang.npy',
    'data/gcvspl.npz'
  ],
  subdir: 'rsscipy/interpolate/tests/data',
  install_tag: 'tests'
)

# `_rbfinterp_kernel_pythran.py` is plain Python and is already listed in the
# `py3.install_sources` block above; compiling it through pythran is gone.
