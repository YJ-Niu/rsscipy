__init__py = fs.copyfile('__init__.py')
_cy_array_utils_pxd = fs.copyfile('_cythonized_array_utils.pxd')

_cython_tree += fs.copyfile('__init__.pxd')

_generate_pyx = find_program('_generate_pyx.py')

ilp64_flag = cython_blas_ilp64 ? '--ilp64' : []
cython_linalg = custom_target('cython_linalg',
  output: [
    'cython_blas.pyx',
    'cython_lapack.pyx',
    'cython_blas.pxd',
    'cython_lapack.pxd',
    '_blas_subroutines.h',
    '_lapack_subroutines.h'
  ],
  input: '_generate_pyx.py',
  command: [_generate_pyx, '-o', '@OUTDIR@', accelerate_flag, ilp64_flag],
  depend_files: [
    'cython_blas_signatures.txt',
    'cython_lapack_signatures.txt',
    '../_build_utils/_generate_blas_wrapper.py',
    '../_build_utils/_wrappers_common.py',
  ],
  install: true,
  # Install only the .pxd files
  install_dir: [
    false,
    false,
    py3.get_install_dir() / 'scipy/linalg',
    py3.get_install_dir() / 'scipy/linalg',
    false,
    false,
  ],
  install_tag: 'devel'
)
cython_blas_pxd = cython_linalg[2]
cython_lapack_pxd = cython_linalg[3]

# pyx -> c, pyx -> cpp generators, depending on __init__.py here.
linalg_init_cython_gen = generator(cython,
  arguments : cython_args,
  output : '@BASENAME@.c',
  depends : [_cython_tree, __init__py])

# pyx -> c, pyx -> cpp generators, depending on _cythonized_array_utils.pxd
linalg_init_utils_cython_gen = generator(cython,
  arguments : cython_args,
  output : '@BASENAME@.c',
  depends : [_cython_tree, __init__py, _cy_array_utils_pxd, cython_lapack_pxd])

# pyx -> c, pyx -> cpp generators, depending on copied pxd files and init
linalg_cython_gen = generator(cython,
  arguments : cython_args,
  output : '@BASENAME@.c',
  depends : [_cython_tree, __init__py, cython_blas_pxd])#, cython_linalg])

# `_fblas` / `_flapack` (and their ILP64 siblings `_fblas_64` / `_flapack_64`)
# used to be built here from `src/blas/*.cpp` and `src/lapack/*.cpp`, linked
# against an external BLAS/LAPACK.  Both are now Rust crates (`rust/fblas`,
# `rust/flapack`) producing `_fblas_rs` / `_flapack_rs`, which implement the
# kernels in Rust and link no external BLAS at all.  `scipy/linalg/_fblas.py`
# and `_flapack.py` import the `_rs` modules, so these C++ targets are dead
# code -- and, because the sources are being removed, referencing them here
# would break the configure step.

# _decomp_interpolative
py3.extension_module('_decomp_interpolative',
  linalg_cython_gen.process('_decomp_interpolative.pyx'),
  c_args: cython_c_args,
  dependencies: np_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'rsscipy/linalg'
)

# batched linalg inv module
# `_batched_linalg` is now pure Python (`_batched_linalg.py`); the C++ sources
# under `src/` are gone.

# `_solve_toeplitz` is now pure Python (`_solve_toeplitz.py`): the Levinson
# recursion is inherently sequential, so there was nothing for Rust to speed up.

# `_matfuncs_sqrtm_triu` is now `_matfuncs_sqrtm_triu.py` (pure Python on
# rsnumpy); the Cython version is superseded.

cython_blas_lapack_dep = cython_blas_ilp64 ? lapack_dep : lapack_lp64_dep
cython_blas = py3.extension_module('cython_blas',
  [
    linalg_cython_gen.process(cython_linalg[0]),  # cython_blas.pyx
    cython_linalg[4],  # _blas_subroutines.h
  ],
  c_args: cython_c_args,
  link_args: version_link_args,
  dependencies: [cython_blas_lapack_dep, np_dep],
  install: true,
  subdir: 'rsscipy/linalg'
)

cython_lapack = py3.extension_module('cython_lapack',
  [
    linalg_cython_gen.process(cython_linalg[1]),  # cython_lapack.pyx
    cython_linalg[5],  # _lapack_subroutines.h
  ],
  c_args: cython_c_args,
  link_args: version_link_args,
  dependencies: [cython_blas_lapack_dep, np_dep],
  install: true,
  subdir: 'rsscipy/linalg'
)

# `_decomp_update` is now `_decomp_update.py` (pure Python: it recomputes the
# QR factorisation instead of applying Givens rotations).  The
# `_decomp_update.pyx.in` template is gone, so the custom target above and the
# Cython build are both removed -- referencing them would break configure.

# `_internal_matfuncs` is now pure Python (`_internal_matfuncs.py`);
# `_matfuncsmodule.c` and `src/_matfuncs_*.c` are gone.


# `_cythonized_array_utils` is now pure Python: `_cythonized_array_utils.py`
# delegates the element walks to the Rust `array_utils` crate.  The `.pyx` is
# gone, so building it here would fail.

# `_linalg_pythran.py` is a plain Python module now, so it is installed as a
# source rather than compiled through pythran.
py3.install_sources(
  ['_linalg_pythran.py'],
  subdir: 'rsscipy/linalg'
)


python_sources = [
  '__init__.py',
  '__init__.pxd',
  '_basic.py',
  '_batched_linalg.py',
  '_cythonized_array_utils.pxd',
  '_cythonized_array_utils.pyi',
  '_cythonized_array_utils.py',
  '_decomp.py',
  '_decomp_cholesky.py',
  '_decomp_cossin.py',
  '_decomp_ldl.py',
  '_decomp_lu.py',
  '_decomp_polar.py',
  '_decomp_qr.py',
  '_decomp_qz.py',
  '_decomp_schur.py',
  '_decomp_svd.py',
  '_decomp_update.py',
  '_expm_frechet.py',
  '_fblas.py',
  '_flapack.py',
  '_internal_matfuncs.py',
  '_matfuncs.py',
  '_matfuncs_inv_ssq.py',
  '_matfuncs_sqrtm.py',
  '_matfuncs_sqrtm_triu.py',
  '_misc.py',
  '_procrustes.py',
  '_sketches.py',
  '_solvers.py',
  '_solve_toeplitz.py',
  '_special_matrices.py',
  '_testutils.py',
  'blas.py',
  'interpolative.py',
  'lapack.py',
]

# Rust extension modules.  `tools/build_rust.py` runs at configure time (see
# the top-level meson.build) and writes these next to the Python modules that
# import them; without the install step below the wheel would build happily and
# then fail at `import scipy.linalg`.
install_data(
  [
    '_array_utils_rs.so',
    '_fblas_rs.so',
    '_flapack_rs.so',
  ],
  install_dir: scipy_dir / 'linalg',
)

py3.install_sources(
  python_sources,
  subdir: 'rsscipy/linalg'
)

subdir('tests')
