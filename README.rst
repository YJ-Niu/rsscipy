rsSciPy
=======

.. image:: https://img.shields.io/badge/powered%20by-rsnumpy-blue.svg
  :target: https://github.com/YJ-Niu/rsnumpy

**rsSciPy** is a port of `SciPy <https://github.com/scipy/scipy>`__ onto
`rsnumpy <https://github.com/YJ-Niu/rsnumpy>`__ — a Rust-backed, NumPy-compatible
array library.

The import name stays ``scipy``, so it is a drop-in replacement::

    pip install rsscipy
    python -c "import scipy.linalg"

Why
---

SciPy's compiled kernels are C, C++, Cython and Fortran. rsSciPy keeps the
algorithms, migrates compiled kernels to Rust incrementally, and swaps the array
backend from NumPy to rsnumpy. Until that migration is complete, some native
extensions still use the upstream implementations. Two consequences worth
knowing up front:

- **No Python NumPy dependency at runtime.** Arrays are rsnumpy arrays. NumPy
  remains a build-time dependency for the C/Cython extensions that have not yet
  been ported.
- **Native-kernel migration is ongoing.** Signal filtering kernels including
  ``lfilter``, ``sosfilt``, ``upfirdn``, direct N-D ``correlate``,
  ``convolve2d``/``correlate2d``, ``remez`` FIR design, spline filtering, and
  2-D median filtering now run in Rust. FFTPACK's packed-spectrum convolution
  kernels and convolution-kernel generation, as well as cluster VQ assignment
  and cluster-mean updates, also run in Rust; other C, C++, Cython, and Fortran
  production extensions are still being replaced. Regular-grid interpolation's
  interval search and 2-D linear interpolation kernels, as well as reverse
  Cuthill-McKee graph ordering, also run in Rust.
- **Recomputation over incremental update.** Where upstream keeps a factorisation
  up to date with Givens rotations (``scipy.linalg.qr_update`` and friends), this
  port recomputes the factorisation. The results match, but the complexity is
  ``O(m n²)`` rather than ``O(m n)``. Each affected function says so in its
  docstring.

Status
------

This is a work in progress. ``import scipy.linalg`` works; other subpackages are
still being ported. Treat the API as experimental and pin the version if you
depend on it.

Building from source
--------------------

The build is **meson-python plus Rust**: meson builds the Python package, and
``tools/build_rust.py`` builds the unified Rust crate declared in
``rust/modules.toml`` into one CPython extension. Its internal Rust modules
remain organized by package domain.

There is a helper that does the whole thing and installs the result into
``.venv``::

    ./build_wheel.sh

Useful flags::

    ./build_wheel.sh --debug        # debug build instead of release
    ./build_wheel.sh --out-dir dist # where to put the wheel
    ./build_wheel.sh --python /path/to/python

To do it by hand::

    python tools/build_rust.py      # build the unified Rust extension
    python -m build --wheel         # build the wheel
    pip install dist/*.whl

``build_wheel.sh`` also runs ``cargo fmt --check`` and ``cargo clippy`` before
building, so a failing lint stops the build rather than shipping a wheel.

Requirements: a Rust toolchain (see ``rust-toolchain.toml``), Python >= 3.12,
and — on Linux — an OpenBLAS for the BLAS/LAPACK parts.

Development
-----------

Clone with submodules; meson hard-errors without them::

    git clone --recursive https://github.com/YJ-Niu/rsscipy.git

or, after a plain clone::

    git submodule update --init --recursive

Run the Rust unit tests from the workspace root::

    cd rust
    cargo test --workspace

Links
-----

- **Source code:** https://github.com/YJ-Niu/rsscipy
- **Bug reports:** https://github.com/YJ-Niu/rsscipy/issues
- **rsnumpy (array backend):** https://github.com/YJ-Niu/rsnumpy
- **Upstream SciPy:** https://github.com/scipy/scipy

License
-------

BSD 3-Clause, matching upstream SciPy. See ``LICENSE.txt``.
