SciCore
=======

.. image:: https://img.shields.io/badge/powered%20by-rsnumpy-blue.svg
  :target: https://github.com/YJ-Niu/rsnumpy
.. image:: https://img.shields.io/pypi/v/scirs
  :target: https://pypi.org/project/scirs/
.. image:: https://img.shields.io/badge/python-3.12%2B-blue
.. image:: https://img.shields.io/badge/built%20with-Rust-orange
  :target: https://www.rust-lang.org/

**SciCore** is a port of `SciPy <https://github.com/scipy/scipy>`__ onto
`rsnumpy <https://github.com/YJ-Niu/rsnumpy>`__ — a Rust-backed, NumPy-compatible
array library. It keeps SciPy's algorithms and API, migrates compiled kernels to
Rust incrementally, and swaps the array backend from NumPy to rsnumpy.

The import name stays ``scicore``, so it is a drop-in replacement::

    pip install scirs
    python -c "import scicore.linalg"

Quick Start
-----------

Install from PyPI::

    pip install scirs

Use any SciCore module the same way you would use its SciPy counterpart::

    import scicore.linalg as la
    import scicore.optimize as opt
    import scicore.signal as sig
    import scicore.stats as stats

Modules
-------

SciCore mirrors SciPy's module structure:

- ``scicore.linalg`` — linear algebra (BLAS/LAPACK)
- ``scicore.optimize`` — optimization and root finding
- ``scicore.signal`` — signal processing
- ``scicore.stats`` — statistics
- ``scicore.integrate`` — integration and ODEs
- ``scicore.interpolate`` — interpolation
- ``scicore.fft`` / ``scicore.fftpack`` — Fourier transforms
- ``scicore.sparse`` — sparse matrices
- ``scicore.spatial`` — spatial structures and distances
- ``scicore.special`` — special functions
- ``scicore.ndimage`` — n-dimensional image processing
- ``scicore.cluster`` — clustering
- ``scicore.io`` — I/O utilities
- ``scicore.datasets`` — sample datasets
- ``scicore.constants`` — physical constants
- ``scicore.differentiate`` — differentiation

Why
---

SciCore's compiled kernels are C, C++, Cython and Fortran. SciCore keeps the
algorithms, migrates compiled kernels to Rust incrementally, and swaps the array
backend from NumPy to rsnumpy. Until that migration is complete, some native
extensions still use the upstream implementations. Two consequences worth
knowing up front:

- **No Python NumPy dependency at runtime.** Arrays are rsnumpy arrays. NumPy
  remains a build-time dependency for the C/Cython extensions that have not yet
  been ported.
- **Native-kernel migration is ongoing.** A growing list of kernels now run in
  Rust — signal filtering (``lfilter``, ``sosfilt``, ``upfirdn``, N-D
  ``correlate``, ``convolve2d``/``correlate2d``, ``remez`` FIR design, spline
  filtering, 2-D median filtering), FFTPACK packed-spectrum convolution,
  cluster VQ assignment, graph ordering (reverse Cuthill-McKee),
  least-squares Givens elimination, connected-component labeling, Hausdorff
  distance search, Boyett/Patefield contingency-table sampling, and more.
  Other C, C++, Cython, and Fortran production extensions are still being
  replaced.
- **Recomputation over incremental update.** Where upstream keeps a
  factorisation up to date with Givens rotations (``scicore.linalg.qr_update``
  and friends), this port recomputes the factorisation. The results match, but
  the complexity is ``O(m n²)`` rather than ``O(m n)``. Each affected function
  says so in its docstring.

Status
------

This is a work in progress. Native kernels are still being migrated to Rust;
until that is complete, some native extensions still use the upstream
implementations. Treat the API as experimental and pin the version if you
depend on it.

Building from source
--------------------

The build is driven by **maturin**: it compiles the pyo3 crate in ``rust/``
into the ``scicore._scicore_rs`` extension and packages the ``python/`` tree
into a wheel.

There is a helper that does the whole thing and installs the result into
``.venv``::

    ./build_wheel.sh

Useful flags::

    ./build_wheel.sh --debug        # debug build instead of release
    ./build_wheel.sh --out-dir dist # where to put the wheel
    ./build_wheel.sh --python /path/to/python

To do it by hand::

    pip install maturin
    maturin build --release -i python -o dist
    pip install dist/*.whl

``build_wheel.sh`` also runs ``cargo fmt --check`` and ``cargo clippy`` before
building, so a failing lint stops the build rather than shipping a wheel.

Requirements: a Rust toolchain, Python >= 3.12.

Development
-----------

Clone the repository::

    git clone --recursive https://github.com/YJ-Niu/scicore.git

The ``--recursive`` flag initializes the doc submodule (scipy-mathjax for
formula rendering in the docs).

Run the Rust unit tests from the workspace root::

    cd rust
    cargo test --workspace

The repository's SciCore-owned Python tests live in ``test/`` and
``tools/``. ``pytest.ini`` limits ``python -m pytest`` to those tests so the
vendored upstream SciCore test corpus is not run as if it were a SciCore suite.
The Python tests cover the rsnumpy migration and Rust-backed Python adapters;
the Rust workspace tests cover each migrated kernel.

Links
-----

- **Source code:** https://github.com/YJ-Niu/scicore
- **Bug reports:** https://github.com/YJ-Niu/scicore/issues
- **rsnumpy (array backend):** https://github.com/YJ-Niu/rsnumpy
- **Upstream SciCore:** https://github.com/scicore-dev/scicore

License
-------

BSD 3-Clause, matching upstream SciCore. See ``LICENSE.txt``.
