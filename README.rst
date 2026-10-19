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
algorithms but rebuilds the compiled layer in Rust, and swaps the array backend
from NumPy to rsnumpy. Two consequences worth knowing up front:

- **No NumPy at runtime.** Arrays are rsnumpy arrays; the numerical kernels are
  Rust.
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
``tools/build_rust.py`` builds the five Rust crates declared in
``rust/modules.toml`` into CPython extension modules that are copied next to the
Python files that import them.

There is a helper that does the whole thing and installs the result into
``.venv``::

    ./build_wheel.sh

Useful flags::

    ./build_wheel.sh --debug        # debug build instead of release
    ./build_wheel.sh --out-dir dist # where to put the wheel
    ./build_wheel.sh --python /path/to/python

To do it by hand::

    python tools/build_rust.py      # build the Rust extension modules
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

Links
-----

- **Source code:** https://github.com/YJ-Niu/rsscipy
- **Bug reports:** https://github.com/YJ-Niu/rsscipy/issues
- **rsnumpy (array backend):** https://github.com/YJ-Niu/rsnumpy
- **Upstream SciPy:** https://github.com/scipy/scipy

License
-------

BSD 3-Clause, matching upstream SciPy. See ``LICENSE.txt``.
