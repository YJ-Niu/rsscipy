"""Vendored ``packaging.version`` shim.

Meson flattens ``packaging_version/src/*.py`` into
``scipy/_external/packaging_version/`` at install time.  The Rust build does
not run Meson, so the ``src`` directory is added to the package's search path
here instead, keeping ``scipy._external.packaging_version.version`` importable
straight from the source tree.
"""

import os as _os

__path__.append(_os.path.join(_os.path.dirname(__file__), "src"))
