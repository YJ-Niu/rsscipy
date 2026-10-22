"""Build the Rust extension modules and expose `scipy` inside a Python env.

Meson cannot configure this tree (7 bundled subprojects are empty), so the
package is exposed to the target environment with a `.pth` file that puts the
source directory on `sys.path`.  The unified compiled extension produced by `tools/build_rust.py` is placed in
the Python package, so the result is a working editable-style install.

Usage::

    python tools/install_to_env.py                       # current interpreter
    python tools/install_to_env.py /path/to/.venv/bin/python
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PTH_NAME = "scicore.pth"


def site_packages(python: str) -> Path:
    code = "import site;print(site.getsitepackages()[0])"
    out = subprocess.run(
        [python, "-c", code], capture_output=True, text=True, check=True
    )
    return Path(out.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    python = str(Path(argv[0]).resolve()) if argv else sys.executable

    # 1. build the Rust extensions against exactly this interpreter
    sys.path.insert(0, str(ROOT / "tools"))
    os.environ["PYO3_PYTHON"] = python
    print(f"[scicore] PYO3_PYTHON={python}")
    from build_rust import build

    build("release", offline=True)

    # 2. make `scipy` importable from the target environment
    target = site_packages(python)
    target.mkdir(parents=True, exist_ok=True)
    pth = target / PTH_NAME
    pth.write_text(f"{ROOT}\n", encoding="utf-8")

    print(f"[scicore] wrote {pth} -> {ROOT}")
    print(f"[scicore] `import scipy` now works with {python}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
