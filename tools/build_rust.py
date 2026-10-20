"""Build the unified Rust extension for rsscipy.

The entry in `rust/modules.toml` maps the unified Cargo crate to its CPython
extension name and destination inside the Python package. The resulting shared
object is copied to `<dest>/<lib>.so`.

Usage::

    python tools/build_rust.py            # release build
    python tools/build_rust.py --debug    # debug build
    python tools/build_rust.py --clean    # remove the generated .so files

Afterwards install the package into an environment with::

    python tools/install_to_env.py [<python-executable>]
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
RUST_DIR = ROOT / "rust"
MANIFEST = RUST_DIR / "modules.toml"


def dylib_name(lib: str) -> str:
    """File name cargo produces for a cdylib called `lib`."""
    system = platform.system()
    if system == "Darwin":
        return f"lib{lib}.dylib"
    if system == "Windows":
        return f"{lib}.dll"
    return f"lib{lib}.so"


def load_modules() -> list[dict[str, str]]:
    with open(MANIFEST, "rb") as fh:
        data = tomllib.load(fh)
    return list(data.get("module", []))


def build(profile: str, offline: bool) -> list[Path]:
    cargo = shutil.which("cargo")
    if cargo is None:
        raise SystemExit("cargo not found on PATH - install Rust first")

    target_dir = RUST_DIR / "target"
    produced: list[Path] = []

    for mod in load_modules():
        crate = mod["crate"]
        lib = mod["lib"]
        dest = ROOT / mod["dest"]
        manifest = RUST_DIR / crate / "Cargo.toml"

        cmd = [
            cargo, "build",
            "--manifest-path", str(manifest),
            "--target-dir", str(target_dir),
            "--features", "extension-module",
        ]
        if profile == "release":
            cmd.append("--release")
        if offline:
            cmd.append("--offline")

        print(f"[rsscipy] cargo build {crate} ({profile})")
        # run from `rust/` so that `rust/.cargo/config.toml` is picked up
        subprocess.run(cmd, check=True, cwd=str(RUST_DIR))

        built = target_dir / profile / dylib_name(lib)
        if not built.exists():
            raise SystemExit(f"expected artifact not found: {built}")

        dest.mkdir(parents=True, exist_ok=True)
        # `.so` is accepted by CPython's extension finder on every platform
        target = dest / f"{lib}.so"
        shutil.copy2(built, target)
        produced.append(target)
        print(f"[rsscipy]   -> {target.relative_to(ROOT)}")

    return produced


def clean() -> None:
    for mod in load_modules():
        target = ROOT / mod["dest"] / f"{mod['lib']}.so"
        if target.exists():
            target.unlink()
            print(f"[rsscipy] removed {target.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="build in debug mode")
    parser.add_argument("--clean", action="store_true", help="remove built .so files")
    parser.add_argument(
        "--online", action="store_true",
        help="allow cargo to access the network (default: use the local cache)",
    )
    args = parser.parse_args(argv)

    if args.clean:
        clean()
        return 0

    # Pin the interpreter pyo3 builds against.  Without this, pyo3 may pick a
    # different Python from PATH and emit module slots that the interpreter
    # used to run SciPy does not understand ("unknown slot ID ...").
    os.environ.setdefault("PYO3_PYTHON", sys.executable)
    os.environ.setdefault("CARGO_TARGET_DIR", str(RUST_DIR / "target"))
    print(f"[rsscipy] PYO3_PYTHON={os.environ['PYO3_PYTHON']}")
    build("debug" if args.debug else "release", offline=not args.online)
    return 0


if __name__ == "__main__":
    sys.exit(main())
