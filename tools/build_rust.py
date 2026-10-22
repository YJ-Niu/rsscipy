"""Build the unified Rust extension for rsscipy.

The entry in `rust/modules.toml` maps the unified Cargo crate to its CPython
extension name and destination inside the Python package. The resulting shared
object is copied to `<dest>/<lib>.so`.

Usage::

    python tools/build_rust.py            # release build
    python tools/build_rust.py --debug    # debug build
    python tools/build_rust.py --clean    # remove the generated .so files
    python tools/build_rust.py --universal2
                                          # macOS: x86_64 + arm64 fat extension

On macOS, `--universal2` (or the `RSSCIPY_UNIVERSAL2=1` environment variable)
builds both architecture slices and merges them with `lipo`, producing a
single fat extension that works on Intel and Apple Silicon.

By default cargo may fetch missing crates from the network (required on CI
runners without a warm cache); pass `--offline` to use the local crate cache
only.

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


# macOS slices for a universal2 (fat) extension.
_DARWIN_TARGETS = ("x86_64-apple-darwin", "aarch64-apple-darwin")


def _env_flag(name: str) -> bool:
    """True when the environment variable is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _ensure_rustup_targets(targets: tuple[str, ...]) -> None:
    """Install missing rustup std targets (idempotent no-op when present)."""
    rustup = shutil.which("rustup")
    if rustup is None:
        return
    result = subprocess.run(
        [rustup, "target", "list", "--installed"],
        capture_output=True, text=True, check=True,
    )
    installed = {line.split()[0] for line in result.stdout.splitlines() if line.strip()}
    for target in targets:
        if target not in installed:
            print(f"[rsscipy] rustup target add {target}")
            subprocess.run([rustup, "target", "add", target], check=True)


def build(profile: str, offline: bool, universal2: bool = False) -> list[Path]:
    cargo = shutil.which("cargo")
    if cargo is None:
        raise SystemExit("cargo not found on PATH - install Rust first")

    if universal2 and platform.system() != "Darwin":
        raise SystemExit("--universal2 is only supported on macOS")

    target_dir = RUST_DIR / "target"
    targets = list(_DARWIN_TARGETS) if universal2 else [None]
    if universal2:
        _ensure_rustup_targets(_DARWIN_TARGETS)
    produced: list[Path] = []

    for mod in load_modules():
        crate = mod["crate"]
        lib = mod["lib"]
        dest = ROOT / mod["dest"]
        manifest = RUST_DIR / crate / "Cargo.toml"

        # One cargo build per architecture; the slices are either copied into
        # place directly (single arch) or merged with `lipo` (universal2).
        slices: list[Path] = []
        for target in targets:
            cmd = [
                cargo, "build",
                "--manifest-path", str(manifest),
                "--target-dir", str(target_dir),
                "--features", "extension-module",
            ]
            if target is not None:
                cmd += ["--target", target]
            if profile == "release":
                cmd.append("--release")
            if offline:
                cmd.append("--offline")

            where = f" ({target})" if target else ""
            print(f"[rsscipy] cargo build {crate} ({profile}{where})")
            # run from `rust/` so that `rust/.cargo/config.toml` is picked up
            subprocess.run(cmd, check=True, cwd=str(RUST_DIR))

            if target is not None:
                profile_dir = target_dir / target / profile
            else:
                profile_dir = target_dir / profile
            built = profile_dir / dylib_name(lib)
            if not built.exists():
                raise SystemExit(f"expected artifact not found: {built}")
            slices.append(built)

        dest.mkdir(parents=True, exist_ok=True)
        # `.so` is accepted by CPython's extension finder on every platform
        target_so = dest / f"{lib}.so"
        if len(slices) == 1:
            shutil.copy2(slices[0], target_so)
        else:
            # Merge the per-architecture slices into one fat binary.
            subprocess.run(
                ["lipo", "-create", *[str(s) for s in slices], "-output", str(target_so)],
                check=True,
            )
        produced.append(target_so)
        print(f"[rsscipy]   -> {target_so.relative_to(ROOT)}")

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
        "--offline", action="store_true",
        help="use the local crate cache only (default: cargo may fetch missing "
             "crates, which is required on CI runners without a warm cache)",
    )
    parser.add_argument(
        "--universal2", action="store_true",
        help="macOS only: build the x86_64 and arm64 slices and lipo them into "
             "one fat extension (also enabled by the RSSCIPY_UNIVERSAL2 env var)",
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
    universal2 = args.universal2 or _env_flag("RSSCIPY_UNIVERSAL2")
    build(
        "debug" if args.debug else "release",
        offline=args.offline,
        universal2=universal2,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
