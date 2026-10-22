"""In-tree PEP 517 build backend for rsscipy.

Replaces the former meson-python + C/Cython build. On a wheel build it:

  1. Builds the unified Rust extension via ``tools/build_rust.py`` (the only
     external tool required is ``cargo`` on PATH).
  2. Packages the ``python/rsscipy`` tree -- Python sources plus the compiled
     ``.so`` kernels -- into a wheel.

The backend deliberately uses only the Python standard library so a wheel can
be produced with no build-time package installs (i.e. fully offline).
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import platform
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "python" / "rsscipy"
PYPROJECT = ROOT / "pyproject.toml"

_EXCLUDE_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", "target"}
# Build inputs & dead C/Cython artifacts are not shipped in the wheel: only the
# Python sources and the compiled ``.so`` kernels are. ``.in`` covers the
# Cython/header templates (``*.pyx.in``, ``*.h.in``); ``.pxi``/``.pxd`` are
# Cython includes; the C/C++/Fortran sources are no longer compiled.
_EXCLUDE_SUFFIXES = {
    ".pyc", ".pyo", ".in", ".pxi", ".pxd",
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".f", ".f90",
}
# Files never shipped: OS junk (``.DS_Store``), dot-prefixed files, and the
# sdist-only ``MANIFEST.in`` (irrelevant to a wheel).
_EXCLUDE_NAMES = {".DS_Store", "MANIFEST.in"}

# Directories / files shipped in a source distribution. The wheel is produced
# from the live tree, so only the sdist needs an explicit manifest.
_SDIST_DIRS = ("python", "rust", "tools")
_SDIST_FILES = ("pyproject.toml", "README.rst", "LICENSE.txt")


def _project() -> dict:
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)["project"]


def _sha256_urlsafe(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")


def _wheel_tags() -> tuple[str, str, str]:
    """Return the (py, abi, platform) wheel tags for the current interpreter.

    The extension is an ABI-specific CPython module (built by pyo3 against the
    running interpreter), so the py and abi tags are both ``cpXY``. The platform
    tag is derived per OS: ``win_amd64``/``win_arm64`` on Windows,
    ``linux_<arch>`` on Linux (auditwheel retags to manylinux at release time),
    and the interpreter's own platform string on macOS (single-arch or
    universal2, e.g. ``macosx_14_0_arm64``).
    """
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    system = platform.system()
    if system == "Windows":
        plat = "win_arm64" if platform.machine() == "ARM64" else "win_amd64"
    elif system == "Darwin":
        plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    else:  # Linux (and other POSIX)
        plat = f"linux_{platform.machine()}"
    return py, py, plat


def _is_debug(config_settings: dict | None) -> bool:
    if not config_settings:
        return False
    return str(config_settings.get("rsbuild.debug", "")).strip().lower() in {"1", "true", "yes", "on"}


def _run_build_rust(release: bool) -> None:
    cmd = [sys.executable, str(ROOT / "tools" / "build_rust.py")]
    if not release:
        cmd.append("--debug")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _package_files() -> list[tuple[str, Path]]:
    if not PKG_DIR.is_dir():
        raise SystemExit(f"package directory not found: {PKG_DIR}")
    files: list[tuple[str, Path]] = []
    for path in sorted(PKG_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in _EXCLUDE_SUFFIXES:
            continue
        if path.name in _EXCLUDE_NAMES or path.name.startswith("."):
            continue
        if any(part in _EXCLUDE_DIRS for part in path.parts):
            continue
        files.append((path.relative_to(PKG_DIR.parent).as_posix(), path))
    return files


def _metadata_text(project: dict) -> str:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
    ]
    if project.get("description"):
        lines.append(f"Summary: {project['description']}")
    if project.get("requires-python"):
        lines.append(f"Requires-Python: {project['requires-python']}")
    for dep in project.get("dependencies", []):
        lines.append(f"Requires-Dist: {dep}")
    for extra, deps in project.get("optional-dependencies", {}).items():
        lines.append(f"Provides-Extra: {extra}")
        for dep in deps:
            lines.append(f"Requires-Dist: {dep} ; extra == '{extra}'")
    for author in project.get("authors", []):
        lines.append(f"Author: {author.get('name', '')}")
    for keyword in project.get("keywords", []):
        lines.append(f"Keywords: {keyword}")
    for url_name, url in project.get("urls", {}).items():
        lines.append(f"Project-URL: {url_name}, {url}")
    for classifier in project.get("classifiers", []):
        lines.append(f"Classifier: {classifier}")
    lines.append("")
    return "\n".join(lines)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    project = _project()
    name, version = project["name"], project["version"]
    # A wheel tagged universal2 must carry a fat extension.  When the running
    # interpreter is universal2, force the dual-arch build so the wheel tag
    # and the .so inside never disagree.
    if platform.system() == "Darwin" and "universal2" in sysconfig.get_platform():
        os.environ.setdefault("RSSCIPY_UNIVERSAL2", "1")
    # (1) Build the Rust kernels into the package tree.
    _run_build_rust(release=not _is_debug(config_settings))

    # (2) Package the tree into a wheel.
    py_tag, abi_tag, plat_tag = _wheel_tags()
    dist_info = f"{name}-{version}.dist-info"
    wheel_name = f"{name}-{version}-{py_tag}-{abi_tag}-{plat_tag}.whl"
    wheel_path = Path(wheel_directory) / wheel_name

    wheel_text = (
        "Wheel-Version: 1.0\n"
        "Generator: rsbuild\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {py_tag}-{abi_tag}-{plat_tag}\n"
    )
    top_level = f"{PKG_DIR.name}\n"

    record: list[list[str]] = []
    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, path in _package_files():
            data = path.read_bytes()
            zf.writestr(arc, data)
            record.append([arc, _sha256_urlsafe(data), str(len(data))])
        for arcname, content in (
            (f"{dist_info}/METADATA", _metadata_text(project)),
            (f"{dist_info}/WHEEL", wheel_text),
            (f"{dist_info}/top_level.txt", top_level),
        ):
            data = content.encode("utf-8")
            zf.writestr(arcname, data)
            record.append([arcname, _sha256_urlsafe(data), str(len(data))])
        record.append([f"{dist_info}/RECORD", "", ""])
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(record)
        zf.writestr(f"{dist_info}/RECORD", buf.getvalue().encode("utf-8"))
    return wheel_name


def _sdist_filter(info: tarfile.TarInfo):
    name = Path(info.name).name
    if name in _EXCLUDE_DIRS or name.endswith(tuple(_EXCLUDE_SUFFIXES)):
        return None
    # Keep the sdist as clean as the wheel: no OS junk, no dot-prefixed files.
    if name in _EXCLUDE_NAMES or name.startswith("."):
        return None
    return info


def build_sdist(sdist_directory, config_settings=None):
    project = _project()
    name, version = project["name"], project["version"]
    top = f"{name}-{version}"
    sdist_name = f"{top}.tar.gz"
    sdist_path = Path(sdist_directory) / sdist_name
    with tarfile.open(sdist_path, "w:gz") as tar:
        for d in _SDIST_DIRS:
            src = ROOT / d
            if src.is_dir():
                tar.add(src, arcname=f"{top}/{d}", filter=_sdist_filter)
        for f in _SDIST_FILES:
            src = ROOT / f
            if src.is_file():
                tar.add(src, arcname=f"{top}/{f}", filter=_sdist_filter)
    return sdist_name


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    project = _project()
    dist_info = f"{project['name']}-{project['version']}.dist-info"
    out = Path(metadata_directory) / dist_info
    out.mkdir(parents=True, exist_ok=True)
    (out / "METADATA").write_text(_metadata_text(project), encoding="utf-8")
    return dist_info


def get_requires_for_build_wheel(config_settings=None) -> list[str]:
    return []


def get_requires_for_build_sdist(config_settings=None) -> list[str]:
    return []
