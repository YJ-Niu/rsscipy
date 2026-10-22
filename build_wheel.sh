#!/usr/bin/env bash
set -euo pipefail

# Build rsscipy into a wheel and install it into the local .venv.
#
# Usage: ./build_wheel.sh [--release|--debug] [--out-dir DIR] [--python PYTHON_EXEC]
#
# The build has two stages, and the order matters:
#
#   1. `tools/build_rust.py` builds the unified Rust crate listed in
#      `rust/modules.toml` and places its extension in the Python package.
#   2. `python -m build --wheel` runs the in-tree `rsbuild` backend
#      (`tools/rsbuild.py`), which packages the Python sources and compiled
#      `.so` kernels into a wheel.
#
# Skipping (1) produces a wheel that builds cleanly but fails when importing
# functionality backed by Rust because the unified extension is missing.
#
# Cross-platform: run directly on macOS / Linux; on Windows use Git Bash or WSL
# (the venv layout switches to Scripts/python.exe automatically).

# ========== Read package metadata from pyproject.toml ==========
NAME=$(grep -m1 '^name = ' pyproject.toml | sed -E 's/name = "(.*)"/\1/')
VERSION=$(grep -m1 '^version = ' pyproject.toml | sed -E 's/version = "(.*)"/\1/')

echo "Project: $NAME, Version: $VERSION"

# ========== Keep scipy/version.py in sync with pyproject.toml ==========
# `scipy/__init__.py` reports this at runtime, so a stale copy silently
# mislabels the installed package.
if [[ -f scipy/version.py ]]; then
  sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" scipy/version.py
  rm -f scipy/version.py.bak
  echo "  -> scipy/version.py updated to $VERSION"
fi

# ========== cargo is required for the Rust extension modules ==========
if ! command -v cargo; then
  echo "Error: 'cargo' not found in PATH." >&2
  echo "Please install Rust and Cargo from https://rustup.rs/" >&2
  echo "After installation, run: source \$HOME/.cargo/env" >&2
  exit 1
fi

RELEASE=true
OUT_DIR="wheelhouse"

# Default python executable; may be overridden by --python or a positional arg
PYTHON_EXEC="python"
PYTHON_EXEC_SET=""

# ========== Platform detection: Windows (Git Bash/MSYS/Cygwin) uses Scripts/ ==========
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*) VENV_BIN=".venv/Scripts"; VENV_PY=".venv/Scripts/python.exe" ;;
  *)                    VENV_BIN=".venv/bin";     VENV_PY=".venv/bin/python" ;;
esac

# Activate a local venv if one exists (puts its build tooling on PATH).
if [[ -f "$VENV_BIN/activate" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_BIN/activate"
fi

while [[ $# -gt 0 ]]; do
  case $1 in
    --release) RELEASE=true; shift ;;
    --debug) RELEASE=false; shift ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --python) PYTHON_EXEC="$2"; PYTHON_EXEC_SET=1; shift 2 ;;
    *) PYTHON_EXEC="$1"; PYTHON_EXEC_SET=1; shift ;;
  esac
done

# Prefer the local venv unless the caller chose an interpreter explicitly.
if [[ -f "$VENV_PY" && -z "${PYTHON_EXEC_SET:-}" ]]; then
  PYTHON_EXEC="$VENV_PY"
fi

if ! command -v "$PYTHON_EXEC" && [[ ! -x "$PYTHON_EXEC" && ! -f "$PYTHON_EXEC" ]]; then
  echo "Error: Python executable '$PYTHON_EXEC' not found or not executable." >&2
  exit 1
fi

# ========== Rust lint gates ==========
# Enforced before building so that a wheel is never produced from code that
# would fail `cargo clippy` in CI.
echo "Running fmt checks (cargo fmt --all -- --check) ..."
if ! (cd rust && cargo fmt --all -- --check); then
  echo "Error: fmt checks failed. Fix the warnings above before building." >&2
  exit 1
fi
echo "  -> fmt checks passed."

echo "Running clippy checks (cargo clippy --all-targets -- -D warnings) ..."
if ! (cd rust && cargo clippy --all-targets -- -D warnings); then
  echo "Error: clippy checks failed. Fix the warnings above before building." >&2
  exit 1
fi
echo "  -> clippy checks passed."

# ========== Stage 1: Rust extension modules ==========
echo "Building Rust extension modules (release=$RELEASE) ..."
if $RELEASE; then
  "$PYTHON_EXEC" tools/build_rust.py
else
  "$PYTHON_EXEC" tools/build_rust.py --debug
fi
echo "  -> Rust extension modules built."

# ========== Stage 2: meson-python wheel ==========
# `build` is the PEP 517 frontend.  This repo's .venv is created with uv and has
# no pip inside, so prefer uv and only fall back to pip.
if ! "$PYTHON_EXEC" -c "import build" 2>/dev/null; then
  echo "Installing 'build' ..."
  if command -v uv; then
    uv pip install --python "$PYTHON_EXEC" build
  elif "$PYTHON_EXEC" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_EXEC" -m pip install --quiet --upgrade build
  else
    echo "Error: neither uv nor pip is available to install 'build'." >&2
    exit 1
  fi
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
echo "Building wheel into $OUT_DIR using $PYTHON_EXEC (release=$RELEASE)"
"$PYTHON_EXEC" -m build --wheel --outdir "$OUT_DIR"

# ========== Install into the local venv ==========
WHEEL=$(ls -t "$OUT_DIR"/*.whl 2>/dev/null | head -n1 || true)
if [[ -n "$WHEEL" && -f "$WHEEL" ]]; then
  echo "Wheel built: $WHEEL"
  if [[ -f "$VENV_PY" ]]; then
    echo "Installing into .venv ..."
    # --no-deps: the wheel already carries everything it needs, and letting pip
    # resolve deps here risks pulling the real `scipy` from PyPI over this port.
    if command -v uv; then
      uv pip install --python "$VENV_PY" --reinstall --no-deps "$WHEEL"
    else
      "$VENV_PY" -m pip install --force-reinstall --no-deps "$WHEEL"
    fi
    echo "Installed into .venv successfully."
  else
    echo "Warning: .venv not found; skip install. Wheels available in $OUT_DIR" >&2
  fi
else
  echo "Warning: no .whl file found in $OUT_DIR; skip install." >&2
fi

echo "Done."
