#!/usr/bin/env bash
set -euo pipefail

# Build scicore with maturin and install it into the local .venv.
#
# Usage: ./build_wheel.sh [--release|--debug] [--universal2] [--out-dir DIR]
#                         [--skip-lint] [--python PYTHON_EXEC]
#
# maturin (the PEP 517 backend declared in pyproject.toml) compiles the pyo3
# crate `rust/scicore_rs` into the `scicore._scicore_rs` extension and packages
# the `python/` tree into a wheel. The only external tool required is `cargo`.
# (On Windows maturin produces a `.pyd` automatically -- no special-casing.)
#
# Cross-platform: run directly on macOS / Linux; on Windows use Git Bash or WSL
# (the venv layout switches to Scripts/python.exe automatically).

# ========== Read package metadata from pyproject.toml ==========
NAME=$(grep -m1 '^name = ' pyproject.toml | sed -E 's/name = "(.*)"/\1/')
VERSION=$(grep -m1 '^version = ' pyproject.toml | sed -E 's/version = "(.*)"/\1/')

echo "Project: $NAME, Version: $VERSION"

# ========== Keep python/scicore/version.py in sync with pyproject.toml ==========
# `scicore/__init__.py` reports this at runtime, so a stale copy silently
# mislabels the installed package.
if [[ -f python/scicore/version.py ]]; then
  sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" python/scicore/version.py
  rm -f python/scicore/version.py.bak
  echo "  -> python/scicore/version.py updated to $VERSION"
fi

# ========== cargo is required for the Rust extension module ==========
if ! command -v cargo; then
  echo "Error: 'cargo' not found in PATH." >&2
  echo "Please install Rust and Cargo from https://rustup.rs/" >&2
  echo "After installation, run: source \$HOME/.cargo/env" >&2
  exit 1
fi

RELEASE=true
SKIP_LINT=false
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
    --release)   RELEASE=true; shift ;;
    --debug)     RELEASE=false; shift ;;
    --skip-lint) SKIP_LINT=true; shift ;;
    --universal2) export SCICORE_UNIVERSAL2=1; shift ;;  # accepted for back-compat; maturin builds the host arch
    --out-dir)   OUT_DIR="$2"; shift 2 ;;
    --python)    PYTHON_EXEC="$2"; PYTHON_EXEC_SET=1; shift 2 ;;
    *)           PYTHON_EXEC="$1"; PYTHON_EXEC_SET=1; shift ;;
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

# ========== Locate maturin (prefer the venv copy, then PATH) ==========
if [[ -x "$VENV_BIN/maturin" ]]; then
  MATURIN=("$VENV_BIN/maturin")
elif command -v maturin >/dev/null 2>&1; then
  MATURIN=(maturin)
elif "$PYTHON_EXEC" -m maturin --version >/dev/null 2>&1; then
  MATURIN=("$PYTHON_EXEC" -m maturin)
else
  echo "Error: 'maturin' not found. Install it with: pip install maturin" >&2
  exit 1
fi
echo "Using maturin: $(${MATURIN[@]} --version)"
echo "Using Python:  $PYTHON_EXEC ($("$PYTHON_EXEC" --version 2>&1))"

# ========== Rust lint gates ==========
# Enforced before building (locally) so that a wheel is never produced from code
# that would fail `cargo clippy`. CI has a dedicated lint job, so it passes
# `--skip-lint` to avoid duplicating the work on every matrix cell.
if ! $SKIP_LINT; then
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
fi

# ========== Stage: build the wheel with maturin ==========
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
echo "Building wheel into $OUT_DIR with maturin (release=$RELEASE) ..."
if $RELEASE; then
  "${MATURIN[@]}" build --release -i "$PYTHON_EXEC" -o "$OUT_DIR"
else
  "${MATURIN[@]}" build -i "$PYTHON_EXEC" -o "$OUT_DIR"
fi

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
