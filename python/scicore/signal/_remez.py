"""Python adapter for the Rust Remez FIR design kernel."""

from scicore._scicore_rs import signal as _rust_signal

_remez = _rust_signal._remez
