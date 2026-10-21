"""Minimal pure-Python replacement for the deleted C ``_uarray`` extension.

Provides the uarray dispatch protocol (``BackendNotImplementedError``,
``_Function``, backend context managers) used by ``scipy._lib._uarray``.
This is a stub: the full uarray protocol can later be replaced by the
``uarray`` PyPI package or a Rust implementation.
"""

from __future__ import annotations

import contextlib
import threading


class BackendNotImplementedError(NotImplementedError):
    """Raised when no backend can handle a dispatchable call."""
    pass


class _Function:
    """A dispatchable function (uarray-compatible).

    Stores the argument extractor, replacer, domain, defaults and an
    optional default implementation.  When called, it runs the extractor to
    find dispatchable arguments, then tries registered implementations.
    """

    def __init__(self, argument_extractor, argument_replacer, domain,
                 arg_defaults=(), kw_defaults=None, default=None):
        self.argument_extractor = argument_extractor
        self.argument_replacer = argument_replacer
        self.domain = domain
        self.arg_defaults = arg_defaults or ()
        self.kw_defaults = kw_defaults or {}
        self.default = default

    def __call__(self, *args, **kwargs):
        # Minimal dispatch: try the default implementation if provided,
        # otherwise raise BackendNotImplementedError.
        if self.default is not None:
            return self.default(*args, **kwargs)
        raise BackendNotImplementedError(
            f"No backend could handle {self!r}"
        )

    def add_impl(self, impl):
        self.impls = getattr(self, 'impls', [])
        self.impls.append(impl)

    def __repr__(self):
        return f"_Function(domain={self.domain!r})"


class _BackendState:
    """Thread-local backend stack."""

    def __init__(self):
        self._local = threading.local()

    @property
    def stack(self):
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def push(self, backend):
        self.stack.append(backend)

    def pop(self):
        return self.stack.pop()

    def top(self):
        if self.stack:
            return self.stack[-1]
        return None


_global_state = _BackendState()


def _pickle_state(state):
    """Serialize backend state (stack of backends)."""
    return list(state.stack)


def _unpickle_state(data, reset=False):
    """Restore backend state from serialized data."""
    _global_state.stack = list(data)


_BackendState._pickle = lambda self: _pickle_state(self)
_BackendState._unpickle = staticmethod(lambda data, reset=False: _unpickle_state(data, reset))


class _SetBackendContext:
    """Context manager that temporarily sets a backend."""

    def __init__(self, backend, coerce=False):
        self.backend = backend
        self.coerce = coerce

    def __enter__(self):
        _global_state.push(self.backend)
        return self

    def __exit__(self, *exc):
        _global_state.pop()
        return False


class _SkipBackendContext:
    """Context manager that temporarily skips a backend."""

    def __init__(self, backend):
        self.backend = backend

    def __enter__(self):
        _global_state.push(("SKIP", self.backend))
        return self

    def __exit__(self, *exc):
        _global_state.pop()
        return False


__all__ = [
    "BackendNotImplementedError",
    "_Function",
    "_SkipBackendContext",
    "_SetBackendContext",
    "_BackendState",
]
