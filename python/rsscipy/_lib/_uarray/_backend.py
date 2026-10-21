"""Pure-Python replacement for the deleted C-backed ``_backend`` module.

This implements a minimal uarray dispatch protocol on top of the
pure-Python ``_uarray`` module, so that ``scipy._lib._uarray`` can be
imported without any C extension.  As Rust backends are added, individual
dispatch paths can be migrated.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import pickle
import threading
import typing
import types

from ._uarray import (
    BackendNotImplementedError,
    _Function,
    _SkipBackendContext,
    _SetBackendContext,
    _BackendState,
    _global_state,
)


__all__ = [
    "set_backend",
    "set_global_backend",
    "skip_backend",
    "register_backend",
    "determine_backend",
    "determine_backend_multi",
    "clear_backends",
    "create_multimethod",
    "generate_multimethod",
    "_Function",
    "BackendNotImplementedError",
    "Dispatchable",
    "wrap_single_convertor",
    "wrap_single_convertor_instance",
    "all_of_type",
    "mark_as",
    "set_state",
    "get_state",
    "reset_state",
    "_BackendState",
    "_SkipBackendContext",
    "_SetBackendContext",
]

ArgumentExtractorType = typing.Callable[..., tuple["Dispatchable", ...]]
ArgumentReplacerType = typing.Callable[
    [tuple, dict, tuple], tuple[tuple, dict]
]


def _unpickle_function(mod_name, qname, self_):
    try:
        module = __import__(mod_name)
        qname_parts = qname.split(".")
        func = module
        for q in qname_parts:
            func = getattr(func, q)
        if self_ is not None:
            func = types.MethodType(func, self_)
        return func
    except (ImportError, AttributeError) as e:
        from pickle import UnpicklingError
        raise UnpicklingError from e


def _pickle_function(func):
    mod_name = getattr(func, "__module__", None)
    qname = getattr(func, "__qualname__", None)
    self_ = getattr(func, "__self__", None)

    try:
        test = _unpickle_function(mod_name, qname, self_)
    except pickle.UnpicklingError:
        test = None

    if test is not func:
        raise pickle.PicklingError(
            f"Can't pickle {func}: it's not the same object as {test}"
        )

    return _unpickle_function, (mod_name, qname, self_)


def _pickle_state(state):
    return _BackendState._unpickle, state._pickle()


def _pickle_set_backend_context(ctx):
    return _SetBackendContext, ctx._pickle()


def _pickle_skip_backend_context(ctx):
    return _SkipBackendContext, ctx._pickle()


copyreg_pickle = getattr(pickle, "copyreg", None)
import copyreg

copyreg.pickle(_Function, _pickle_function)
copyreg.pickle(_BackendState, _pickle_state)
copyreg.pickle(_SetBackendContext, _pickle_set_backend_context)
copyreg.pickle(_SkipBackendContext, _pickle_skip_backend_context)


def get_state():
    """
    Returns an opaque object containing the current state of all the backends.
    """
    return _global_state._pickle()


def set_state(state):
    """
    A context manager that sets the state of the backends.
    """
    return _SetStateContext(state)


class _SetStateContext:
    def __init__(self, state):
        self.state = state
        self.old_state = None

    def __enter__(self):
        self.old_state = get_state()
        _BackendState._unpickle(self.state)
        return self

    def __exit__(self, *exc):
        _BackendState._unpickle(self.old_state, True)
        return False


@contextlib.contextmanager
def reset_state():
    """
    Returns a context manager that resets all state once exited.
    """
    with set_state(get_state()):
        yield


def create_multimethod(*args, **kwargs):
    """Creates a decorator for generating multimethods."""

    def wrapper(a):
        return generate_multimethod(a, *args, **kwargs)

    return wrapper


def generate_multimethod(
    argument_extractor,
    argument_replacer,
    domain,
    default=None,
):
    """Generates a multimethod."""
    kw_defaults, arg_defaults, opts = get_defaults(argument_extractor)
    ua_func = _Function(
        argument_extractor,
        argument_replacer,
        domain,
        arg_defaults,
        kw_defaults,
        default,
    )
    return functools.update_wrapper(ua_func, argument_extractor)


def set_backend(backend, coerce=False, only=False):
    """A context manager that sets the preferred backend."""
    tid = threading.get_native_id()
    try:
        return backend.__ua_cache__[tid, "set", coerce, only]
    except AttributeError:
        backend.__ua_cache__ = {}
    except KeyError:
        pass

    ctx = _SetBackendContext(backend, coerce, only)
    backend.__ua_cache__[tid, "set", coerce, only] = ctx
    return ctx


def skip_backend(backend):
    """A context manager that allows one to skip a given backend."""
    tid = threading.get_native_id()
    try:
        return backend.__ua_cache__[tid, "skip"]
    except AttributeError:
        backend.__ua_cache__ = {}
    except KeyError:
        pass

    ctx = _SkipBackendContext(backend)
    backend.__ua_cache__[tid, "skip"] = ctx
    return ctx


def get_defaults(f):
    sig = inspect.signature(f)
    kw_defaults = {}
    arg_defaults = []
    opts = set()
    for k, v in sig.parameters.items():
        if v.default is not inspect.Parameter.empty:
            kw_defaults[k] = v.default
        if v.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            arg_defaults.append(v.default)
        opts.add(k)
    return kw_defaults, tuple(arg_defaults), opts


def set_global_backend(backend, coerce=False, only=False, *, try_last=False):
    """Replaces the default backend for permanent use."""
    # Minimal stub: register the backend globally
    _register_global_backend(backend, coerce, only, try_last)


_global_backends = []


def _register_global_backend(backend, coerce, only, try_last):
    _global_backends.append((backend, coerce, only, try_last))


def register_backend(backend):
    """Registers a backend for permanent use."""
    _global_backends.append((backend, False, False, False))


def clear_backends(domain, registered=True, globals=False):
    """Clears registered backends."""
    if globals:
        _global_backends.clear()


class Dispatchable:
    """A utility class which marks an argument with a specific dispatch type."""

    def __init__(self, value, dispatch_type, coercible=True):
        self.value = value
        self.type = dispatch_type
        self.coercible = coercible

    def __getitem__(self, index):
        return (self.type, self.value)[index]

    def __str__(self):
        return f"<{type(self).__name__}: type={self.type!r}, value={self.value!r}>"

    __repr__ = __str__


def mark_as(dispatch_type):
    """Creates a utility function to mark something as a specific type."""
    return functools.partial(Dispatchable, dispatch_type=dispatch_type)


def all_of_type(arg_type):
    """Marks all unmarked arguments as a given type."""

    def outer(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            extracted_args = func(*args, **kwargs)
            return tuple(
                Dispatchable(arg, arg_type)
                if not isinstance(arg, Dispatchable)
                else arg
                for arg in extracted_args
            )

        return inner

    return outer


def wrap_single_convertor(convert_single):
    """Wraps a ``__ua_convert__`` defined for a single element to all elements."""

    @functools.wraps(convert_single)
    def __ua_convert__(dispatchables, coerce):
        converted = []
        for d in dispatchables:
            c = convert_single(d.value, d.type, coerce and d.coercible)
            if c is NotImplemented:
                return NotImplemented
            converted.append(c)
        return converted

    return __ua_convert__


def wrap_single_convertor_instance(convert_single):
    """Wraps a ``__ua_convert__`` defined for a single element to all elements."""

    @functools.wraps(convert_single)
    def __ua_convert__(self, dispatchables, coerce):
        converted = []
        for d in dispatchables:
            c = convert_single(self, d.value, d.type, coerce and d.coercible)
            if c is NotImplemented:
                return NotImplemented
            converted.append(c)
        return converted

    return __ua_convert__


def determine_backend(value, dispatch_type, *, domain, only=True, coerce=False):
    """Set the backend to the first active backend that supports ``value``."""
    dispatchables = (Dispatchable(value, dispatch_type, coerce),)
    # Minimal stub: no backend determined, return a no-op context
    return _NoOpContext()


def determine_backend_multi(
    dispatchables, *, domain, only=True, coerce=False, **kwargs
):
    """Set a backend supporting all ``dispatchables``."""
    if "dispatch_type" in kwargs:
        disp_type = kwargs.pop("dispatch_type")
        dispatchables = tuple(
            d if isinstance(d, Dispatchable) else Dispatchable(d, disp_type)
            for d in dispatchables
        )
    else:
        dispatchables = tuple(dispatchables)
        if not all(isinstance(d, Dispatchable) for d in dispatchables):
            raise TypeError("dispatchables must be instances of uarray.Dispatchable")

    if len(kwargs) != 0:
        raise TypeError(f"Received unexpected keyword arguments: {kwargs}")

    return _NoOpContext()


class _NoOpContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
