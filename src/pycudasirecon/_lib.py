"""Wrapper so that we can still import this package (for docs) without the library."""

import typing as _t

lib: _t.Any
try:
    from . import _libwrap as lib
except FileNotFoundError as e:
    import warnings

    t = e
    warnings.warn(f"\n{e}\n\nMost functionality will fail!\n")

    class _stub:
        def __getattr__(self, name: str) -> _t.Any:
            raise t

    lib = _stub()
