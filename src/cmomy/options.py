"""Sets up optional values."""

from __future__ import annotations

from typing import Any, Callable

# Useful if doing any command line editing of doc string stuff

NMAX = "nmax"
FASTMATH = "fastmath"
PARALLEL = "parallel"
CACHE = "cache"


OPTIONS: dict[str, Any] = {NMAX: 20, PARALLEL: True, CACHE: True, FASTMATH: True}


def _isbool(x: Any) -> bool:
    return isinstance(x, bool)


def _isint(x: Any) -> bool:
    return isinstance(x, int)


# def _isstr(x: Any) -> bool:
#     return isinstance(x, str)


_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    NMAX: _isint,
    PARALLEL: _isbool,
    CACHE: _isbool,
    FASTMATH: _isbool,
}

_SETTERS: dict[str, Callable[[Any], Any]] = {}


class set_options:
    """
    Set options for xarray in a controlled context.

    Currently supported options:

    - `NMAX` : max moment size
    You can use ``set_options`` either as a context manager:
    - CACHE : bool, default=True
    - FASTMATH : bool, default=True
    """

    def __init__(self, **kwargs: Any) -> None:
        self.old: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    f"argument name {k!r} is not in the set of valid options {set(OPTIONS)!r}"
                )
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                raise ValueError(f"option {k!r} given an invalid value: {v!r}")
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict: dict[str, Any]) -> None:
        for k, v in options_dict.items():
            if k in _SETTERS:
                _SETTERS[k](v)
        OPTIONS.update(options_dict)

    def __enter__(self) -> None:
        return

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self._apply_update(self.old)
