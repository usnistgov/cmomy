"""Sets up optional values."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

# TODO(wpk): Add in environment variables for these things...
# CMOMY_FASTMATH ->
# see https://github.com/numbagg/numbagg/blob/main/numbagg/decorators.py

NMAX = "nmax"
FASTMATH = "fastmath"
PARALLEL = "parallel"
CACHE = "cache"


def _set_options_from_environment() -> dict[str, Any]:
    import os

    def _get_boolean(key: str, default: str) -> bool:
        return os.environ.get(key, default).lower() in {"true", "t", "1"}

    return {
        NMAX: int(os.environ.get("CMOMY_NMAX", "20")),
        CACHE: _get_boolean("CMOMY_NUMBA_CACHE", "true"),
        PARALLEL: _get_boolean("CMOMY_NUMBA_PARALLEL", "true"),
        FASTMATH: _get_boolean("CMOMY_NUMBA_FASTMATH", "true"),
    }


OPTIONS: dict[str, Any] = _set_options_from_environment()


def _isbool(x: Any) -> bool:
    return isinstance(x, bool)


def _isint(x: Any) -> bool:
    return isinstance(x, int)


_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    NMAX: _isint,
    PARALLEL: _isbool,
    CACHE: _isbool,
    FASTMATH: _isbool,
}

_SETTERS: dict[str, Callable[[Any], Any]] = {}


class set_options:  # noqa: N801
    """
    Set options for xarray in a controlled context.

    Can be used as a context manager as well

    Parameters
    ----------
    nmax : int, default=20
        Maximum number of moments.
        [env var: CMOMY_NMAX]
    cache: bool, default=True
        If ``True``, cache numba functions.
        [env var: CMOMY_NUMBA_CACHE]
    fastmath: bool, default=True
        If ``True``, use `fastmath` option to numba.
        [env var: CMOMY_NUMBA_FASTMATH]
    parallel : bool, default=True
        If ``True``, allow parallel numba functions (can still be
        overridden in function calls.)
        [env var: CMOMY_NUMBA_PARALLEL]


    Notes
    -----
    You can turn off options setrting the corresponding environment variable to
    0. For example to turn off caching, you can set ``CMOMY_NUMBA_CACHE=0``.
    These only take effect at load time, so make sure the variable is set
    before importing ``cmomy``.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.old: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                msg = f"argument name {k!r} is not in the set of valid options {set(OPTIONS)!r}"
                raise ValueError(msg)
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                msg = f"option {k!r} given an invalid value: {v!r}"
                raise ValueError(msg)
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    @staticmethod
    def _apply_update(options_dict: dict[str, Any]) -> None:
        for k, v in options_dict.items():
            if k in _SETTERS:
                _SETTERS[k](v)
        OPTIONS.update(options_dict)

    def __enter__(self) -> None:
        return

    def __exit__(self, type: object, value: object, traceback: object) -> None:  # noqa: A002 # pylint: disable=redefined-builtin)
        self._apply_update(self.old)
