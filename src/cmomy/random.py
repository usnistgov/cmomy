"""
Internal :class:`numpy.random.Generator` (:mod:`cmomy.random`)
==============================================================

:mod:`cmomy` is setup to use :class:`numpy.random.Generator` objects, as
opposed to using the "classic" :mod:`numpy.random` interface. For simplicity,
you can set a shared :class:`~numpy.random.Generator` object using the routine
:func:`default_rng`. You can set the seed of this random number generator (rng)
used across routines. You can also pass :class:`~numpy.random.Generator`
objects to routines that require random number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .core.typing import RngTypes


_DATA: dict[str, np.random.Generator] = {}


def set_internal_rng(rng: np.random.Generator) -> None:
    """
    Set the internal random number :class:`~numpy.random.Generator`.

    The function :func:`default_rng` will call `rng` if called with a new seed
    (or when called the first time). However, if want to override the internal
    rng, you can use this function.
    """
    _DATA["rng"] = rng


def get_internal_rng() -> np.random.Generator:
    """Get the internal random number :class:`~numpy.random.Generator`."""
    if "rng" not in _DATA:
        msg = "must set rng."
        raise ValueError(msg)
    return _DATA["rng"]


def _missing_internal_rng() -> bool:
    return "rng" not in _DATA


def default_rng(seed: RngTypes | None = None) -> np.random.Generator:
    """
    Get default internal random number generator.

    This is a shared default random number generator.  Calling it with a new
    seed will create a new shared random generator.  To create a one off
    generator, use :func:`numpy.random.default_rng`

    Parameters
    ----------
    seed :
        If specified, set the internal seed to this value. If pass in a
        :class:`numpy.random.Generator`, return that object.

    Returns
    -------
    Generator
        If called with ``seed=None`` (default), return the previously created rng
        (if already created). This means you can call ``default_rng(seed=...)``
        and subsequent calls of form ``default_rng()`` or ``default_rng(None)``
        will continue rng sequence from first call with ``seed=...``. If called
        with ``seed``, create a new rng sequence. Note that if you pass a
        :class:`~numpy.random.Generator` for seed, that object will be
        returned, but in this case, the internal generator will not be altered.

    See Also
    --------
    numpy.random.Generator
    """
    if isinstance(seed, np.random.Generator):
        return seed

    if seed is None:
        if _missing_internal_rng():
            set_internal_rng(np.random.default_rng())

    else:
        set_internal_rng(np.random.default_rng(seed=seed))

    return get_internal_rng()


def validate_rng(
    rng: RngTypes | None,
) -> np.random.Generator:
    """
    Decide whether to use passed :class:`~numpy.random.Generator` or that from :func:`default_rng`.

    Parameters
    ----------
    rng :
        If ``None``, use ``default_rng()``.
        Otherwise, try to return ``np.random.default_rng(rng)``.  If this fails, just return rng

    Returns
    -------
    Generator
    """
    if rng is None:
        return default_rng()
    return np.random.default_rng(rng)
