"""Utilities."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, cast

from .docstrings import docfiller
from .validate import (
    validate_mom,
    validate_mom_ndim,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Iterator,
    )

    from .typing import (
        Mom_NDim,
        Moments,
        MomentsStrict,
    )
    from .typing_compat import TypeVar

    T = TypeVar("T")


# * peek at iterable ----------------------------------------------------------
def peek_at(iterable: Iterable[T]) -> tuple[T, Iterator[T]]:
    """Returns the first value from iterable, as well as a new iterator with
    the same content as the original iterable
    """
    gen = iter(iterable)
    peek = next(gen)
    return peek, chain([peek], gen)


@docfiller.decorate
def mom_to_mom_ndim(mom: Moments) -> Mom_NDim:
    """
    Calculate mom_ndim from mom.

    Parameters
    ----------
    {mom}

    Returns
    -------
    {mom_ndim}
    """
    mom_ndim = 1 if isinstance(mom, int) else len(mom)
    return validate_mom_ndim(mom_ndim)


@docfiller.decorate
def select_mom_ndim(*, mom: Moments | None, mom_ndim: int | None) -> Mom_NDim:
    """
    Select a mom_ndim from mom or mom_ndim

    Parameters
    ----------
    {mom}
    {mom_ndim}

    Returns
    -------
    {mom_ndim}
    """
    if mom is not None:
        mom_ndim_calc = mom_to_mom_ndim(mom)
        if mom_ndim and mom_ndim_calc != mom_ndim:
            msg = f"{mom=} is incompatible with {mom_ndim=}"
            raise ValueError(msg)
        return validate_mom_ndim(mom_ndim_calc)

    if mom_ndim is None:
        msg = "must specify mom_ndim or mom"
        raise TypeError(msg)

    return validate_mom_ndim(mom_ndim)


@docfiller.decorate
def mom_to_mom_shape(mom: int | Iterable[int]) -> MomentsStrict:
    """Convert moments to moments shape."""
    return cast("MomentsStrict", tuple(m + 1 for m in validate_mom(mom)))


@docfiller.decorate
def mom_shape_to_mom(mom_shape: int | Iterable[int]) -> MomentsStrict:
    """Convert moments shape to moments"""
    return cast("MomentsStrict", tuple(m - 1 for m in validate_mom(mom_shape)))
