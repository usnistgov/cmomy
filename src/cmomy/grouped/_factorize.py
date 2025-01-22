from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cmomy.core.docstrings import docfiller

if TYPE_CHECKING:
    from typing import Any

    from cmomy.core.typing import BlockByModes, Groups, IndexAny, NDArrayInt


def factor_by(
    by: Groups,
    sort: bool = True,
) -> tuple[list[Any] | IndexAny | pd.MultiIndex, NDArrayInt]:
    """
    Factor by to codes and groups.

    Parameters
    ----------
    by : sequence
        Values to group by. Negative or ``None`` values indicate to skip this
        value. Note that if ``by`` is a pandas :class:`pandas.Index` object,
        missing values should be marked with ``None`` only.
    sort : bool, default=True
        If ``True`` (default), sort ``groups``.
        If ``False``, return groups in order of first appearance.

    Returns
    -------
    groups : list or :class:`pandas.Index`
        Unique group names (excluding negative or ``None`` Values.)
    codes : ndarray of int
        Indexer into ``groups``.


    Examples
    --------
    >>> by = [1, 1, 0, -1, 0, 2, 2]
    >>> groups, codes = factor_by(by, sort=False)
    >>> groups
    [1, 0, 2]
    >>> codes
    array([ 0,  0,  1, -1,  1,  2,  2])

    Note that with sort=False, groups are in order of first appearance.

    >>> groups, codes = factor_by(by)
    >>> groups
    [0, 1, 2]
    >>> codes
    array([ 1,  1,  0, -1,  0,  2,  2])

    This also works for sequences of non-integers.

    >>> by = ["a", "a", None, "c", "c", -1]
    >>> groups, codes = factor_by(by)
    >>> groups
    ['a', 'c']
    >>> codes
    array([ 0,  0, -1,  1,  1, -1])

    And for :class:`pandas.Index` objects

    >>> import pandas as pd
    >>> by = pd.Index(["a", "a", None, "c", "c", None])
    >>> groups, codes = factor_by(by)

    >>> groups
    Index(['a', 'c'], dtype='object')
    >>> codes
    array([ 0,  0, -1,  1,  1, -1])

    """
    from pandas import factorize  # pyright: ignore[reportUnknownVariableType]

    # filter None and negative -> None
    by_: Groups = (
        by
        if isinstance(by, pd.Index)
        else np.fromiter(
            (None if isinstance(x, (int, np.integer)) and x < 0 else x for x in by),  # pyright: ignore[reportUnknownArgumentType]
            dtype=object,
        )
    )

    codes, groups = factorize(by_, sort=sort)  # type: ignore[arg-type]

    codes = codes.astype(np.int64)
    if isinstance(by_, (pd.Index, pd.MultiIndex)):
        if not isinstance(groups, (pd.Index, pd.MultiIndex)):  # pragma: no cover
            msg = f"{type(groups)=} should be instance of pd.Index"  # pyright: ignore[reportUnknownArgumentType]
            raise TypeError(msg)
        groups.names = by_.names
        return groups, codes  # pyright: ignore[reportUnknownVariableType]

    return list(groups), codes  # pyright: ignore[reportUnknownArgumentType]


def block_by(
    ndat: int,
    block: int,
    mode: BlockByModes = "drop_last",
) -> NDArrayInt:
    """
    Get group by array for block reduction.

    Parameters
    ----------
    ndat : int
        Size of ``by``.
    block : int
        Block size. Negative values is a single block.
    mode : {"drop_first", "drop_last", "expand_first", "expand_last"}
        What to do if ndat does not divide equally by ``block``.

        - "drop_first" : drop first samples
        - "drop_last" : drop last samples
        - "expand_first": expand first block size
        - "expand_last": expand last block size

    Returns
    -------
    by : ndarray
        Group array for block reduction.

    See Also
    --------
    cmomy.grouped.reduce_data_grouped

    Examples
    --------
    >>> block_by(5, 2)
    array([ 0,  0,  1,  1, -1])

    >>> block_by(5, 2, mode="drop_first")
    array([-1,  0,  0,  1,  1])

    >>> block_by(5, 2, mode="expand_first")
    array([0, 0, 0, 1, 1])

    >>> block_by(5, 2, mode="expand_last")
    array([0, 0, 1, 1, 1])

    """
    if block <= 0 or block == ndat:
        return np.broadcast_to(np.int64(0), ndat)

    if block > ndat:
        msg = f"{block=} > {ndat=}."
        raise ValueError(msg)

    if mode not in {"drop_first", "drop_last", "expand_first", "expand_last"}:
        msg = f"Unknown {mode=}"
        raise ValueError(msg)

    nblock = ndat // block
    by = np.arange(nblock).repeat(block).astype(np.int64, copy=False)
    if len(by) == ndat:
        return by

    shift = ndat - len(by)
    pad_width = (shift, 0) if mode.endswith("_first") else (0, shift)
    if mode.startswith("drop"):
        return np.pad(by, pad_width, mode="constant", constant_values=-1)
    return np.pad(by, pad_width, mode="edge")


@docfiller.decorate
def factor_by_to_index(
    by: Groups,
    **kwargs: Any,
) -> tuple[list[Any] | IndexAny | pd.MultiIndex, NDArrayInt, NDArrayInt, NDArrayInt]:
    """
    Transform group_idx to quantities to be used with :func:`reduce_data_indexed`.

    Parameters
    ----------
    by: array-like
        Values to factor.
    **kwargs
        Extra arguments to :func:`numpy.argsort`

    Returns
    -------
    groups : list or :class:`pandas.Index`
        Unique groups in `group_idx` (excluding Negative or ``None`` values in
        ``group_idx`` if ``exclude_negative`` is ``True``).
    index : ndarray
        Indexing array. ``index[start[k]:end[k]]`` are the index with group
        ``groups[k]``.
    start : ndarray
        See ``index``
    end : ndarray
        See ``index``.

    See Also
    --------
    cmomy.grouped.reduce_data_indexed
    factor_by

    Examples
    --------
    >>> factor_by_to_index([0, 1, 0, 1])
    ([0, 1], array([0, 2, 1, 3]), array([0, 2]), array([2, 4]))

    >>> factor_by_to_index(["a", "b", "a", "b"])
    (['a', 'b'], array([0, 2, 1, 3]), array([0, 2]), array([2, 4]))

    Also, missing values (None or negative) are excluded:

    >>> factor_by_to_index([None, "a", None, "b"])
    (['a', 'b'], array([1, 3]), array([0, 1]), array([1, 2]))

    You can also pass :class:`pandas.Index` objects:

    >>> factor_by_to_index(pd.Index([None, "a", None, "b"], name="my_index"))
    (Index(['a', 'b'], dtype='object', name='my_index'), array([1, 3]), array([0, 1]), array([1, 2]))

    """
    # factorize by to groups and codes
    groups, codes = factor_by(by, sort=True)

    # exclude missing
    keep = codes >= 0
    if not np.all(keep):
        index = np.where(keep)[0]
        codes = codes[keep]
    else:
        index = None

    indexes_sorted = np.argsort(codes, **kwargs)
    group_idx_sorted = codes[indexes_sorted]
    _groups, n_start, count = np.unique(
        group_idx_sorted, return_index=True, return_counts=True
    )
    n_end = n_start + count

    if index is not None:
        indexes_sorted = index[indexes_sorted]

    return groups, indexes_sorted, n_start, n_end
