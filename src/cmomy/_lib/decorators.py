from __future__ import annotations

import itertools
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numba as nb
from numba import guvectorize, njit

from ..options import OPTIONS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from typing import Any

    from cmomy.core.typing import FuncT, NumbaType


# * Threading
def is_in_unsafe_thread_pool() -> bool:
    import threading

    current_thread = threading.current_thread()
    # ThreadPoolExecutor threads typically have names like 'ThreadPoolExecutor-0_1'
    return (
        current_thread.name.startswith("ThreadPoolExecutor")
        and _thread_backend() == "workqueue"
    ) or False


@lru_cache
def _thread_backend() -> str | None:
    # Note that `importlib.util.find_spec` doesn't work for these; it will falsely return True
    try:
        from numba.np.ufunc import (  # pylint: disable=unused-import
            tbbpool,  # noqa: F401  # pyright: ignore[reportAttributeAccessIssue, reportUnusedImport]
        )
    except ImportError:
        pass
    else:
        return "tbb"

    try:
        from numba.np.ufunc import (  # pylint: disable=unused-import
            omppool,  # noqa: F401  # pyright: ignore[reportAttributeAccessIssue, reportUnusedImport]
        )
    except ImportError:
        pass
    else:
        return "omp"

    return "workqueue"


# * Generate signatures.
INCLUDE_FLOAT32 = True


def myguvectorize(
    gufunc_sig: str,
    signature: Iterable[Sequence[NumbaType]] | None = None,
    signature_generator: Iterable[NumbaType | Sequence[NumbaType]] | None = None,
    *,
    nopython: bool = True,
    parallel: bool = False,
    cache: bool | None = None,
    fastmath: bool | None = None,
    writable: str | Iterable[str] | None = "out",
    **kwargs: Any,
) -> Callable[[FuncT], FuncT]:
    target = _get_target(parallel)

    signatures = _get_signatures(signature, signature_generator)

    args = (signatures, gufunc_sig) if signatures else (gufunc_sig,)

    if cache is None:
        cache = OPTIONS["cache"]
    if fastmath is None:
        fastmath = OPTIONS["fastmath"]
    if writable:
        kwargs["writable_args"] = (
            (writable,) if isinstance(writable, str) else tuple(writable)
        )

    return cast(
        "Callable[[FuncT], FuncT]",
        guvectorize(
            *args,
            nopython=nopython,
            target=target,
            fastmath=fastmath,
            cache=cache,
            **kwargs,
        ),
    )


def myjit(
    signature: Iterable[Sequence[NumbaType]] | None = None,
    signature_generator: Iterable[NumbaType | Sequence[NumbaType]] | None = None,
    *,
    parallel: bool = False,
    fastmath: bool | None = None,
    cache: bool | None = None,
    inline: bool | None = None,
    **kwargs: Any,
) -> Callable[[FuncT], FuncT]:
    """Perform jitting."""
    signatures = _get_signatures(signature, signature_generator)

    args = (signatures,) if signatures else ()

    if cache is None:
        cache = OPTIONS["cache"]
    if fastmath is None:
        fastmath = OPTIONS["fastmath"]

    if inline is not None:
        kwargs["inline"] = "always" if inline else "never"

    return cast(
        "Callable[[FuncT], FuncT]",
        njit(*args, fastmath=fastmath, cache=cache, parallel=parallel, **kwargs),
    )


def _get_target(parallel: bool) -> str:
    if not parallel:
        return "cpu"
    if is_in_unsafe_thread_pool():
        warnings.warn(
            "Detected unsafe thread pool.  Falling back to non-parallel", stacklevel=1
        )
        return "cpu"
    return "parallel"


def _get_signatures(
    signature: Iterable[Sequence[NumbaType]] | None,
    signature_generator: Iterable[NumbaType | Sequence[NumbaType]] | None,
) -> list[tuple[NumbaType, ...]]:
    signatures = [] if signature is None else [tuple(x) for x in signature]

    if signature_generator is not None:
        signatures.extend(
            itertools.product(
                *(
                    (x,) if isinstance(x, (nb.types.Integer, nb.types.Array)) else x
                    for x in signature_generator
                )
            )
        )

    if INCLUDE_FLOAT32:
        return signatures

    # limit to only float64:
    def _filter_float32(sig: Any) -> bool:
        x = sig[0]
        return (x == nb.float64) or (hasattr(x, "dtype") and x.dtype == nb.float64)

    return [s for s in signatures if _filter_float32(s)]
