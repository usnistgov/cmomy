"""Load submodules for building."""

from __future__ import annotations

import itertools
import logging
from importlib import import_module
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._typing_compat import Self


FORMAT = "[%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class Catchtime:
    def __enter__(self) -> Self:
        self.start = perf_counter()
        return self

    def __exit__(self, *args: object, **kwargs: object) -> None:
        self.time = perf_counter() - self.start


def time_modules(
    *modules: str, prefix: str | None = None, package: str | None = None
) -> None:
    from operator import itemgetter

    prefix = "" if prefix is None else f"{prefix}."

    out = []
    with Catchtime() as total:
        for module in modules:
            logger.info("loading mod %s", module)
            with Catchtime() as t:
                import_module(f"{prefix}{module}", package=package)
            out.append((t.time, module))
    for time, module in sorted(out, key=itemgetter(0), reverse=True):
        logger.info("%7.1E sec %s", time, module)
    logger.info("%7.1E sec total", total.time)


def load(
    include_all: bool = True,
    include_cov: bool | None = None,
    include_parallel: bool | None = None,
    include_vec: bool | None = None,
    include_resample: bool | None = None,
    include_indexed: bool | None = None,
    include_convert: bool | None = None,
) -> None:
    from pathlib import Path

    from .utils import supports_parallel

    def _set_default(x: bool | None) -> bool:
        return include_all if x is None else x

    include_cov = _set_default(include_cov)
    include_vec = _set_default(include_vec)
    include_resample = _set_default(include_resample)
    include_indexed = _set_default(include_indexed)
    include_parallel = _set_default(include_parallel) if supports_parallel() else False
    include_convert = _set_default(include_convert)

    _covs = ["", "_cov"] if include_cov else [""]
    _parallels = ["", "_parallel"] if include_parallel else [""]

    _modules = ["pushscalar"]
    if include_vec:
        _modules.append("pushvec")
    if include_resample:
        _modules.append("resample")
    if include_indexed:
        _modules.append("reduceindexed")
    if include_convert:
        _modules.append("convert")

    mods = itertools.chain(
        # (f"pushscalar{cov}" for cov in _covs),
        (
            f"{mod}{cov}{parallel}"
            for mod in _modules
            for cov in _covs
            for parallel in _parallels
        ),
    )

    # filter to those that exist
    root = Path(__file__).parent
    mods = filter(lambda name: (root / name).with_suffix(".py").exists(), mods)

    time_modules(*mods, prefix="cmomy._lib2")
