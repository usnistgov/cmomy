"""
Pre-compile ``numba`` based core routines (:mod:`cmomy.compile`)
===================================================================

This can be called from python using :func:``load_numba_modules`` or from
the command line with

.. code-block:: console

    python -m cmomy.compile [--help]

"""

# pylint: disable=missing-class-docstring
from __future__ import annotations

import itertools
import logging
from importlib import import_module
from time import perf_counter
from typing import TYPE_CHECKING

from .core.typing_compat import override

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from typing import Any

    from .core.typing_compat import Self


FORMAT = "[%(name)s] %(message)s"
logging.basicConfig(level=logging.WARNING, format=FORMAT)
logger = logging.getLogger("cmomy.compile")


class _Catchtime:
    def __init__(self) -> None:
        self.start: float = 0
        self.time: float = 0

    def __enter__(self) -> Self:
        self.start = perf_counter()
        return self

    def __exit__(self, *args: object, **kwargs: object) -> None:
        self.time = perf_counter() - self.start


def _time_modules(
    *modules: str, prefix: str | None = None, package: str | None = None
) -> None:
    from operator import itemgetter

    prefix = "" if prefix is None else f"{prefix}."

    out: list[tuple[Any, Any]] = []
    with _Catchtime() as total:
        for module in modules:
            logger.warning("loading mod %s", module)
            with _Catchtime() as t:
                _ = import_module(f"{prefix}{module}", package=package)
            out.append((t.time, module))
    for time, module in sorted(out, key=itemgetter(0), reverse=True):
        logger.warning("%7.1E sec %s", time, module)
    logger.warning("%7.1E sec total", total.time)


def load_numba_modules(
    include_all: bool = True,
    include_cov: bool | None = None,
    include_parallel: bool | None = None,
    include_vec: bool | None = None,
    include_resample: bool | None = None,
    include_grouped: bool | None = None,
    include_convert: bool | None = None,
    include_rolling: bool | None = None,
) -> None:
    """
    Compile numba modules by loading them.

    Default is to include all submodules. To override, set
    ``include_all=False``, and select other packages manually. For example, to
    only load resampling with with co-moments, use
    ``load_numba_modules(include_all=False, include_cov=True,
    include_resample=True)``.

    """
    from pathlib import Path

    from .factory import supports_parallel

    def _set_default(x: bool | None) -> bool:
        return include_all if x is None else x

    include_cov = _set_default(include_cov)
    include_vec = _set_default(include_vec)
    include_resample = _set_default(include_resample)
    include_grouped = _set_default(include_grouped)
    include_parallel = _set_default(include_parallel) if supports_parallel() else False
    include_convert = _set_default(include_convert)
    include_rolling = _set_default(include_rolling)

    covs = ["", "_cov"] if include_cov else [""]
    parallels = ["", "_parallel"] if include_parallel else [""]

    modules = ["utils", "_push"]
    if include_vec:
        modules.append("push")
    if include_resample:
        modules.append("resample")
    if include_grouped:
        modules.append("grouped")
    if include_rolling:
        modules.append("rolling")
    if include_convert:
        modules.append("convert")

    mods = itertools.chain(
        (
            f"{mod}{cov}{parallel}"
            for mod in modules
            for cov in covs
            for parallel in parallels
        ),
    )

    # filter to those that exist
    root = Path(__file__).parent

    def _filter(name: str) -> bool:
        return (root / "_lib" / name).with_suffix(".py").exists()

    _time_modules(*(m for m in mods if _filter(m)), prefix="cmomy._lib")


def _parser_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    from argparse import Action, ArgumentParser, Namespace

    class BooleanAction(Action):
        def __init__(
            self,
            option_strings: str,
            dest: str,
            **kwargs: Any,
        ) -> None:
            super().__init__(option_strings, dest, nargs=0, **kwargs)

        @override
        def __call__(
            self,
            parser: ArgumentParser,  # noqa: ARG002
            namespace: Namespace,
            values: object,  # noqa: ARG002
            option_string: str | None = None,
        ) -> None:
            setattr(
                namespace,
                self.dest,
                not option_string.startswith("--no")
                if option_string is not None
                else False,
            )

    parser = ArgumentParser(description="Load numba modules.")
    msg = "Load or ignore {name} routines.  Default is to include if not `--no-all`. "

    _ = parser.add_argument(
        "--no-all",
        dest="include_all",
        action="store_false",
        help="Load all numba modules",
    )
    _ = parser.add_argument(
        "--cov",
        "--no-cov",
        dest="include_cov",
        action=BooleanAction,
        default=None,
        help=msg.format(name="comoment"),
    )
    _ = parser.add_argument(
        "--parallel",
        "--no-parallel",
        dest="include_parallel",
        action=BooleanAction,
        default=None,
        help=msg.format(name="parallel"),
    )
    _ = parser.add_argument(
        "--vec",
        "--no-vec",
        dest="include_vec",
        action=BooleanAction,
        default=None,
        help=msg.format(name="vector"),
    )
    _ = parser.add_argument(
        "--resample",
        "--no-resample",
        dest="include_resample",
        action=BooleanAction,
        default=None,
        help=msg.format(name="resample"),
    )
    _ = parser.add_argument(
        "--grouped",
        "--no-grouped",
        dest="include_grouped",
        action=BooleanAction,
        default=None,
        help=msg.format(name="grouped"),
    )
    _ = parser.add_argument(
        "--rolling",
        "--no-rolling",
        dest="include_rolling",
        action=BooleanAction,
        default=None,
        help=msg.format(name="rolling"),
    )
    _ = parser.add_argument(
        "--convert",
        "--no-convert",
        dest="include_convert",
        action=BooleanAction,
        default=None,
        help=msg.format(name="convert"),
    )
    return parser.parse_args() if args is None else parser.parse_args(args)


def _main(args: Sequence[str] | None = None) -> int:
    options = _parser_args(args)
    load_numba_modules(**vars(options))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_main())
