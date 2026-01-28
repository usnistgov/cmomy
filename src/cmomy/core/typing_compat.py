"""Typing compatibility."""
# pyright: reportUnreachable=false

import sys
from types import EllipsisType
from typing import TypeAlias, TypeGuard

from typing_extensions import TypedDict

if sys.version_info >= (3, 11):
    from typing import Required, Self, Unpack
else:
    from typing_extensions import Required, Self, Unpack


if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


if sys.version_info >= (3, 13):  # pragma: no cover
    from typing import TypeIs, TypeVar
else:
    from typing_extensions import TypeIs, TypeVar


__all__ = [
    "EllipsisType",
    "Required",
    "Self",
    "TypeAlias",
    "TypeGuard",
    "TypeIs",
    "TypeVar",
    "TypedDict",
    "Unpack",
    "override",
]
