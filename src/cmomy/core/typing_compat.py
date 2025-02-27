"""Typing compatibility."""

import sys
from typing import Any

if sys.version_info >= (3, 10):
    from types import EllipsisType
    from typing import TypeAlias, TypeGuard
else:
    EllipsisType = Any
    from typing_extensions import TypeAlias, TypeGuard


if sys.version_info >= (3, 11):
    from typing import Required, Self, Unpack
else:
    from typing_extensions import Required, Self, Unpack


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
    "Unpack",
]
