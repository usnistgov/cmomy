"""Typing compatibility."""

import sys
from typing import Any

if sys.version_info < (3, 10):
    EllipsisType = Any
    from typing_extensions import TypeAlias, TypeGuard
else:
    from types import EllipsisType
    from typing import TypeAlias, TypeGuard


if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


if sys.version_info < (3, 13):
    from typing_extensions import TypeVar
else:  # pragma: no cover
    from typing import TypeVar


__all__ = [
    "EllipsisType",
    "Self",
    "TypeAlias",
    "TypeGuard",
    "TypeVar",
]
