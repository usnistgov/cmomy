import sys

if sys.version_info < (3, 10):
    from typing_extensions import TypeAlias, TypeGuard
else:
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
    "Self",
    "TypeAlias",
    "TypeGuard",
    "TypeVar",
]
