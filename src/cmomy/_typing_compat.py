import sys

if sys.version_info < (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias


if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


__all__ = [
    "Self",
    "TypeAlias",
]
