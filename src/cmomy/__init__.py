"""Public api for :mod:`cmomy`"""
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

# To change top level imports edit __init__.pyi
import lazy_loader as _lazy

<<<<<<< before updating
__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)

try:
=======
try:  # ruff:ignore[non-empty-init-module]
>>>>>>> after updating
    __version__ = _version("cmomy")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"
