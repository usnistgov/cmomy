"""Public api for :mod:`cmomy`"""
# Top level API (:mod:`cmomy`)
# ============================
# """

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

# To change top level imports edit __init__.pyi
import lazy_loader as _lazy  # pyright: ignore[reportMissingTypeStubs]

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)  # pyright: ignore[reportUnknownVariableType]

try:
    __version__ = _version("cmomy")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"
