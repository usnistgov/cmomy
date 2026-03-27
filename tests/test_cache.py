from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import xarray as xr

import cmomy.cache

if TYPE_CHECKING:
    from typing import Any


def test_machine_hash() -> None:
    import platform
    import sys

    import numba

    expected = [
        "numba",
        numba.__version__,
        platform.python_implementation().lower(),
        ".".join(platform.python_version_tuple()[:-1]),
        sys.platform,
        platform.machine(),
    ]

    assert expected == cmomy.cache.MACHINE_HASH


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        (
            {},
            "0",
        ),
        (
            {"NUMBA_CACHE_TOOLS_MAGIC": "hello"},
            "hello",
        ),
        (
            {"NUMBA_CACHE_TOOLS_MAGIC": __file__},
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    ],
)
def test_config(env: dict[str, str], expected: str) -> None:
    with patch.dict(cmomy.cache.os.environ, env, clear=True):  # type: ignore[attr-defined]
        config = cmomy.cache._Config()
        assert config.magic == expected


DUMMY_FUNC = xr.get_options
DUMMY_FILE = inspect.getfile(DUMMY_FUNC)
HASH_FILE = hashlib.sha256(Path(DUMMY_FILE).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "cls",
    [
        cmomy.cache.HashCacheLocator,
        cmomy.cache.SharedHashCacheLocator,
    ],
)
def test_hashcachelocator(cls: Any) -> None:
    x = cls(DUMMY_FUNC, DUMMY_FILE)
    assert x.get_source_stamp() == (cmomy.cache.config.magic, HASH_FILE)


def test_sharedhashcachelocator() -> None:

    c = cmomy.cache.SharedHashCacheLocator(DUMMY_FUNC, DUMMY_FILE)

    assert c.get_suitable_cache_subpath(DUMMY_FILE) == "-".join([
        *("xarray", "core"),
        cmomy.cache.config.magic,
        "xarray",
        xr.__version__,
        *cmomy.cache.MACHINE_HASH,
    ])


def test_sharedhashcachelocator_no_find() -> None:
    with pytest.raises(ValueError, match=r"Can't determine .*"):
        _ = cmomy.cache.SharedHashCacheLocator(
            inspect.getfile, inspect.getfile(inspect.getfile)
        )
