from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

import cmomy.options

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


def _get_options(
    nmax: int = 20, cache: bool = True, parallel: bool = True, fastmath: bool = True
) -> dict[str, Any]:
    return {"nmax": nmax, "cache": cache, "parallel": parallel, "fastmath": fastmath}


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        pytest.param({}, _get_options(), id="None"),
        pytest.param({"CMOMY_NMAX": "30"}, _get_options(nmax=30), id="nmax"),
        pytest.param({"CMOMY_CACHE": "false"}, _get_options(cache=False), id="cache"),
        pytest.param(
            {"CMOMY_NUMBA_CACHE": "false"}, _get_options(cache=False), id="numba_cache"
        ),
        pytest.param(
            {"CMOMY_CACHE": "t", "CMOMY_NUMBA_CACHE": "false"},
            _get_options(),
            id="cache_override",
        ),
        pytest.param(
            {"CMOMY_PARALLEL": "f"},
            _get_options(parallel=False),
            id="parallel",
        ),
        pytest.param(
            {"CMOMY_FASTMATH": "f"},
            _get_options(fastmath=False),
            id="fastmath",
        ),
    ],
)
def test__get_options_from_mapping(
    env: Mapping[str, str], expected: Mapping[str, Any]
) -> None:
    assert cmomy.options._get_options_from_mapping(env) == expected


@pytest.mark.parametrize(
    ("options", "kwargs", "expected"),
    [
        pytest.param(
            _get_options(),
            {},
            _get_options(),
            id="None",
        ),
        pytest.param(
            _get_options(),
            {"nmax": 10},
            _get_options(nmax=10),
            id="nmax",
        ),
        pytest.param(
            _get_options(),
            {"parallel": False},
            _get_options(parallel=False),
            id="parallel",
        ),
        pytest.param(
            _get_options(),
            {"cache": False},
            _get_options(cache=False),
            id="parallel",
        ),
        pytest.param(
            _get_options(cache=False),
            {"nmax": 10, "parallel": False},
            _get_options(nmax=10, parallel=False, cache=False),
            id="multiple",
        ),
        pytest.param(
            _get_options(cache=False),
            {"cache": True},
            _get_options(cache=True),
            id="override",
        ),
    ],
)
def test_set_options(
    options: dict[str, Any], kwargs: dict[str, Any], expected: dict[str, Any]
) -> None:

    with patch.dict(cmomy.options.OPTIONS, options, clear=True):
        assert options == cmomy.options.OPTIONS
        with cmomy.options.set_options(**kwargs):
            assert expected == cmomy.options.OPTIONS
        assert options == cmomy.options.OPTIONS
