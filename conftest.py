from functools import lru_cache

import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace) -> None:
    import numpy as np
    import pandas as pd
    import xarray as xr

    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd
    doctest_namespace["xr"] = xr

    np.set_printoptions(precision=4)


def pytest_collectstart(collector) -> None:
    if collector.fspath and collector.fspath.ext == ".ipynb":
        collector.skip_compare += (
            "text/html",
            "application/javascript",
            "stderr",
        )


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--typing", action="store_true", default=False, help="run tests on typing"
    )
    parser.addoption(
        "--compile",
        action="store_true",
        help="""
        Run full compile before testing.
        This means all the numba modules will be pre-loaded before test run.
        Makes timing/benchmarking consistent.
        """,
    )


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "typing: mark that test if for typing")
    if config.getoption("--compile"):
        from cmomy.compile import load_numba_modules

        load_numba_modules()


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run")
    if "typing" in item.keywords and not item.config.getoption("--typing"):
        pytest.skip("need --typing option to run")


@lru_cache
def _is_default_version() -> bool:
    import sys
    from pathlib import Path

    return sys.version_info[:2] == tuple(
        map(int, Path(".python-version").read_text(encoding="utf-8").strip().split("."))
    )


def pytest_ignore_collect(collection_path) -> bool:
    if not _is_default_version():
        return "cmomy/tests" not in str(collection_path)

    return False
