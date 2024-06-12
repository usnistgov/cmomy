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
    if config.getoption("--compile"):
        from cmomy.compile import load_numba_modules

        load_numba_modules()


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--run-slow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_ignore_collect(collection_path) -> None:
    import sys

    if sys.version_info < (3, 9):
        return "cmomy/tests" not in str(collection_path)

    return False
