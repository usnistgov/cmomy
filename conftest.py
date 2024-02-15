# flake8: noqa
import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace):
    import numpy as np
    import pandas as pd
    import xarray as xr

    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd
    doctest_namespace["xr"] = xr

    np.set_printoptions(precision=4)


def pytest_collectstart(collector):
    if collector.fspath and collector.fspath.ext == ".ipynb":
        collector.skip_compare += (
            "text/html",
            "application/javascript",
            "stderr",
        )
