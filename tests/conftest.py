# mypy: disable-error-code="no-untyped-def, no-untyped-call"
from __future__ import annotations

import numpy as np
import pytest

import cmomy.random
import cmomy.reduction
import cmomy.resample

default_rng = cmomy.default_rng(0)


@pytest.fixture(scope="session")
def rng():
    return default_rng


@pytest.fixture(params=[True, False])
def as_dataarray(request):
    return request.param


@pytest.fixture(scope="session")
def get_zero_rng():
    return lambda: np.random.default_rng(0)
