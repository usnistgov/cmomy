# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import pytest

from cmomy.core import utils
from cmomy.core.validate import validate_mom


# * catch all args only test
def _do_test(func, *args, expected=None, match=None, **kwargs):
    if isinstance(expected, type):
        with pytest.raises(expected, match=match):
            func(*args, **kwargs)
    else:
        assert func(*args, **kwargs) == expected


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ((1, 2), (1, (1, 2))),
        (("a", "b"), ("a", ("a", "b"))),
    ],
)
def test_peek_at(arg, expected):
    def func(x):
        xx, args = utils.peek_at(x)
        return xx, tuple(args)

    _do_test(func, arg, expected=expected)


# * Order validation
@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        (2, 1),
        ((2, 2), 2),
        ((2, 2, 2), ValueError),
        ([2, 2], 2),
    ],
)
def test_mom_to_mom_ndim(arg, expected) -> None:
    _do_test(utils.mom_to_mom_ndim, arg, expected=expected)


@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        ({"mom": 2, "mom_ndim": None}, 1),
        ({"mom": (2, 2), "mom_ndim": None}, 2),
        ({"mom": (2, 2), "mom_ndim": 1}, ValueError),
        ({"mom": None, "mom_ndim": None}, TypeError),
        ({"mom": None, "mom_ndim": 1}, 1),
        ({"mom": None, "mom_ndim": 3}, ValueError),
    ],
)
def test_select_mom_ndim(kws, expected) -> None:
    _do_test(utils.select_mom_ndim, expected=expected, **kws)


@pytest.mark.parametrize(
    ("mom", "mom_shape"),
    [
        (1, (2,)),
        ((1,), (2,)),
        ((1, 2), (2, 3)),
    ],
)
def test_mom_to_mom_shape(mom, mom_shape) -> None:
    assert utils.mom_to_mom_shape(mom) == mom_shape
    assert utils.mom_shape_to_mom(mom_shape) == validate_mom(mom)
