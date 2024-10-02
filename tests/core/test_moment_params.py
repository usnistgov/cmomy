# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import pytest

from cmomy.core.moment_params import MomParamsXArray, MomParamsXArrayOptional


# * catch all args only test
def _do_test(func, *args, expected=None, match=None, **kwargs):
    if isinstance(expected, type):
        with pytest.raises(expected, match=match):
            func(*args, **kwargs)
    else:
        assert func(*args, **kwargs) == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((None, 1), (("mom_0",), 1)),
        ((None, 2), (("mom_0", "mom_1"), 2)),
        (("a", None), (("a",), 1)),
        ((("a", "b"), None), (("a", "b"), 2)),
        ((["a"], None), (("a",), 1)),
        ((["a", "b"], None), (("a", "b"), 2)),
        ((["a", "b"], 1), ValueError),
        (("a", 2), ValueError),
        ((("a,"), 2), ValueError),
        ((None, None), (None, None)),
        ((None, None, None, 1), (("mom_0",), 1)),
    ],
)
def test_MomParamsXArrayOptional(args, expected) -> None:
    def _func(*args, **kwargs):
        m = MomParamsXArrayOptional.factory(*args, **kwargs)
        return m.dims, m.ndim

    kws = dict(zip(["dims", "ndim", "axes", "default_ndim"], args))
    _do_test(_func, expected=expected, **kws)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((None, 1), (("mom_0",), 1)),
        ((None, 2), (("mom_0", "mom_1"), 2)),
        (("a", None), (("a",), 1)),
        ((("a", "b"), None), (("a", "b"), 2)),
        ((["a"], None), (("a",), 1)),
        ((["a", "b"], None), (("a", "b"), 2)),
        ((["a", "b"], 1), ValueError),
        (("a", 2), ValueError),
        ((("a,"), 2), ValueError),
        ((None, None), ValueError),
        ((None, None, None, 1), (("mom_0",), 1)),
    ],
)
def test_MomParamsXArray(args, expected) -> None:
    def _func(*args, **kwargs):
        m = MomParamsXArray.factory(*args, **kwargs)
        return m.dims, m.ndim

    kws = dict(zip(["dims", "ndim", "axes", "default_ndim"], args))
    _do_test(_func, expected=expected, **kws)
