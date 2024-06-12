# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false


import itertools
from unittest.mock import call, patch

import pytest


def _add_parameter(param, flag, args, expected):
    if param is None:
        return args, expected

    args = [*args, f"--{flag}" if param else f"--no-{flag}"]

    expected = expected.copy()
    key = f"include_{flag}"
    expected[key] = param

    return args, expected


@pytest.fixture()
def import_module():
    with patch("cmomy.compile.import_module") as mocked:
        yield mocked


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            [],
            {
                "include_all": True,
                "include_cov": None,
                "include_parallel": None,
                "include_vec": None,
                "include_resample": None,
                "include_indexed": None,
                "include_convert": None,
            },
        ),
        (
            ["--no-all"],
            {
                "include_all": False,
                "include_cov": None,
                "include_parallel": None,
                "include_vec": None,
                "include_resample": None,
                "include_indexed": None,
                "include_convert": None,
            },
        ),
    ],
)
@pytest.mark.parametrize("cov", [True, False, None])
@pytest.mark.parametrize("others", [True, False, None])
def test__parser(args, expected, cov, others, import_module) -> None:  # noqa: PLR0914
    from cmomy.compile import _main, _parser_args

    args, expected = _add_parameter(cov, "cov", args, expected)
    args, expected = _add_parameter(others, "parallel", args, expected)
    args, expected = _add_parameter(others, "vec", args, expected)
    args, expected = _add_parameter(others, "resample", args, expected)
    args, expected = _add_parameter(others, "indexed", args, expected)
    args, expected = _add_parameter(others, "convert", args, expected)

    assert vars(_parser_args(args)) == expected

    all_ = expected["include_all"]
    cov = all_ if cov is None else cov
    others = all_ if others is None else others
    parallel = vec = resample = indexed = convert = others

    _main(args)

    # modules = ["utils", "_push"]
    modules = []
    if vec:
        modules.append("push")
    if resample:
        modules.append("resample")
    if indexed:
        modules.append("indexed")
    # if convert:
    #     modules.append("convert")

    covs = ["", "_cov"] if cov else [""]
    parallels = ["", "_parallel"] if parallel else [""]

    modules = itertools.chain(  # type: ignore[assignment]
        f"{m}{c}{p}" for m in modules for c in covs for p in parallels
    )

    # prepend with utils and push
    pre = ["utils", "_push"]
    if cov:
        pre.append("_push_cov")

    # append convert
    post = []
    if convert:
        post.append("convert")
        if cov:
            post.append("convert_cov")

    modules = [*pre, *modules, *post]

    prefix = "cmomy._lib"
    call_list = [call(f"{prefix}.{module}", package=None) for module in modules]

    assert import_module.call_args_list == call_list
