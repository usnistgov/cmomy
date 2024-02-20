# mypy: disable-error-code="no-untyped-def, no-untyped-call"
import numpy as np
import pytest
import xarray as xr

import cmomy


@pytest.fixture(scope="module")
def data(rng):
    out = rng.random((1, 2, 3, 4))
    return np.moveaxis(out, [2, 1], [1, 2])


@pytest.fixture(scope="module")
def xdata(data):
    return xr.DataArray(data)


def get_c_contig(data):
    return data.flags["C_CONTIGUOUS"]


def test_data(data, xdata) -> None:
    assert get_c_contig(data) is False
    assert xdata.to_numpy() is data


@pytest.fixture(params=[1, 2])
def mom_ndim(request):
    return request.param


# These tests don't make a lot of sense anymore...
@pytest.mark.parametrize(
    ("copy", "verify", "copy_order", "same_data", "c_contig"),
    [
        (False, False, "C", True, False),
        (True, False, "K", False, False),
        (True, False, "C", False, True),
        (False, True, None, False, True),
    ],
)
def test_Central(data, mom_ndim, copy, verify, copy_order, same_data, c_contig) -> None:  # noqa: ARG001
    # no copy no verify should yield same thing this

    # with pytest.raises(ValueError):
    c = cmomy.CentralMoments.from_data(
        data,
        mom_ndim=mom_ndim,
        copy=copy,
        copy_kws={"order": copy_order},
        verify=verify,
    )

    assert np.shares_memory(c.data, c._data_flat)

    # print(f"hello, {data.shape=}, {mom_ndim=}, {copy=}, {verify=}, {copy_order=}, share0={np.shares_memory(c.data, data)}, share1={np.shares_memory(c.data, c._data_flat)}, c_contig={get_c_contig(c.data)}, {c_contig=}")

    #     assert (c.data is data) is same_data
    #     assert get_c_contig(c.data) is c_contig

    # except ValueError as e:
    #     print(e)
    #     raise e


@pytest.mark.parametrize(
    ("copy", "verify", "copy_order", "same_data", "c_contig"),
    [
        (False, False, "C", True, False),
        (True, False, "K", False, False),
        (True, False, "C", False, True),
        (False, True, None, False, True),
    ],
)
def test_xCentral(
    data,
    mom_ndim,
    copy,
    verify,
    copy_order,
    same_data,  # noqa: ARG001
    c_contig,  # noqa: ARG001
) -> None:
    # no copy no verify should yield same thing this
    c = cmomy.xCentralMoments.from_data(
        data,
        mom_ndim=mom_ndim,
        copy=copy,
        copy_kws={"order": copy_order},
        verify=verify,
    )
    assert np.shares_memory(c.data, c._data_flat)
    assert np.shares_memory(c.data, c._xdata.to_numpy())
    assert np.shares_memory(c.data, c._xdata.variable._data)


#     assert c.data is c.to_numpy()
#     assert (c.data is data) is same_data
#     assert get_c_contig(c.data) is c_contig


@pytest.mark.parametrize(
    ("copy", "verify", "copy_order", "same_data", "c_contig", "same_xdata"),
    [
        (False, False, "C", True, False, True),
        (True, False, "K", False, False, False),
        (True, False, "C", False, True, False),
        (False, True, None, False, True, False),
    ],
)
def test_xCentral_xdata(
    xdata,
    mom_ndim,
    copy,
    verify,
    copy_order,
    same_data,  # noqa: ARG001
    c_contig,  # noqa: ARG001
    same_xdata,  # noqa: ARG001
) -> None:
    # no copy no verify should yield same thing this
    c = cmomy.xCentralMoments.from_data(
        xdata,
        mom_ndim=mom_ndim,
        copy=copy,
        copy_kws={"order": copy_order},
        verify=verify,
    )

    assert np.shares_memory(c.data, c._data_flat)
    assert np.shares_memory(c.data, c._xdata.to_numpy())
    assert np.shares_memory(c.data, c._xdata.variable._data)


#     assert c.data is c.to_numpy()

#     assert (c.data is xdata.to_numpy()) is same_data
#     assert (c.to_dataarray() is xdata) is same_xdata
#     assert get_c_contig(c.data) is c_contig
