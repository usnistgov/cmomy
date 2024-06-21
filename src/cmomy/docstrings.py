"""Common docstrings."""

from __future__ import annotations

from module_utilities.docfiller import DocFiller


def _dummy_docstrings() -> None:
    """
    Parameters
    ----------
    copy : bool, optional
        If True, copy the data. If None or False, attempt to use view. Note
        that ``False`` values will be converted to ``None`` for numpy versions
        ``>2.0``. This will be changed to reflect the new behavior of the
        ``copy`` parameter to :func:`numpy.array` when the minimum numpy
        version ``>2.0``.
    copy_tf | copy : bool
        If ``True``, copy the data.  If False, return a view if possible.
    copy_kws : mapping
        extra arguments to copy
    verify : bool
        If True, make sure data is c-contiguous.
    mom : int or tuple of int
        Order or moments.  If integer or length one tuple, then moments are for
        a single variable.  If length 2 tuple, then comoments of two variables
    mom_moments_to_comoments | mom : tuple of int
        Moments for comoments array. Pass a negative value for one of the
        moments to fill all available moments for that dimensions. For example,
        if original array has moments `m` (i.e., ``values.shape=(..., m +
        1)``), and pass in ``mom = (2, -1)``, then this will be transformed to
        ``mom = (2, m - 2)``.
    mom_ndim : {1, 2}
        Value indicates if moments (``mom_ndim = 1``) or comoments (``mom_ndim=2``).
    val_shape : tuple
        Shape of `values` part of data.  That is, the non-moment dimensions.
    shape : tuple
        Total shape.  ``shape = val_shape + tuple(m+1 for m in mom)``
    dtype : dtype
        Optional :class:`~numpy.dtype` for output data.
    zeros_kws : mapping
        Optional parameters to :func:`numpy.zeros`
    axis : int
        Axis to reduce along.
    axis_data | axis : int, optional
        Axis to reduce along. Note that negative values are relative to
        ``data.ndim - mom_ndim``. It is assumed that the last dimensions are
        for moments. For example, if ``data.shape == (1,2,3)`` with
        ``mom_ndim=1``, ``axis = -1 `` would be equivalent to ``axis = 1``.
        Defaults to ``axis=-1``.
    broadcast : bool
        If True, and ``x=(x0, x1)``, then perform 'smart' broadcasting.
        In this case, if ``x1.ndim = 1`` and ``len(x1) == x0.shape[axis]``, then
        broadcast `x1` to ``x0.shape``.
    freq : array of int
        Array of shape ``(nrep, size)`` where `nrep` is the number of replicates and
        ``size = self.shape[axis]``.  `freq` is the weight that each sample contributes
        to resamples values.  See :func:`~cmomy.resample.randsamp_freq`
    indices : array of int
        Array of shape ``(nrep, size)``.  If passed, create `freq` from indices.
        See :func:`~cmomy.resample.randsamp_freq`.
    nrep : int
        Number of resample replicates.
    nsamp : int
        Number of samples in a single resampled replicate. Defaults to size of
        data along sampled axis.
    ndat : int
        Size of data along resampled axis.
    pushed : object
        Same as object, with new data pushed onto `self.data`
    resample_kws : mapping
        Extra arguments to :func:`~cmomy.resample.resample_vals`
    full_output : bool
        If True, also return ``freq`` array
    convert_kws : mapping
        Extra arguments to :func:`~cmomy.convert.to_central_moments` or
        :func:`~cmomy.convert.to_central_comoments`
    dims : hashable or sequence of hashable
        Dimension of resulting :class:`xarray.DataArray`.

        * If ``len(dims) == self.ndim``, then dims specifies all dimensions.
        * If ``len(dims) == self.val_ndim``, ``dims = dims + mom_dims``

        Default to ``('dim_0', 'dim_1', ...)``
    mom_dims : hashable or tuple of hashable
        Name of moment dimensions. Defaults to ``("mom_0",)`` for
        ``mom_ndim==1`` and ``(mom_0, mom_1)`` for ``mom_ndim==2``
    attrs : mapping
        Attributes of output
    coords : mapping
        Coordinates of output
    name : hashable
        Name of output
    indexes : Any
        indexes attribute.  This is ignored.
    template : DataArray
        If present, output will have attributes of `template`.
        Overrides other options.
    dim : hashable
        Dimension to reduce along.
    rep_dim : hashable
        Name of new 'replicated' dimension:
    rec_dim : hashable
        Name of dimension for 'records', i.e., multiple observations.
    data : DataArray or ndarray
        Moment collection array
    data_numpy | data : ndarray
        Moments collection array.  It is assumed moment dimensions are last.
    data_numpy_or_dataarray | data : ndarray or DataArray
        Moments collection array.  It is assumed moment dimensions are last.
    parallel : bool, default=True
        flags to `numba.njit`
    rng : :class:`~numpy.random.Generator`
        Random number generator object.  Defaults to output of :func:`~cmomy.random.default_rng`.
    kwargs | **kwargs
        Extra keyword arguments.

    coords_policy : {'first', 'last', 'group', None}
        Policy for handling coordinates along ``dim`` if ``by`` is specified
        for :class:`~xarray.DataArray` data.
        If no coordinates do nothing, otherwise use:

        * 'first': select first value of coordinate for each block.
        * 'last': select last value of coordinate for each block.
        * 'group': Assign unique groups from ``group_idx`` to ``dim``
        * None: drop any coordinates.

        Note that if ``coords_policy`` is one of ``first`` or ``last``, parameter ``groups``
        will be ignored.
    by : array-like of int
        Groupby values of same length as ``data`` along sampled dimension.
        Negative values indicate no group (i.e., skip this index).
    group_dim : str, optional
        Name of the output group dimension.  Defaults to ``dim``.
    groups : sequence, optional
        Sequence of length ``by.max() + 1`` to assign as coordinates for ``group_dim``.
    out : ndarray
        Optional output array. If specified, output will be a reference to this
        array.
    order : {"C", "F", "A", "K"}, optional
        Order argument to :func:`numpy.asarray`.
    weight : array-like, optional
        Optional weights. Can be scalar, 1d array of length
        ``args[0].shape[axis]`` or array of same form as ``args[0]``.

    keep_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts", "override"} or bool, optional
        - 'drop' or False: empty attrs on returned xarray object.
        - 'identical': all attrs must be the same on every object.
        - 'no_conflicts': attrs from all objects are combined, any that have the same name must also have the same value.
        - 'drop_conflicts': attrs from all objects are combined, any that have the same name but different values are dropped.
        - 'override' or True: skip comparing and copy attrs from the first object to the result.

    """


def _dummy_docstrings_central() -> None:
    """
    Parameters
    ----------
    data : ndarray
        Moment collection array
    """


def _dummy_docstrings_xcentral() -> None:
    """
    Parameters
    ----------
    data : DataArray
        Moment collection array
    """


_docstrings = _dummy_docstrings.__doc__ or ""

docfiller = (
    DocFiller.from_docstring(_docstrings, combine_keys="parameters")
    .assign_combined_key(
        "xr_params", ["dims", "attrs", "coords", "name", "indexes", "template"]
    )
    .assign_combined_key(
        "xr_params_complete",
        ["dims", "attrs", "coords", "name", "indexes", "template", "mom_dims"],
    )
    .assign(
        klass="object", t_array=":class:`numpy.ndarray` or :class:`xarray.DataArray`"
    )
    .assign_combined_key("axis_and_dim", ["axis"])
    .assign_combined_key("axis_data_and_dim", ["axis_data"])
)


docfiller_central = (
    docfiller.update(
        DocFiller.from_docstring(
            _dummy_docstrings_central,
            combine_keys="parameters",
        ).data
    )
    .assign(
        klass="CentralMoments",
        t_array=":class:`numpy.ndarray`",
    )
    .assign_combined_key("axis_and_dim", ["axis"])
    .assign_combined_key("axis_data_and_dim", ["axis_data"])
)


docfiller_xcentral = (
    docfiller.update(
        DocFiller.from_docstring(
            _dummy_docstrings_xcentral,
            combine_keys="parameters",
        ).data
    )
    .assign(
        klass="xCentralMoments",
        t_array=":class:`xarray.DataArray`",
    )
    .assign_combined_key("axis_and_dim", ["axis", "dim"])
    .assign_combined_key("axis_data_and_dim", ["axis_data", "dim"])
)


docfiller_decorate = docfiller()


# --- Factory functions ----------------------------------------------------------------
# from typing import Any, Callable, cast

# from custom_inherit import doc_inherit

# from .typing import F
# from .options import DOC_SUB


# def _my_doc_inherit(parent, style) -> Callable[[F], F]:
#     if DOC_SUB:
#         return cast(Callable[[F], F], doc_inherit(parent=parent, style=style))
#     else:

#         def wrapper(func: F) -> F:
#             return func

#         return wrapper


# def factory_docfiller_from_parent(
#     cls: Any, docfiller: DocFiller
# ) -> Callable[..., Callable[[F], F]]:
#     """Decorator with docfiller inheriting from cls"""

#     def decorator(*name: str, **params) -> Callable[[F], F]:
#         if len(name) == 0:
#             _name = None
#         elif len(name) == 1:
#             _name = name[0]
#         else:
#             raise ValueError("can only pass a single name")

#         def decorated(method: F) -> F:
#             template = getattr(cls, _name or method.__name__)
#             return docfiller(template, **params)(method)

#         return decorated

#     return decorator


# def factory_docinherit_from_parent(
#     cls: Any, style="numpy_with_merge"
# ) -> Callable[..., Callable[[F], F]]:
#     """Create decorator inheriting from cls"""

#     def decorator(name: str | None = None) -> Callable[[F], F]:
#         def decorated(method: F) -> F:
#             template = getattr(cls, name or method.__name__)
#             return cast(F, _my_doc_inherit(parent=template, style=style)(method))

#         return decorated

#     return decorator


# def factory_docfiller_inherit_from_parent(
#     cls: Any, docfiller: DocFiller, style="numpy_with_merge"
# ) -> Callable[..., Callable[[F], F]]:
#     """
#     Do combination of doc_inherit and docfiller

#     1. Fill parent and child with docfiller (from this module).
#     2. Merge using doc_inherit
#     """

#     def decorator(*name: str, **params) -> Callable[[F], F]:
#         if len(name) == 0:
#             _name = None
#         elif len(name) == 1:
#             _name = name[0]
#         else:
#             raise ValueError("can only pass a single name")

#         def decorated(method: F) -> F:
#             template = getattr(cls, _name or method.__name__)

#             @docfiller(template, **params)
#             def dummy():
#                 pass

#             method = docfiller(**params)(method)
#             return cast(
#                 F, _my_doc_inherit(parent=dummy, style="numpy_with_merge")(method)
#             )

#         return decorated

#     return decorator
