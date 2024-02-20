"""Common docstrings."""
from __future__ import annotations

from module_utilities.docfiller import DocFiller


def _dummy_docstrings() -> None:
    """
    Parameters
    ----------
    copy : bool, optional
        If True, copy the data.  If False, attempt to use view.
    copy_kws : mapping, optional
        extra arguments to copy
    verify : bool, optional
        If True, make sure data is c-contiguous.
    mom : int or tuple of int
        Order or moments.  If integer or length one tuple, then moments are for
        a single variable.  If length 2 tuple, then comoments of two variables
    mom_ndim : {1, 2}
        Value indicates if moments (``mom_ndim = 1``) or comoments (``mom_ndim=2``).
    val_shape : tuple, optional
        Shape of `values` part of data.  That is, the non-moment dimensions.
    shape : tuple, optional
        Total shape.  ``shape = val_shape + tuple(m+1 for m in mom)``
    dtype : dtype, optional
        Optional ``dtype`` for output data.
    zeros_kws : mapping, optional
        Optional parameters to :func:`numpy.zeros`
    axis : int
        Axis to reduce along.
    broadcast : bool, optional
        If True, and ``x=(x0, x1)``, then perform 'smart' broadcasting.
        In this case, if ``x1.ndim = 1`` and ``len(x1) == x0.shape[axis]``, then
        broadcast `x1` to ``x0.shape``.
    freq : array of int, optional
        Array of shape ``(nrep, size)`` where `nrep` is the number of replicates and
        ``size = self.shape[axis]``.  `freq` is the weight that each sample contributes
        to resamples values.  See :func:`~cmomy.resample.randsamp_freq`
    indices : array of int, optional
        Array of shape ``(nrep, size)``.  If passed, create `freq` from indices.
        See :func:`~cmomy.resample.randsamp_freq`.
    nrep : int, optional
        Number of replicates.  Create `freq` with this many replicates.
        See :func:`~cmomy.resample.randsamp_freq`
    pushed : object
        Same as object, with new data pushed onto `self.data`
    resample_kws : mapping
        Extra arguments to :func:`~cmomy.resample.resample_vals`
    full_output : bool, optional
        If True, also return `freq` array
    convert_kws : mapping
        Extra arguments to :func:`~cmomy.convert.to_central_moments` or
        :func:`~cmomy.convert.to_central_comoments`
    dims : hashable or sequence of hashable, optional
        Dimension of resulting :class:`xarray.DataArray`.

        * If ``len(dims) == self.ndim``, then dims specifies all dimensions.
        * If ``len(dims) == self.val_ndim``, ``dims = dims + mom_dims``

        Default to ``('dim_0', 'dim_1', ...)``
    mom_dims : hashable or tuple of hashable
        Name of moment dimensions.  Defaults to ``('xmom', 'umom')``
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
    dim : hashable, optional
        Dimension to reduce along.
    rep_dim : hashable, optional
        Name of new 'replicated' dimension:
    rec_dim : hashable, optional
        Name of dimension for 'records', i.e., multiple observations.
    data : DataArray or ndarray
        Moment collection array
    parallel : bool, default=True
        flags to `numba.njit`
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator object.  Defaults to output of :func:`~cmomy.random.default_rng`.
    kwargs | **kwargs
        Extra keyword arguments.
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
