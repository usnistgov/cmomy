"""Common docstrings."""

from __future__ import annotations

from ._docfiller.docfiller import DocFiller

_docstring_cmomy = """\
Parameters
----------
copy : bool, optional
    If True, copy the data.  If False, attempt to use view.
copy_kws : mapping, optional
    extra arguments to copy
verify : bool, optional
    If True, make sure data is c-contiguous
check_shape : bool, optional
    If True, check that shape of resulting object is correct.
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
pushed : same as object
    Same as object, with new data pushed onto `self.data`
resample_kws : mapping
    Extra arguments to :func:`~cmomy.resample.resample_vals`
full_output : bool, optional
    If True, also return `freq` array
convert_kws : mapping
    Extra arguments to :func:`~cmomy.convert.to_central_moments` or :func:`~cmomy.convert.to_central_comoments`
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
"""

DOCFILLER_CMOMY = DocFiller.from_docstring(
    _docstring_cmomy, combine_keys="parameters"
).assign_combined_key(
    "xr_params", ["dims", "attrs", "coords", "name", "indexes", "template"]
)


docfiller_shared = DOCFILLER_CMOMY()
