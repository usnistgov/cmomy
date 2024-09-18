"""Common docstrings."""

from __future__ import annotations

from textwrap import dedent

from module_utilities.docfiller import DocFiller


def _dummy_docstrings() -> None:
    """
    Parameters
    ----------
    mom : int or tuple of int
        Order or moments.  If integer or length one tuple, then moments are for
        a single variable.  If length 2 tuple, then comoments of two variables
    mom_ndim : {1, 2}
        Value indicates if moments (``mom_ndim = 1``) or comoments (``mom_ndim=2``).
    mom_ndim_optional | mom_ndim : {1, 2, None}
        If ``mom_ndim`` is not ``None``, then wrap axis relative to ``mom_ndim``.
        For Example, with mom_ndim=``2``, ``axis = -1`` will be transformed to ``axis = -3``.


    copy : bool, optional
        If True, copy the data. If None or False, attempt to use view. Note
        that ``False`` values will be converted to ``None`` for numpy versions
        ``>2.0``. This will be changed to reflect the new behavior of the
        ``copy`` parameter to :func:`numpy.array` when the minimum numpy
        version ``>2.0``.
    copy_tf | copy : bool
        If ``True``, copy the data.  If False, return a view if possible.
    verify : bool
        If True, make sure data is c-contiguous.
    val_shape : tuple
        Shape of `values` part of data.  That is, the non-moment dimensions.
    fastpath : bool
        Internal variable.


    data : DataArray or ndarray
        Moment collection array
    data_numpy | data : ndarray
        Moments array.  It is assumed moment dimensions are last.
    data_numpy_or_dataarray | data : ndarray or DataArray
        Moments array.  It is assumed moment dimensions are last.
    data_numpy_or_dataarray_or_dataset | data : ndarray or DataArray or Dataset
        Moments array(s).  It is assumed moment dimensions are last.
    weight : array-like, optional
        Optional weights. Can be scalar, 1d array of length
        ``args[0].shape[axis]`` or array of same form as ``args[0]``.
    wrapped_out | wrapped : CentralMomentsArray or CentralMomentsData
        Wrapped object. If input data is an :mod:`xarray` object, then return
        :class:`.CentralMomentsData` instance. Otherwise, return
        :class:`.CentralMomentsArray` instance.

    x_genarray | x : array-like or DataArray or Dataset
        Values to reduce.
    y_genarray | *y : array-like or DataArray or Dataset
        Additional values (needed if ``len(mom)==2``).
        ``y`` has same type restrictions and broadcasting rules as ``weight``.
    weight_genarray | weight : array-like or DataArray or Dataset
        Optional weight.  The type of ``weight`` must be "less than" the type of ``x``.

        - ``x`` is :class:`~xarray.Dataset`:  ``weight`` can be a :class:`~xarray.Dataset`, :class:`~xarray.DataArray`, or array-like
        - ``x`` is :class:`~xarray.DataArray`: ``weight`` can be :class:`~xarray.DataArray` or array-like
        - ``x`` is array-like: ``weight`` can be array-like

        In the case that ``weight`` is array-like, it must broadcast to ``x``
        using usual broadcasting rules (see :func:`numpy.broadcast_to`), with the
        following exceptions: If ``weight`` is a 1d array of length
        ``x.shape[axis]]``, it will be formatted to broadcast along the other
        dimensions of ``x``. For example, if ``x`` has shape ``(10, 2, 3)`` and
        ``weight`` has shape ``(10,)``, then ``weight`` will be converted to
        the broadcastable shape ``(10, 1, 1)``. If ``weight`` is a scalar, it
        will be broadcast to ``x.shape``.


    out : ndarray
        Optional output array. If specified, output will be a reference to this
        array.  Note that if the output if method returns a :class:`~xarray.Dataset`, then this
        option is ignored.
    dtype : dtype
        Optional :class:`~numpy.dtype` for output data.
    order : {"C", "F", "A", "K"}, optional
        Order argument.  See :func:`numpy.asarray`.
    order_cf | order : {"C", "F"}, optional
        Order argument. See :func:`numpy.zeros`.
    parallel : bool, default=True
        flags to `numba.njit`
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.

        - 'no' means the data types should not be cast at all.
        - 'equiv' means only byte-order changes are allowed.
        - 'safe' means only casts which can preserve values are allowed.
        - 'same_kind' means only safe casts or casts within a kind, like float64 to float32, are allowed.
        - 'unsafe' (default) means any data conversions may be done.
    keepdims : bool
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the input array.
    move_axis_to_end : bool
        If ``True``, place sampled dimension at end (just before moments
        dimensions) in output. Otherwise, place sampled dimension at same
        position as input ``axis``. Note that if the result is a
        :class:`xarray.Dataset` object, then ``move_axis_to_end = True``
        always.


    axis : int
        Axis to reduce/sample along.
    axis_data | axis : int, optional
        Axis to reduce/sample along. Note that negative values are relative to
        ``data.ndim - mom_ndim``. It is assumed that the last dimensions are
        for moments. For example, if ``data.shape == (1,2,3)`` with
        ``mom_ndim=1``, ``axis = -1 `` would be equivalent to ``axis = 1``.
        Defaults to ``axis=-1``.
    axis_data_mult | axis : int, tuple of int, optional
        Axis(es) to reduce/sample along. Note that negative values are relative to
        ``data.ndim - mom_ndim``. It is assumed that the last dimensions are
        for moments. For example, if ``data.shape == (1,2,3)`` with
        ``mom_ndim=1``, ``axis = -1 `` would be equivalent to ``axis = 1``.
        Defaults to ``axis=-1``.  To reduce over multiple dimensions, specify
        `axis = (axis_0, axis_1, ...)`.  Passing `axis=None` reduces over all
        value dimensions (i.e., all dimensions excluding moment dimensions).


    freq : array-like of int
        Array of shape ``(nrep, size)`` where `nrep` is the number of replicates and
        ``size = self.shape[axis]``.  `freq` is the weight that each sample contributes
        to resamples values.  See :func:`.randsamp_freq`
    freq_xarray | freq : array-like, DataArray, or Dataset of int
        Array of shape ``(nrep, size)`` where `nrep` is the number of
        replicates and ``size = self.shape[axis]``. `freq` is the weight that
        each sample contributes to resamples values. If ``freq`` is an
        :mod:`xarray` object, it is assumed that the dimensions are in order of
        ``(rep_dim, dim)`` where ``rep_dim`` and ``dim`` are the name of the
        replicated and sampled dimension, respectively.
        See :func:`.randsamp_freq`
    indices : array of int
        Array of shape ``(nrep, size)``.  If passed, create `freq` from indices.
        See :func:`.randsamp_freq`.
    nrep : int
        Number of resample replicates.
    nrep_optional | nrep : int, optional
        Construct ``freq`` (see :func:`.randsamp_freq`) with ``nrep``
        replicates if ``freq`` is not passed directly.
    paired : bool
        If ``False`` and generating ``freq`` from ``nrep`` with ``data`` of type
        :class:`~xarray.Dataset`, Generate unique ``freq`` for each variable in
        ``data``. If ``True``, treat all variables in ``data`` as paired, and
        use same ``freq`` for each.
    nsamp : int
        Number of samples in a single resampled replicate. Defaults to size of
        data along sampled axis.
    ndat : int
        Size of data along resampled axis.
    rng :
        Random number generator object. Defaults to output of
        :func:`~.random.default_rng`. If pass in a seed value, create a new
        :class:`~numpy.random.Generator` object with this seed


    dims : hashable or sequence of hashable
        Dimension of resulting :class:`xarray.DataArray`.

        - If ``len(dims) == self.ndim``, then dims specifies all dimensions.
        - If ``len(dims) == self.val_ndim``, ``dims = dims + mom_dims``

        Default to ``('dim_0', 'dim_1', ...)``
    mom_dims : hashable or tuple of hashable
        Name of moment dimensions. Defaults to ``("mom_0",)`` for
        ``mom_ndim==1`` and ``(mom_0, mom_1)`` for ``mom_ndim==2``
    mom_dims_data | mom_dims : hashable or tuple of hashable
        Name of moment dimensions. Defaults to ``data.dims[-mom_ndim:]``. This
        is primarily used if ``data`` is a :class:`~xarray.Dataset`, and the
        first variable does not contain moments data.
    on_missing_core_dim : {"raise", "copy", "drop"}
        How to handle missing core dimensions on input variables.
    apply_ufunc_kwargs : dict-like
        Extra parameters to :func:`xarray.apply_ufunc`
    keep_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts", "override"} or bool, optional
        - 'drop' or False: empty attrs on returned xarray object.
        - 'identical': all attrs must be the same on every object.
        - 'no_conflicts': attrs from all objects are combined, any that have the same name must also have the same value.
        - 'drop_conflicts': attrs from all objects are combined, any that have the same name but different values are dropped.
        - 'override' or True: skip comparing and copy attrs from the first object to the result.
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
        Dimension to reduce/sample along.
    dim_mult | dim : hashable or iterable of hashable
        Dimension(s) to reduce along.  Value of `None` implies reduce over all "value" dimensions.
    rep_dim : hashable
        Name of new 'replicated' dimension:
    rec_dim : hashable
        Name of dimension for 'records', i.e., multiple observations.


    coords_policy : {'first', 'last', 'group', None}
        Policy for handling coordinates along ``dim`` if ``by`` is specified
        for :class:`~xarray.DataArray` data.
        If no coordinates do nothing, otherwise use:

        - 'first': select first value of coordinate for each block.
        - 'last': select last value of coordinate for each block.
        - 'group': Assign unique groups from ``group_idx`` to ``dim``
        - None: drop any coordinates.

        Note that if ``coords_policy`` is one of ``first`` or ``last``, parameter ``groups``
        will be ignored.
    by : array-like of int
        Groupby values of same length as ``data`` along sampled dimension.
        Negative values indicate no group (i.e., skip this index).
    group_dim : str, optional
        Name of the output group dimension.  Defaults to ``dim``.
    groups : sequence, optional
        Sequence of length ``by.max() + 1`` to assign as coordinates for ``group_dim``.


    min_periods : int, optional
        Minimum number of observations in window required to have a value
        (otherwise result is NA). The default, None, is equivalent to
        setting min_periods equal to the size of the window.
    center : bool, default=False
        If ``True``, set the labels at the center of the window.
    zero_missing_weights : bool, default=True
        If ``True``, set missing weights (``np.nan``) to ``0``.
    window : int
        Size of moving window.


    select_moment_name | moment : {"weight", "ave", "var", "cov", "xave", "xvar", "yave", "yvar", "xmom_0", "xmom_1", "ymom_0", "ymom_1"}
        Name of moment(s) to select.

        - ``"weight"`` : weights
        - ``"ave"`` : Averages.
        - ``"var"``: Variance.
        - ``"cov"``: Covariance if ``mom_ndim == 2``, or variace if ``mom_ndim == 1``.
        - ``"xave"``: Average of first variable.
        - ``"xvar"``: Variance of first variable.
        - ``"yave"``: Average of second variable (if ``mom_ndim == 2``).
        - ``"yvar"``: Variace of second variable (if ``mom_ndim == 2``).
        - ``"all"``: All values.

        Names ``"weight", "xave", "yave", "xvar", "yvar", "cov"`` imply shape
        ``data.shape[:-mom_ndim]``. Names ``"ave", "var"`` imply shape
        ``(*data.shape[:-mom_ndim], mom_ndim)``, unless ``mom_ndim == 1`` and
        ``squeeze = True``.
    assign_moment_mapping | moment : mapping of str to array-like
        Mapping from moment name to new value.  Allowed moment names are:

        - ``"weight"`` : weights
        - ``"ave"`` : Averages.
        - ``"var"``: Variance.
        - ``"cov"``: Covariance if ``mom_ndim == 2``, or variace if ``mom_ndim == 1``.
        - ``"xave"``: Average of first variable.
        - ``"xvar"``: Variance of first variable.
        - ``"yave"``: Average of second variable (if ``mom_ndim == 2``).
        - ``"yvar"``: Variace of second variable (if ``mom_ndim == 2``).
        - ``"xmom_n", "ymom_n"``: All values with first (second) variable moment == n.
        - ``"all"``: All values.

        Names ``"weight", "xave", "yave", "xvar", "yvar", "cov"`` imply shape
        ``data.shape[:-mom_ndim]``. Names ``"ave", "var"`` imply shape
        ``(*data.shape[:-mom_ndim], mom_ndim)``, unless ``mom_ndim == 1`` and
        ``squeeze = True``.
    select_squeeze | squeeze : bool, default=False
        If True, squeeze last dimension if ``name`` is one of ``ave`` or ``var`` and ``mom_ndim == 1``.
    select_dim_combined | dim_combined: str, optional
        Name of dimension for options that produce multiple values (e.g., ``name="ave"``).
    select_coords_combined | coords_combined: str or sequence of str, optional
        Coordates to assign to ``dim_combined``.  Defaults to names of moments dimension(s)


    mom_moments_to_comoments | mom : tuple of int
        Moments for comoments array. Pass a negative value for one of the
        moments to fill all available moments for that dimensions. For example,
        if original array has moments `m` (i.e., ``values.shape=(..., m +
        1)``), and pass in ``mom = (2, -1)``, then this will be transformed to

        ``mom = (2, m - 2)``.
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
        klass="object",
        t_array=":class:`numpy.ndarray` or :class:`xarray.DataArray` or :class:`xarray.Dataset`",
    )
    .assign_combined_key("axis_and_dim", ["axis"])
    .assign_combined_key("axis_data_and_dim", ["axis_data"])
    .update(
        vals_resample_note=dedent(
            """\
            Note that the resampled axis (``resamp_axis``) is at position
            ``-(len(mom) + 1)``, just before the moment axes. This is opposed
            to the behavior of resampling moments arrays (e.g.,
            func:`.resample_data`), where the resampled axis is the same as the
            argument ``axis``. This is because the shape of the output array
            when resampling values is dependent the result of broadcasting
            ``x`` and ``y`` and ``weight``.
            """
        )
    )
    .assign(
        DataArray=":class:`~xarray.DataArray`",
        Dataset=":class:`~xarray.Dataset`",
        ndarray=":class:`~numpy.ndarray`",
    )
)


docfiller_central = (
    docfiller.update(
        DocFiller.from_docstring(
            _dummy_docstrings_central,
            combine_keys="parameters",
        ).data
    )
    .assign(
        klass="CentralMomentsArray",
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
        klass="CentralMomentsData",
        t_array=":class:`~xarray.DataArray` or :class:`~xarray.Dataset`",
    )
    .assign_combined_key("axis_and_dim", ["axis", "dim"])
    .assign_combined_key("axis_data_and_dim", ["axis_data", "dim"])
)


docfiller_decorate = docfiller()
