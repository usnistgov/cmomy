<!-- markdownlint-disable MD024 -->
<!-- markdownlint-disable MD013 -->
<!-- prettier-ignore-start -->

# Changelog

Changelog for `cmomy`

## Unreleased

[changelog.d]: https://github.com/usnistgov/cmomy/tree/main/changelog.d

See the fragment files in [changelog.d]

<!-- prettier-ignore-end -->

<!-- markdownlint-enable MD013 -->

<!-- scriv-insert-here -->

## v0.24.0 — 2025-01-23

### Added

- Python version 3.13 now supported.

### Changed

- Moved grouped/indexed routines (e.g., `reduce_data_indexed`, `factor_by`, etc)
  to `cmomy.grouped` submodule. Main functions are still available at top level
  (`cmomy.reduce_data_grouped` is `cmomy.grouped.reduce_data_grouped`).

## v0.23.0 — 2024-10-30

### Changed

- Routines now accept `mom_axes` parameter. This allows for moment axes to be in
  arbitrary location
- Routines now accept `mom_params` parameter. This object contains all the logic
  for working with moment arrays. This Also simplifies calling routines from
  other classes/routines.
- Update requirements, typing, and linting
- Routines now correctly respect the `order` parameter. Passing in `None` will
  lead to arrays that are `c` ordered when "normalized" (i.e., which have `axis`
  and `mom_axes` at the end), which is the default behavior of
  `numb.guvectorize`. Passing order `c` will lead to outputs that are `c`
  ordered regardless of `mom_axes` location.

- Negative axis/axes are now treated relative to the end of the array (just like
  python lists and numpy ndarray objects). To get the old behavior of counting
  relative to the `mom_ndim`, pass in imaginary axis. For example, passing
  `axis=-1j` with `mom_ndim=1` is equivalent to passing `axis=-2`.

- renamed parameter `move_axes_to_end` to `axes_to_end`. This will lead to axes
  being in the form `(..., *mom_axes)` if `axes` are reduced or
  `(..., *axes, *mom_axes)` if `axes` are kept.

- Applying routines to `dataset` objects now keeps correct order if
  `axes_to_end` is `False`.

## v0.22.0 — 2024-09-28

### Changed

- For xarray (DataArray and Dataset) central moments data, routines will infer
  `mom_ndim` from `mom_dims`. For example, can calling
  `cmomy.reduce_data(data, mom_dims=("mom_0", "mom_1"), dim="a")` will infer
  `mom_ndim=1`.

## v0.21.0 — 2024-09-25

### Removed

- `cmomy.randsamp_freq` has been replaced with `cmomy.factory_sampler`

### Added

- `cmomy.factory_sampler` to create a sampler for resample routines.
- `cmomy.IndexSampler` to wrap resample "indices" and "freq" table.

### Changed

- Added `IndexSampler` class to handle resampling. This wraps either resampling
  `indices` or a "frequency" table `freq`, and can produce whichever is not
  provided.
- replace `randsamp_freq` with `factory_sampler` which creates `IndexSampler`
  from parameters or a mapping.
- Removed arguments `freq`, `nrep`, `rng`, `paired` from resampling routines
  (`resample_data`, etc). This was replaced by an arguments `sampler` which can
  be either an `IndexSampler` or a mapping which is passed to `factory_sampler`.
  This means that to call the resampling routines, you'll have to pass a sampler
  or a dict of parameters. While this is a little annoying, it greatly cleans up
  a bunch of parameters that may not be used. For example, the call
  `cmomy.resample_data(data, axis=0, nrep=10, rng=0)` is now
  `cmomy.resampler_data(data, axis=0, sampler={"nrep": 10, "rng": 0})`. Plus,
  you can now pass `indices` with
  `cmomy.resample_data(data, axis=0, sampler={"indices": indices})`
- Removed parameter `on_missing_core_dim` from routines that call
  `xarray.apply_ufunc` behind the scenes. You can still pass this parameter
  using `apply_ufunc_kwargs`. For example,
  `cmomy.reduce_data(data, axis=0, apply_ufunc_kwargs={"on_missing_core_dim": "drop"})`.

## v0.20.0 — 2024-09-21

### Fixed

- Made `cmomy.resample.freq_to_indices` and `cmomy.resample.indices_to_freq`
  gufunc's. Much faster than old code (`freq_to_indices` was a bottleneck).

## v0.19.0 — 2024-09-20

### Added

- Added `cmomy.convert.comoments_to_moments` routine, which is the inverse of
  `cmomy.convert.moments_to_comoments`.

## v0.18.0 — 2024-09-18

### Added

- `cmomy.assign_moment` and `cmomy.select_moment` now accept options `xmom_0`,
  `xmom_1`, `ymom_0`, and `ymom_1`. These allow selecting/assigning to slices.
  Useful when converting values to central moments array.

## v0.17.0 — 2024-09-16

### Added

- Now fully support wrapping `xarray.Dataset` objects
- Complete rewrite of wrapper classes
- Added `wrapper` factory methods `cmomy.wrap`, `cmomy.wrap_reduce_vals`, etc.
  These automatically select from `CentralMomentsArray` and
  `CentralMomentsData`.
- Renamed `CentralMoments` to `CentralMomentsArray` and `xCentralMoments` to
  `CentralMomentsData`. The former is parameterized across numpy arrays. The
  later is parameterized to work with either `xarray.DataArray` or
  `xarray.Dataset` objects
- Full support `dask` backed `xarray` objects. This will be lazily evaluated.
- Removed `CentralMomentsArray/Data.block` method. Instead, there is a `block`
  keyword in `reduce` method.
- `to_dataarray`, `to_dataset`, etc, method of `CentralMomentsArray/Data` now
  return a wrapped object instead of an `xarray` object. To access the
  underlying `xarray` object, use the `obj` attribute.
- Removed `values` and `data` attributes from `CentralMomentsArray/Data` wrapper
  classes. Now wrap either an array or `xarray` object directly (instead of only
  really wrapping `numpy` arrays in the old code). To access the wrapped object,
  now use `obj` attribute.
- Now include testing of typing (`mypy` and `pyright`) support.

## v0.16.0 — 2024-08-06

### Added

- Added `rolling` module for rolling mean and exponential rolling mean.
- Added `bootstrap_confidence_interval` method to calculate confidence intervals
- Added `moveaxis` function to cleanly handle axes movement of central moments
  array
- Added `select_moment` method to select specific moments (weight, average, etc)
  from a central moments array
- Added `assign_moment` method to assign values to specific moments.
- Added `vals_to_data` to simplify using `_data` methods for raw values.
- Added `jackknife_` an routines to perform jackknife analysis.
- Update `_vals` routines to properly place the `axis` parameter in result.
- Added `axes_to_end` option to most routines.
- Added `keepdims` option from `reduce_data` and `reduce_vals`

## v0.15.0 — 2024-06-21

### Added

- Added `cmomy.concat` method to concatenate moments objects.

- Added `__getitem__` to `(x)CentralMoments` objects. This method **does not\***
  allow changing the moments shape. If you want to do that, you'll need to work
  directly with `(x)CentralMoments.to_values()`

## v0.14.0 — 2024-06-20

### Added

- added `cmomy.resample.select_ndat` to select data size along reduction
  dimension
- Added `cmomy.randsamp_freq` to top level api

### Changed

- Updated `cmomy.resample.randsamp_freq` to select ndat from array

## v0.13.0 — 2024-06-18

### Added

- Added `cmomy.convert.moments_to_comoments` (and
  `(x)CentralMoments.moments_to_comoments`)to convert from single variable
  moments to comoments. This is useful in `thermoextrap`.
- Added `cmomy.convert.assign_weight` (and `(x)CentralMoments.assign_weights`)
  to update weights (useful in `thermoextrap`).

- Added support for `numpy>=2.0.0`. Because we still support older versions, we
  still use the old convention for the `copy` parameter to `numpy.array`. Will
  change this when minimum numpy is 2.0.

### Changed

- Renamed `cmomy.convert` function to `cmomy.convert.moments_type`A bullet item
  for the Changed category.

## v0.12.0 — 2024-06-13

### Added

- Now supports python3.12

## v0.11.0 — 2024-06-12

### Changed

- Switch to underlying numba functions using `guvectorize`. This significantly
  simplifies the code. Previously, we had separate functions for "vector" vs
  "scalar" moments. To handle arbitrary vector dimensions, the arrays were
  reshaped behind the scenes (to a single "meta" dimension). Now, this is all
  handled by the `gufunc` based library code.
- Typing support improved.
- Added `(x)CentralMoments.astype`
- Added `(x)CentralMoments.`
- Added alias `CentralMoments.to_x` which is the same as
  `CentralMoments.to_xcentralmoments`.
- Added alias `xCentralMoments.to_c` which is the same as
  `xCentralMoments.to_centralmoments`.
- Most constructors now accept `order` and `dtype` arguments.
- Most routines that process central moments accept a `parallel` parameter.
- Instead of complicated internal validation routines in `(x)CentralMoments`,
  most of this is now handled by `cmomy.reduction` or similar routines.
- Now using `xr.apply_ufunc` for most of the `xarray.DataArray` based
  calculations.

### Deprecated

- Removed classmethod `(x)CentralMoments.from_raws`. Instead, use
  `(x)CentralMoments.from_raw(...).reduce(...)`.
- Removed classmethod `(x)CentralMoments.from_datas`. Instead, use
  `(x)CentralMoments.from_data(...).reduce(...)`.
- Removed classmethod `(x)CentralMoments.from_data`. Instead, use
  `(x)CentralMoments(....)`.
- Removed ability to create `xCentralMoments` objects directly from
  `numpy.ndarray` objects. (e.g., passing in array-like to
  `xCentralmoments.from_vals` does not work anymore). Instead use
  `CentralMoments.from_vals(....).to_xcentralmoments(...)`, etc.
- Removed methods `push_stat`, `push_stats`, `from_stat`, `from_stats`. Instead
  use, for example, `numpy.concatenate`, to combine weights, average, and
  variance into a `data` array. A helper function may be added if called for.
- `(x)CentralMoments.resample_and_reduce` and
  `(x)CentralMoments.from_resample_vals` no longer accept `nrep=...` or
  `indices=...`. They only accept `freq=...`.

## v0.9.0 — 2024-04-10

### Changed

- Can now resample with an arbitrary number of samples. Previously, it was
  assumed that resampling should be done with a shape `(nrep, ndat)`, where
  `nrep` is the number of replicates and `ndat` is the shape of the data along
  the resampled axis. Now you can pass sample with shape `(nrep, nsamp)` where
  `nsamp` is the specified number of samples in a replicate (defaulting to
  `ndat`). This allows users to do things like jackknife resampling, etc, with
  `resample_and_reduce` methods.
- Preliminary support for using type hints in generated documentation. The
  standard sphinx autodoc support does not quite work for `cmomy`, as it
  requires type hints to be accessible at run time, and not in `TYPE_CHECKING`
  blocks. Instead, we use
  [`sphinx_autodoc_type`](https://github.com/tox-dev/sphinx-autodoc-typehints).
  This has the downside of expanding type aliases, but handles (most things)
  being in `TYPE_CHECKING` blocks. Over time, we'll replace some of the explicit
  parameter type documentation with those from type hints.
- Fixed creation of templates in reduction routines of `xCentralMoments`.
  Previously, we build the template for the result using something like
  `da.isel(dim=0)`. This kept scalar coordinates of `da` with `dim`. Now we use
  `da.isel(dim=0, drop=True)` to drop these.
- Updated dependencies.

## v0.8.0 — 2024-02-20

### Added

- Added `to_values` method to access underlying array data. This should be
  preferred to `.values` attribute.
- Added `to_numpy` method to access underlying `numpy.ndarray`.
- Added `to_dataarray` method to access underlying `xarray.DataArray` in
  `xCentralMoment s`

- Added submodule `cmomy.random` to handle random numbers generation. This uses
  `numpy.random.Generator` behind the scenes.
- Updated `ruff` linting rules
- Now using `hatchling` for package building
- Update repo template

### Changed

- Now CentralMoments and xCentralMoments ensure that data/data_flat share
  memory. This may result in passed data not being the same as the internal
  data, if reshaping data creates a copy.
- Made little used arguments keyword only

## v0.7.0 — 2023-08-11

### Added

- Now use [lazy_loader](https://github.com/scientific-python/lazy_loader) to
  speed up initial load time.

- Now using `module_utilities >=0.6`.
- Changed from `custom-inherit` to `docstring-inheritance`
- Now fully supports typing (passing mypy --strict and pyright)
- Relocated numba functions to submodule `cmomy._lib`.

### Changed

- Moved tests to top level of repo (`src/cmomy/tests` to `tests`)

## v0.5.0 — 2023-06-14

### Added

- Package now available on conda-forge

- Bumped maximum python version to 3.11

[`v0.4.1...v0.5.0`](https://github.com/usnistgov/cmomy/compare/v0.4.1...v0.5.0)

### Changed

- Testing now handled with nox.

## v0.4.0 — 2023-05-02

### Added

- Moved module `_docstrings_` to `docstrings`. This can be used by other
  modules.

### Changed

- Update package layout
- New linters via pre-commit
- Development env now handled by tox

- Now use `module-utilities` to handle caching and docfiller.

[`v0.3.0...v0.4.0`](https://github.com/usnistgov/cmomy/compare/v0.3.0...v0.4.0)

## v0.3.0 - 2023-04-24

Full set of changes:
[`v0.2.2...v0.3.0`](https://github.com/usnistgov/cmomy/compare/v0.2.2...v0.3.0)

## v0.2.2 - 2023-04-05

Full set of changes:
[`v0.2.1...v0.2.2`](https://github.com/usnistgov/cmomy/compare/v0.2.1...v0.2.2)

## v0.2.1 - 2023-04-05

Full set of changes:
[`v0.2.0...v0.2.1`](https://github.com/usnistgov/cmomy/compare/v0.2.0...v0.2.1)

## v0.2.0 - 2023-03-22

Full set of changes:
[`v0.1.9...v0.2.0`](https://github.com/usnistgov/cmomy/compare/v0.1.9...v0.2.0)

## v0.1.9 - 2023-02-15

Full set of changes:
[`v0.1.8...v0.1.9`](https://github.com/usnistgov/cmomy/compare/v0.1.8...v0.1.9)

## v0.1.8 - 2022-12-02

Full set of changes:
[`v0.1.7...v0.1.8`](https://github.com/usnistgov/cmomy/compare/v0.1.7...v0.1.8)

## v0.1.7 - 2022-09-28

Full set of changes:
[`v0.1.6...v0.1.7`](https://github.com/usnistgov/cmomy/compare/v0.1.6...v0.1.7)

## v0.1.6 - 2022-09-27

Full set of changes:
[`v0.1.5...v0.1.6`](https://github.com/usnistgov/cmomy/compare/v0.1.5...v0.1.6)

## v0.1.5 - 2022-09-26

Full set of changes:
[`v0.1.4...v0.1.5`](https://github.com/usnistgov/cmomy/compare/v0.1.4...v0.1.5)

## v0.1.4 - 2022-09-15

Full set of changes:
[`v0.1.3...v0.1.4`](https://github.com/usnistgov/cmomy/compare/v0.1.3...v0.1.4)

## v0.1.3 - 2022-09-15

Full set of changes:
[`v0.1.2...v0.1.3`](https://github.com/usnistgov/cmomy/compare/v0.1.2...v0.1.3)

## v0.1.2 - 2022-09-13

Full set of changes:
[`v0.1.1...v0.1.2`](https://github.com/usnistgov/cmomy/compare/v0.1.1...v0.1.2)

## v0.1.1 - 2022-09-13

Full set of changes:
[`v0.1.0...v0.1.1`](https://github.com/usnistgov/cmomy/compare/v0.1.0...v0.1.1)

## v0.1.0 - 2022-09-13

Full set of changes:
[`v0.0.7...v0.1.0`](https://github.com/usnistgov/cmomy/compare/v0.0.7...v0.1.0)

## v0.0.7 - 2021-05-18

Full set of changes:
[`v0.0.6...v0.0.7`](https://github.com/usnistgov/cmomy/compare/v0.0.6...v0.0.7)

## v0.0.6 - 2021-02-03

Full set of changes:
[`v0.0.4...v0.0.6`](https://github.com/usnistgov/cmomy/compare/v0.0.4...v0.0.6)

## v0.0.4 - 2020-12-21

Full set of changes:
[`v0.0.3...v0.0.4`](https://github.com/usnistgov/cmomy/compare/v0.0.3...v0.0.4)
