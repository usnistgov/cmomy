<!-- markdownlint-disable MD041 -->
<!--
A new scriv changelog fragment.

Uncomment the section that is right (remove the HTML comment wrapper).
-->

<!--
### Removed

- A bullet item for the Removed category.

-->

### Added

- Added `reduce_vals_grouped` and `reduce_vals_indexed` to create grouped
  central moments from values.
- Added better typing fallbacks for unions of array-like or xarray objects.
- Value reduction (`reduce_vals` `resample_vals`, etc) now support `mom_axes`
  parameter. This allows `mom_axes` to be placed where you like, with output in
  proper order
- Fully support output order (e.g., `order="C"`) for xarray objects, including
  reordering `mom_axes` of value reduction (`reduce_vals`, etc) for `DataArray`
  objects.
- Added `coords_policy` to `reduce_data/vals_grouped`.
- Added `coords_policy` support to `Dataset` objects.
- Added testing coverage to default python version.

### Changed

- Bug fix to numba decorator
- Update `cmomy.core.prepare`
- Rework `cmomy.core.moment_params` to better handle subclassing
- Programmatic import in `cmomy.factory` module
- Use `__init__.pyi` with `lazy_loader`.
- Cleanup core calculation in `_reduce_data`

<!--
### Deprecated

- A bullet item for the Deprecated category.

-->
<!--
### Fixed

- A bullet item for the Fixed category.

-->
<!--
### Security

- A bullet item for the Security category.

-->
