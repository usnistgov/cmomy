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

## v0.8.0 — 2024-02-20

### Added

- Added `to_values` method to access underlying array data. This should be
  preferred to `.values` attribute.
- Added `to_numpy` method to access underlying `numpy.ndarray`.
- Added `to_dataarray` method to access underlying `xarray.DataArray` in
  `xCentralMoment s`

- Added submodule `cmomy.random` to handle random numbers generation. This uses
  `numpy.random.Generator` behind the scenes.
- Updated `ruff` lintering rules
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
- Now fully supports typing (passing mypy --stict and pyright)
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
