[![PyPI version](https://badge.fury.io/py/cmomy.svg)](https://badge.fury.io/py/cmomy)
[![Code style: black][black-badge]][black-link]
<!-- [![PyPi][pypi-badge]][pypi-link] -->
[![Conda (channel only)](https://img.shields.io/conda/vn/wpk-nist/cmomy)](https://anaconda.org/wpk-nist/cmomy)


<!-- [github-ci]: https://github.com/executablebooks/MyST-Parser/workflows/continuous-integration/badge.svg?branch=master -->
<!-- [github-link]: https://github.com/executablebooks/MyST-Parser -->
<!-- [codecov-badge]: https://codecov.io/gh/executablebooks/MyST-Parser/branch/master/graph/badge.svg -->
<!-- [codecov-link]: https://codecov.io/gh/executablebooks/MyST-Parser -->
<!-- [rtd-badge]: https://readthedocs.org/projects/myst-parser/badge/?version=latest -->
<!-- [rtd-link]: https://myst-parser.readthedocs.io/en/latest/?badge=latest -->
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]: https://github.com/ambv/black
[pypi-badge]: https://img.shields.io/pypi/v/cmomy
[pypi-link]: https://pypi.org/project/cmomy
<!-- [conda-badge]: https://anaconda.org/conda-forge/myst-parser/badges/version.svg -->
<!-- [conda-link]: https://anaconda.org/conda-forge/myst-parser -->
<!-- [install-badge]: https://img.shields.io/pypi/dw/myst-parser?label=pypi%20installs -->
<!-- [install-link]: https://pypistats.org/packages/myst-parser -->
# cmomy

A Python package to calculate and manipulate Central (co)moments. The main features of ``cmomy`` are as follows:

* [Numba](https://numba.pydata.org/) accelerated computation of central moments and co-moments
* Routines to combine, and resample central moments.
* Both [numpy](https://numpy.org/) array-like and [xarray](https://docs.xarray.dev/en/stable/) DataArray interfaces to
  Data.
* Routines to convert between central and raw moments.



## Overview

`cmomy` is an open source package to calculate central moments and
co-moments in a numerical stable and direct way. Behind the scenes,
`cmomy` makes use of [Numba](https://numba.pydata.org/) to rapidly
calculate moments. A good introduction to the type of formulas used can
be found
[here](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance).


## Features
* Fast calculation of central moments and central co-moments with weights
* Support for scalar or vector inputs
* numpy and xarray api's
* bootstrap resampling


## Status

This package is actively used by the author.  Please feel free to create a pull request for wanted features and suggestions!


## Quick start

Use one of the following

``` bash
pip install cmomy
```

or

``` bash
conda install -c wpk-nist cmomy
```


<!-- end-docs -->

## Note on caching

This code makes extensive use of the numba python package. This uses a
jit compiler to speed up vital code sections. This means that the first
time a function called, it has to compile the underlying code. However,
caching has been implemented. Therefore, the very first time you run a
function, it may be slow. But all subsequent uses (including other
sessions) will be already compiled.

## Testing

Tests are packaged with the distribution intentionally. To test code
run:

``` bash
pytest --pyargs cmomy
```

By running the tests once, you create a cache of the numba code for most
cases. The first time you run the tests, it will take a while (about 1.5
min on my machine). However, subsequent runs will be much faster (about
3 seconds on my machine).

There are a variety of tests.  More testing is always needed!


## Examples

See the [motivation](docs/notebooks/motivation.ipynb) and [usage](docs/notebook/usage_notebook.ipynb) for examples of ``cmomy`` in action.

For a deeper dive, look at the [documentation](https://pages.nist.gov/cmomy/)

## License

This is free software.  See [LICENSE](LICENSE).

## Related work

This package is used extensively in the newest version of ``thermoextrap``.  See [here](https://github.com/usnistgov/thermo-extrap).


## Contact

The author can be reached at wpk@nist.gov.

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[wpk-nist-gov/cookiecutter-pypackage](https://github.com/wpk-nist-gov/cookiecutter-pypackage)
Project template forked from
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
