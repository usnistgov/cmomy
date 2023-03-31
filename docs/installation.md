```{highlight} shell
```

# Installation

## Stable release

To install cmomy, run this command in your terminal:

```console
$ pip install cmomy
```

or

```console
$ conda install -c wpk-nist cmomy
```

This is the preferred method to install cmomy, as it will always install the most recent stable release.

If you don't have [pip] installed, this [Python installation guide] can guide
you through the process.

## From sources

The sources for cmomy can be downloaded from the [Github repo].

You can either clone the public repository:

```console
$ git clone git://github.com/usnistgov/cmomy.git
```

Once you have a copy of the source, you can install it with:

```console
$ pip install .
```

To install dependencies with conda/mamba, use:

```
.. code-block:: console
```

> \$ conda/mamba env create -n \{name} -f environment.yaml
> \$ pip install . --no-deps

To install an editable version, add the `-e` option to pip.

[github repo]: https://github.com/usnistgov/cmomy
[pip]: https://pip.pypa.io
[python installation guide]: http://docs.python-guide.org/en/latest/starting/installation/
