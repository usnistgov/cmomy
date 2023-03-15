# Installation



## Stable release

To install cmomy, run one of the following command.  From pip:
```console
pip install cmomy
```

from conda/mamba:
```console
conda install -c wpk-nist cmomy
```

## From sources

The sources for cmomy can be downloaded from the [Github repo].

You can either clone the public repository:

```console
$ git clone {repo}
```

Once you have a copy of the source, you can install it with:

```console
# You may want a separate virtual environment.  You can create a conda env with
$ conda env create -n {env-name} -f environment-dev.yml
$ conda activate {env-name}

# install editable package
$ pip install -e . --no-deps
```

[github repo]: https://github.com/usnistgov/cmomy
