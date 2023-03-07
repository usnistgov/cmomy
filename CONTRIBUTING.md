```{highlight} shell
```

# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/wpk-nist-gov/cmomy/issues>.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

cmomy could always use more documentation, whether as part of the
official cmomy docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at <https://github.com/wpk-nist-gov/cmomy/issues>.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `cmomy` for local development.

01. Fork the `cmomy` repo on GitHub.

02. Clone your fork locally:

    ```
    $ git clone git@github.com:your_name_here/cmomy.git
    ```

03. Install dependecies.  There are useful commands in the makefile, that depend on
    `pre-commit` and `conda-merge`.  These can be installed with `pip`, `pipx`, or `conda/mamba`.

04. Initiate pre-commit with:

    ```
    $ pre-commit install
    ```

    To update the recipe, use:

    ```
    $ pre-commit autoupdate
    ```

05. Create virtual env:

    ```
    $ make mamba-dev
    $ conda activate cmomy-env
    ```

    Alternatively, to create a different named env, use:

    ```
    $ make environment-dev.yml
    $ conda/mamba env create -n {env-name} -f environment-dev.yml
    $ conda activate {env-name}
    ```

06. Install editable package:

    ```
    $ pip install -e . --no-deps
    ```

07. Create a branch for local development:

    ```
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.  Alternatively, we recommend using git flow.

08. When you're done making changes, check that your changes pass flake8 and the
    tests, including testing other Python versions with tox:

    ```
    $ pre-commit run [--all-files]
    $ pytest
    ```

    To get flake8 and tox, just pip install them into your virtualenv.

09. Commit your changes and push your branch to GitHub:

    ```
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

10. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.8, 3.9, 3.10.

## Using tox

The package is setup to use tox to test, build and release pip and conda distributions, and release the docs.  Most of these tasks have a command in the makefie.  To test against multiple versions, use:

```
$ make test-all
```

To build the documentation in an isolated environment, use:

```
$ make docs-build
```

To release the documentation use:

```
$ make docs-release posargs='-m "commit message" -r origin -p'
```

Where posargs is are passed to ghp-import.  Note that the branch created is called `nist-pages`.  This can be changed in `tox.ini`.

To build the distribution, use:

```
$ make dist-pypi-[build-testrelease-release]
```

where `build` build to distro, `testrelease` tests putting on `testpypi` and release puts the distro on pypi.

To build the conda distribution, use:

```
$ make dist-conda-[recipe, build]
```

where `recipe` makes the conda recipy (using grayskull), and `build` makes the distro.  This can be manually added to a channel.

To test the created distrobutions, you can use one of:

```
$ make test-dist-[pypi, conda]-[local,remote] py=[38, 39, 310]
```