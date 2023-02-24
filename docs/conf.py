#!/usr/bin/env python
#
# python_boilerplate documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import cmomy

# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "sphinx_automodapi.automodapi",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    # "sphinx_autosummary_accessors",
    # "scanpydoc.rtd_github_links",
    # view source code on created page
    # "sphinx.ext.viewcode",
    # view source code on github
    "sphinx.ext.linkcode",
    # RTD dark
    # "sphinx_rtd_dark_mode"
]

# defined stuff, from xarray
nbsphinx_prolog = """

{% set docname = env.doc2path(env.docname, base=None) %}


You can view this notebook `on Github <https://github.com/usnistgov/cmomy/blob/master/docs/{{ docname }}>`_.
"""


# The kernelname to use.
nbsphinx_kernel_name = "python3"

# autosummary_generate = True
autosummary_generate = True

autoclass_content = "both"  # include both class docstring and __init__
autodoc_default_flags = [
    # Make sure that any autodoc declarations show the right members
    "members",
    "inherited-members",
    "private-members",
    "show-inheritance",
]
# # for scanpydoc's jinja filter
# project_dir = pathlib.Path(__file__).parent.parent
html_context = {
    "github_user": "usnistgov",
    "github_repo": "cmomy",
    "github_version": "master",
    "doc_path": "doc",
}

autodoc_typehints = "none"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # general terms
    "sequence": ":term:`sequence`",
    "iterable": ":term:`iterable`",
    "callable": ":py:func:`callable`",
    "dict_like": ":term:`dict-like <mapping>`",
    "dict-like": ":term:`dict-like <mapping>`",
    "path-like": ":term:`path-like <path-like object>`",
    "mapping": ":term:`mapping`",
    "hashable": ":term:`hashable`",
    # special terms
    # "same type as caller": "*same type as caller*",  # does not work, yet
    # "same type as values": "*same type as values*",  # does not work, yet
    # stdlib type aliases
    "MutableMapping": "~collections.abc.MutableMapping",
    "sys.stdout": ":obj:`sys.stdout`",
    "timedelta": "~datetime.timedelta",
    "string": ":class:`string <str>`",
    # numpy terms
    "array_like": ":term:`array_like`",
    "array-like": ":term:`array-like <array_like>`",
    "scalar": ":term:`scalar`",
    "array": ":term:`array`",
    # matplotlib terms
    "color-like": ":py:func:`color-like <matplotlib.colors.is_color_like>`",
    "matplotlib colormap name": ":doc:`matplotlib colormap name <matplotlib:gallery/color/colormap_reference>`",
    "matplotlib axes object": ":py:class:`matplotlib axes object <matplotlib.axes.Axes>`",
    "colormap": ":py:class:`colormap <matplotlib.colors.Colormap>`",
    # objects without namespace: xarray
    "DataArray": "~xarray.DataArray",
    "Dataset": "~xarray.Dataset",
    "Variable": "~xarray.Variable",
    "DatasetGroupBy": "~xarray.core.groupby.DatasetGroupBy",
    "DataArrayGroupBy": "~xarray.core.groupby.DataArrayGroupBy",
    "CentralMoments": "~cmomy.CentralMoments",
    "xCentralMoments": "~cmomy.xCentralMoments",
    # objects without namespace: numpy
    "ndarray": "~numpy.ndarray",
    "MaskedArray": "~numpy.ma.MaskedArray",
    "dtype": "~numpy.dtype",
    "ComplexWarning": "~numpy.ComplexWarning",
    # objects without namespace: pandas
    "Index": "~pandas.Index",
    "MultiIndex": "~pandas.MultiIndex",
    "CategoricalIndex": "~pandas.CategoricalIndex",
    "TimedeltaIndex": "~pandas.TimedeltaIndex",
    "DatetimeIndex": "~pandas.DatetimeIndex",
    "Series": "~pandas.Series",
    "DataFrame": "~pandas.DataFrame",
    "Categorical": "~pandas.Categorical",
    "Path": "~~pathlib.Path",
    # objects with abbreviated namespace (from pandas)
    "pd.Index": "~pandas.Index",
    "pd.NaT": "~pandas.NaT",
}

numpydoc_class_members_toctree = True
numpydoc_show_class_members = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "cmomy"
author = "William P. Krekelberg"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
# versioning with scm with editable install has issues.
# instead, try to use scm if available.
try:
    from setuptools_scm import get_version

    version = get_version(root="..", relative_to=__file__)
    release = version
except ImportError:
    version = cmomy.__version__
    # The full version, including alpha/beta/rc tags.
    release = cmomy.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = "sphinx_rtd_theme"
html_theme = "nature"

html_css_files = [
    "https://pages.nist.gov/nist-header-footer/css/nist-combined.css",
    "https://pages.nist.gov/leaveNotice/css/jquery.leaveNotice.css",
]

html_js_files = [
    "https://code.jquery.com/jquery-3.6.2.min.js",
    "https://pages.nist.gov/nist-header-footer/js/nist-header-footer.js",
    "https://pages.nist.gov/leaveNotice/js/jquery.leaveNotice-nist.min.js",
    "leave_notice.js",
]


# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {"logo_only": True}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# Sometimes the savefig directory doesn't exist and needs to be created
# https://github.com/ipython/ipython/issues/8733
# becomes obsolete when we can pin ipython>=5.2; see ci/requirements/doc.yml
ipython_savefig_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_build", "html", "_static"
)
if not os.path.exists(ipython_savefig_dir):
    os.makedirs(ipython_savefig_dir)


# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
today_fmt = "%Y-%m-%d"
html_last_updated_fmt = today_fmt


# code_url = f"https://github.com/usnistgov/cmomy/blob/develop/cmomy/"
# def linkcode_resolve(domain, info):
#     # Non-linkable objects from the starter kit in the tutorial.
#     assert domain == "py", "expected only Python objects"

#     import importlib, inspect

#     mod = importlib.import_module(info["module"])
#     if "." in info["fullname"]:
#         objname, attrname = info["fullname"].split(".")
#         obj = getattr(mod, objname)
#         try:
#             # object is a method of a class
#             obj = getattr(obj, attrname)
#         except AttributeError:
#             # object is an attribute of a class
#             return None
#     else:
#         obj = getattr(mod, info["fullname"])

#     try:
#         file = inspect.getsourcefile(obj)
#         lines = inspect.getsourcelines(obj)
#     except TypeError:
#         # e.g. object is a typing.Union
#         return None
#     file = os.path.relpath(file, os.path.abspath(".."))
#     # if not file.startswith("src/websockets"):
#     #     # e.g. object is a typing.NewType
#     #     return None
#     start, end = lines[1], lines[1] + len(lines[0]) - 1

#     return f"{code_url}/{file}#L{start}-L{end}"


# based on numpy doc/source/conf.py
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    import inspect

    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(cmomy.__file__))

    return f"https://github.com/usnistgov/cmomy/blob/master/src/cmomy/{fn}{linespec}"


# only set spelling stuff if installed:
try:
    import sphinxcontrib.spelling  # noqa: F401

    extensions += ["sphinxcontrib.spelling"]
    spelling_word_list_filename = "spelling_wordlist.txt"

except ImportError:
    pass


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "cmomydoc"


# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, "cmomy.tex", "cmomy Documentation", "William P. Krekelberg", "manual"),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "cmomy", "cmomy Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "cmomy",
        "cmomy Documentation",
        author,
        "cmomy",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "numba": ("https://numba.pydata.org/numba-doc/latest", None),
    # "matplotlib": ("https://matplotlib.org", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "cftime": ("https://unidata.github.io/cftime", None),
    "sparse": ("https://sparse.pydata.org/en/latest/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# think jinja stuff
# def escape_underscores(string):
#     return string.replace("_", r"\_")


# def setup(app):
#     DEFAULT_FILTERS["escape_underscores"] = escape_underscores


# based on numpy doc/source/conf.py
# def linkcode_resolve(domain, info):
#     """
#     Determine the URL corresponding to Python object
#     """
#     import inspect


#     if domain != "py":
#         return None

#     modname = info["module"]
#     fullname = info["fullname"]

#     submod = sys.modules.get(modname)
#     if submod is None:
#         return None

#     obj = submod
#     for part in fullname.split("."):
#         try:
#             obj = getattr(obj, part)
#         except AttributeError:
#             return None

#     try:
#         fn = inspect.getsourcefile(inspect.unwrap(obj))
#     except TypeError:
#         fn = None
#     if not fn:
#         return None

#     try:
#         source, lineno = inspect.getsourcelines(obj)
#     except OSError:
#         lineno = None

#     if lineno:
#         linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
#     else:
#         linespec = ""

#     fn = os.path.relpath(fn, start=os.path.dirname(cmomy.__file__))

#     return f"https://github.com/usnistgov/cmomy/blob/develop/cmomy/{fn}{linespec}"
