# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import SOAPify


sys.path.insert(0, os.path.abspath("../../src"))
print(sys.path)

# -- Project information -----------------------------------------------------

project = "SOAPify"
copyright = "2021, Daniele Rapetti"
author = "Daniele Rapetti"

# The full version, including alpha/beta/rc tags
release = SOAPify.__version__
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    # "matplotlib.sphinxext.plot_directive",
    # "matplotlib.sphinxext.mathmpl",
    # "sphinx.ext.todo",
]

autodoc_default_options = {
    "special-members": "__init__",
    "inherited-members": True,
    "show-inheritance": True,
    "ignore-module-all": True,
}

# source_suffix = {
#    ".rst": "restructuredtext",
#    ".txt": "markdown",
#    ".md": "markdown",
# }

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_theme = "mpl_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for ToDO output -------------------------------------------------
todo_include_todos = True

# -- Options for coverage output ---------------------------------------------
coverage_show_missing_items = True
