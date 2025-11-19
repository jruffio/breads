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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))



from datetime import datetime
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

try:
    from sphinx_astropy.conf.v1 import *  # noqa
except ImportError:
    print(
        "ERROR: the documentation requires the sphinx-astropy package to be installed"
    )
    sys.exit(1)

# -- Project information -----------------------------------------------------
with open(Path(__file__).parent.parent.parent / "pyproject.toml", "rb") as metadata_file:
    configuration = tomllib.load(metadata_file)
    metadata = configuration["project"]
    project = metadata["name"]
    author = metadata["authors"][0]["name"]
    copyright = f"{datetime.now().year}, {author}"

    # The short X.Y version.
    try:
        version = project.__version__.split("-", 1)[0]
        # The full version, including alpha/beta/rc tags.
        release = project.__version__
    except AttributeError:
        version = "dev"
        release = "dev"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              "sphinx.ext.inheritance_diagram",
                "sphinx.ext.viewcode",
                "sphinx.ext.autosummary",

              "sphinx_automodapi.automodapi",
                  "nbsphinx",
              #"nbsphinx",
              "numpydoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
