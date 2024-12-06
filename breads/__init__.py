
from .fit import *


# Infer package version number from metadata, via setuptools scm
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("breads")
except PackageNotFoundError:
    __version__ = "unknown version"
