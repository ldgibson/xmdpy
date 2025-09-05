from __future__ import absolute_import, division, print_function

from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ""  # use '' for first of series, number for 1 and above
_version_extra = "dev0"
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = ".".join(map(str, _ver))

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
]

# Description should be a one-liner:
description = "xmdpy: an Xarray interface for molecular dynamics trajectories"
# Long description will go up on the pypi page
long_description = """

xmdpy
=====
xmdpy is a package that provides Xarray interfaces for molecular dynamics 
trajectory files to facilitate fast and easy analysis in large trajectories. 
It can be used to lazily load trajectory data into Xarray Datasets and can 
leverage Dask to perform custom analyses in a memory-sensitive and efficient 
manner. By using Xarray as the container for trajectories, The powerful and 
flexible indexing provided by Xarray also allows for complex operations on 
large trajectories.

This package is in early active development and will frequently have 
significant changes.

License
=======
``xmdpy`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2025--, Luke Gibson.
"""

NAME = "xmdpy"
MAINTAINER = "Luke Gibson"
MAINTAINER_EMAIL = "ldgibson819@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/ldgibson/xmdpy"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Luke Gibson"
AUTHOR_EMAIL = "ldgibson819@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {"xmdpy": [pjoin("data", "*")]}
REQUIRES = ["dask", "numpy", "xarray", "pandas"]
PYTHON_REQUIRES = ">= 3.11"
ENTRY_POINTS = {"xarray.backends": ["xmdpy=xmdpy.backend:XMDPYBackendEntrypoint"]}
