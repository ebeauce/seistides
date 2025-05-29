"""
Minimal setup file for the seistides library for Python packaging.
:copyright:
    Eric Beauce
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

from __future__ import print_function
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as build_ext_original
from subprocess import call


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="seistides",
    version="0.0.1",
    author="Eric Beauce",
    author_email="ebeauce@ldeo.columbia.edu",
    description="Package for analysis of tidal modulation of seismicity.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ebeauce/seistides",
    project_urls={
        "Bug Tracker": "https://github.com/ebeauce/seistides/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    license="GPL",
    packages=['seistides'],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "numpy", "pandas", "matplotlib", "scipy", "h5py"
        ],
    python_requires=">=3.10",
)
