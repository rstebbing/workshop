##########################################
# File: setup.py                         #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import os
import sys
from glob import glob
from pathlib import Path

from setuptools import find_packages, setup


if (sys.version_info.major, sys.version_info.minor) < (3, 10):
    raise RuntimeError("Python 3.10 or higher is required")


PY_ROOT = Path(__file__).parent


tests_require = [
    # These should match .pre-commit-config.yaml.
    "black == 22.10.0",
    "isort == 5.10.1",
    "pre-commit >= 2.20",
    "pytest >= 7.2",
]

setup(
    name="workshop",
    version="0.1.dev0",
    author="Richard Stebbing",
    author_email="richie.stebbing@gmail.com",
    # References:
    # https://docs.python.org/3/distutils/setupscript.html#listing-whole-packages
    # https://docs.python.org/3/distutils/setupscript.html#listing-individual-modules
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")],
    # Reference:
    # http://setuptools.readthedocs.io/en/latest/setuptools.html#including-data-files
    include_package_data=True,
    # Explicitly set `zip_safe=False`. If this is not set, this is implicitly
    # set by `setuptools` by analyzing the compiled Python modules in the
    # egg. In particular, it looks for usage of `__file__` and `__path__` in a
    # module, and if `inspect` is used, it looks for constituent package
    # functions. See `analyze_egg` in `setuptools.command.bdist_egg` for
    # details.
    zip_safe=False,
    install_requires=[
        "devtools >= 0.9",
        "matplotlib >= 3.6",
        "numpy >= 1.23",
        "pydantic >= 1.10",
        "scikit-learn >= 1.1",
        "torch >= 1.12",
        "tqdm >= 4.64",
    ],
    tests_require=tests_require,
    extras_require={
        "test": tests_require,
    },
)
