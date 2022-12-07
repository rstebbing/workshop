#!/bin/bash

set -euxo pipefail


PY_ROOT=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "${PY_ROOT}"


# Ensure `pip` and `setuptools` are up-to-date.
pip install --upgrade pip setuptools


# Install exact versions from requirements.txt.
pip install --requirement requirements.txt


# Install `workshop` as an editable package.
pip install --editable '.[test]'
