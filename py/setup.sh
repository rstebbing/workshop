#!/bin/bash
##########################################
# File: setup.sh                         #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################


set -euxo pipefail


PY_ROOT=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "${PY_ROOT}"


# Ensure `pip` and `setuptools` are up-to-date.
pip install --upgrade pip setuptools


# Install exact versions from requirements.txt.
pip install --requirement requirements.txt


# Install `workshop` as an editable package.
pip install --editable '.[test]'
