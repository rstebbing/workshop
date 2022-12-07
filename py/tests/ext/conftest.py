##########################################
# File: conftest.py                      #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    d = None
    try:
        d = Path(tempfile.mkdtemp())
        yield d

    finally:
        if d is not None and d.exists():
            shutil.rmtree(d)
