##########################################
# File: logging.py                       #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import contextlib
import logging
import typing as t


_level_like = t.Union[int, str]


@contextlib.contextmanager
def logger_level(name: str, level: _level_like):
    """Temporarily set the logger identified by `name` to `level`."""
    if name not in logging.Logger.manager.loggerDict:
        raise ValueError(f"unable to find logger\n{name = }")

    logger = logging.getLogger(name)

    original_level = logger.level
    try:
        logger.setLevel(level)

        yield
    finally:
        logger.setLevel(original_level)
