##########################################
# File: env.mk                           #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

export SRC_ROOT := $(shell dirname $(realpath $(lastword ${MAKEFILE_LIST})))

export OPT_ROOT := ${SRC_ROOT}/opt

OPT_PY := ${OPT_ROOT}/py
export VIRTUAL_ENV := ${OPT_PY}
export PATH := ${VIRTUAL_ENV}/bin:${PATH}

# Effectively disable `devtools.debug` doing its own line wraping.
export PY_DEVTOOLS_WIDTH=10000

.PHONY: env
env:
	@/bin/bash -c "declare -px"
