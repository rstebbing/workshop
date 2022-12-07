##########################################
# File: Makefile                         #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

.PHONY: all
all: py

include env.mk

PY_INIT := ${OPT_PY}/.init

${PY_INIT}: py/init.sh
	./py/init.sh ${OPT_PY}
	@touch $@

PY_SETUP := ${OPT_PY}/.setup

${PY_SETUP}: ${PY_INIT} py/setup.sh py/setup.py
	./py/setup.sh
	@touch $@

PY_PRE_COMMIT := ${OPT_PY}/.pre_commit

${PY_PRE_COMMIT}: ${PY_SETUP}
	pre-commit install
	@touch $@

PY := ${OPT_PY}/.py

${PY}: ${PY_PRE_COMMIT}
	@touch $@

.PHONY: py
py: ${PY}

.PHONY: test
test: ${PY}
	python3 -m pytest -sv py/tests
