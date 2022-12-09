# Workshop

*Workshop* is a repository of simple, documented, and locally reproducible experiments.

The primary purpose of this repository is to support learning: all experiments are intended to be easy to tweak, run quickly, and interrogate. Errors in code and/or reports should be reported so that they can be fixed too.

License: MIT (refer to LICENSE)

## Dependencies

This repository is tested to work under Python 3.10 on macOS Ventura and Ubuntu 20.04.

## Getting Started

*workshop* is implemented as a single Python package and can be set up in its own virtual environment via:

``` bash
git clone git@github.com:rstebbing/workshop.git
cd workshop
source env.sh
export PYTHON3=python3.10
make
```

All tests can be run via:

``` bash
make test
```

## Experiments

- [Transformer Sequence Walkthrough ⧉](experiments/transformer_sequence_classification)
