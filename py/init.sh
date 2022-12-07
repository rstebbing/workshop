#!/bin/bash

set -euo pipefail


python3=${PYTHON3:-python3}

opt_py=$1


if ! "${python3}" -c 'import sys; sys.exit(0 if (sys.version_info.major, sys.version_info.minor) >= (3, 10) else 1)' 2>/dev/null; then
    python3_path=$(command -v "${python3}" 2>/dev/null || true)
    python3_version=$("${python3}" -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))' 2>/dev/null || true)

    >&2 echo "\
Error: Python 3.10 or higher is required but the \"${python3}\" interpreter at \"${python3_path}\" has version \"${python3_version}\".

To use a different interpreter, set the PYTHON3 environment variable and re-run make. For example:

  export PYTHON3=python3.10
  make

This only needs to be done once to initialize the virtual environment."
    exit 1
fi


set -x

rm -rf "${opt_py}"
"${python3}" -m venv "${opt_py}"
