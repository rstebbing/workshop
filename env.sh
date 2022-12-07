##########################################
# File: env.sh                           #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

SRC_ROOT=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )


function rm_script {
    if [ -n "${script:-}" ]; then
        rm "${script}"
    fi
}

trap rm_script EXIT

script=$(mktemp)

make --file "${SRC_ROOT}/env.mk" \
     --no-print-directory \
     env > "${script}"

# shellcheck disable=SC1090
source "${script}"

ps1=${PS1:-}
if [[ "${ps1}" != "(workshop)"* ]]; then
    x=${ps1}
    if [ -n "${x}" ]; then
        x=" ${x}"
    fi

    export PS1="(workshop)${x}"
fi
