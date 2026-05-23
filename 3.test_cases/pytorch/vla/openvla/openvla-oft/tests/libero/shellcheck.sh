#!/usr/bin/env bash
# shellcheck.sh — lint kubernetes/libero/env_vars with shellcheck.
#
# The env_vars file is meant to be `source`d, not executed, and has no
# shebang. shellcheck therefore needs an explicit `--shell=bash` hint.
#
# shellcheck isn't a hard dependency for this recipe: the fail-loud
# behaviour of `${VAR:?msg}` is already covered by Property 2 in
# tests/libero/test_properties.py. If shellcheck isn't installed locally
# (e.g. minimal CI runners), this script prints a notice to stderr and
# exits 0 so it doesn't block the rest of the test suite. Install it via
# `apt-get install shellcheck`, `brew install shellcheck`, or equivalent
# to get linting.
#
# Usage:
#   tests/libero/shellcheck.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_VARS="${SCRIPT_DIR}/../../kubernetes/libero/env_vars"

if ! command -v shellcheck >/dev/null 2>&1; then
    echo "shellcheck not installed, skipping lint of ${ENV_VARS}" >&2
    exit 0
fi

if [[ ! -f "${ENV_VARS}" ]]; then
    echo "ERROR: ${ENV_VARS} does not exist" >&2
    exit 1
fi

# --shell=bash because env_vars uses bash parameter expansion (`${VAR:?msg}`)
# and has no shebang for shellcheck to sniff.
# --external-sources lets shellcheck follow any future `source` directives
# without failing on SC1091.
exec shellcheck --shell=bash --external-sources "${ENV_VARS}"
