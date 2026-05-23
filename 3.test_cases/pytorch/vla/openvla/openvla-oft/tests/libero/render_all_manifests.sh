#!/usr/bin/env bash
# render_all_manifests.sh — Testing Strategy Layer 1 for the LIBERO recipe.
#
# Sources the test-only env_vars fixture, then renders every manifest in
#
#     3.test_cases/pytorch/vla/openvla/openvla-oft/kubernetes/libero/*.yaml
#
# through `envsubst` and pipes the result to `kubectl apply --dry-run=client
# -f -`. This catches YAML structural errors and Kubernetes schema drift
# before any property test runs (Req 1.1, 5.1, 10.6).
#
# Behaviour:
#   * `set -euo pipefail` -> the first validation failure aborts the script
#     with a non-zero exit code.
#   * A one-line banner ("validating: <file>") precedes each render so CI
#     logs name the offending manifest on failure.
#   * If `kubectl` is unavailable (not on PATH, or on PATH but unable to
#     reach an API server), the script falls back to a pure-Python YAML
#     parse via `python3 -c 'import sys,yaml; list(yaml.safe_load_all(
#     sys.stdin))'`. That degraded mode still catches YAML structural
#     errors, which is the core value of Layer 1.
#
# Usage:
#   bash tests/libero/render_all_manifests.sh
#
# or (from the repo root):
#   bash 3.test_cases/pytorch/vla/openvla/openvla-oft/tests/libero/render_all_manifests.sh

set -euo pipefail

# Resolve this script's directory, the fixture it sources, and the manifest
# directory it iterates over — all via paths relative to this script, so the
# script works regardless of the caller's cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE="${SCRIPT_DIR}/fixtures/env_vars.test"
MANIFEST_DIR="${SCRIPT_DIR}/../../kubernetes/libero"

if [[ ! -f "${FIXTURE}" ]]; then
    echo "ERROR: env_vars test fixture not found: ${FIXTURE}" >&2
    exit 1
fi

if [[ ! -d "${MANIFEST_DIR}" ]]; then
    echo "ERROR: manifest directory not found: ${MANIFEST_DIR}" >&2
    exit 1
fi

if ! command -v envsubst >/dev/null 2>&1; then
    echo "ERROR: envsubst is required but not on PATH" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${FIXTURE}"

# ---- Pick the validator -----------------------------------------------------
#
# Preferred path: `kubectl apply --dry-run=client -f -`.
#
# Fallback path: when kubectl is not on PATH, or kubectl is on PATH but
# cannot reach an API server (e.g. in a minimal CI container or an
# uncredentialed dev laptop), use python3 + PyYAML to at least confirm the
# rendered stream parses as YAML. This is the documented fallback from the
# openvla-oft-libero-recipe design / tasks (Task 8.1).
USE_KUBECTL=0
if command -v kubectl >/dev/null 2>&1; then
    # A short-timeout cluster probe: if kubectl can reach the API server
    # within a few seconds we trust `kubectl apply --dry-run=client` to
    # give us both schema validation and structural validation. If it
    # can't, we fall through to the python fallback.
    if kubectl cluster-info --request-timeout=3s >/dev/null 2>&1; then
        USE_KUBECTL=1
    else
        echo "note: kubectl on PATH but cannot reach a cluster; falling back to python YAML parse" >&2
    fi
else
    echo "note: kubectl not on PATH; falling back to python YAML parse" >&2
fi

if [[ "${USE_KUBECTL}" -eq 0 ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: neither kubectl (with a reachable cluster) nor python3 is available" >&2
        exit 1
    fi
    if ! python3 -c 'import yaml' >/dev/null 2>&1; then
        echo "ERROR: python3 PyYAML module not available for the fallback validator" >&2
        exit 1
    fi
fi

validate_manifest() {
    local file="$1"
    if [[ "${USE_KUBECTL}" -eq 1 ]]; then
        envsubst < "${file}" | kubectl apply --dry-run=client -f -
    else
        # Parse every YAML document in the stream. PyYAML raises on
        # malformed YAML and exits non-zero, which propagates out under
        # `set -o pipefail`.
        envsubst < "${file}" | python3 -c 'import sys, yaml; list(yaml.safe_load_all(sys.stdin))'
    fi
}

# ---- Iterate over manifests -------------------------------------------------
shopt -s nullglob
manifests=("${MANIFEST_DIR}"/*.yaml)
shopt -u nullglob

if [[ "${#manifests[@]}" -eq 0 ]]; then
    echo "note: no *.yaml manifests under ${MANIFEST_DIR}; nothing to validate" >&2
    exit 0
fi

for manifest in "${manifests[@]}"; do
    echo "validating: ${manifest}"
    validate_manifest "${manifest}"
done

echo "render_all_manifests: OK (${#manifests[@]} manifests validated)"
