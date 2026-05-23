#!/usr/bin/env bash
# Smoke test for kubernetes/libero/verify-tfds-layout.sh.
#
# Exercises the happy path and the two malformed-tree failure modes using
# only bash, so it runs without Python or pytest. This is the Layer 3
# smoke test from the openvla-oft-libero-recipe design.
#
# Covers:
#   Requirements 3.2, 3.3
#   Properties P4 (well-formed accepted), P5 (malformed rejected + FAIL: stderr)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_SCRIPT="${SCRIPT_DIR}/../../kubernetes/libero/verify-tfds-layout.sh"

if [[ ! -f "${VERIFY_SCRIPT}" ]]; then
  echo "smoke: FAIL: expected verify script at ${VERIFY_SCRIPT}" >&2
  exit 1
fi

SUITE="libero_spatial_no_noops"

TMPDIR_ROOT="$(mktemp -d)"
cleanup() {
  rm -rf -- "${TMPDIR_ROOT}"
}
trap cleanup EXIT

DATA_ROOT="${TMPDIR_ROOT}/datasets/rlds"
SUITE_DIR="${DATA_ROOT}/${SUITE}/1.0.0"
DATASET_INFO="${SUITE_DIR}/dataset_info.json"
SHARD="${SUITE_DIR}/${SUITE}-train.tfrecord-00000-of-00001"

populate_tree() {
  mkdir -p -- "${SUITE_DIR}"
  cat > "${DATASET_INFO}" <<'JSON'
{"name": "libero_spatial_no_noops", "version": "1.0.0"}
JSON
  : > "${SHARD}"
}

# Run verify-tfds-layout.sh against the tmpdir, capturing exit code and stderr.
# Usage: run_verify <stderr_file>
run_verify() {
  local stderr_file="$1"
  local rc=0
  LIBERO_DATA_ROOT="${DATA_ROOT}" bash "${VERIFY_SCRIPT}" "${SUITE}" \
    >/dev/null 2>"${stderr_file}" || rc=$?
  printf '%s' "${rc}"
}

fail() {
  echo "smoke: FAIL: $*" >&2
  exit 1
}

# -----------------------------------------------------------------------------
# Step 1: populate a well-formed tmpdir tree.
# -----------------------------------------------------------------------------
populate_tree

# -----------------------------------------------------------------------------
# Step 2: well-formed tree -> exit 0.
# -----------------------------------------------------------------------------
STDERR_OK="$(mktemp)"
trap 'rm -f -- "${STDERR_OK}" "${STDERR_NO_SHARD:-}" "${STDERR_NO_INFO:-}"; cleanup' EXIT
RC="$(run_verify "${STDERR_OK}")"
if [[ "${RC}" != "0" ]]; then
  echo "----- stderr from verify (step 2, expected exit 0) -----" >&2
  cat "${STDERR_OK}" >&2
  fail "well-formed tree rejected (exit ${RC})"
fi

# -----------------------------------------------------------------------------
# Step 3: remove the *.tfrecord-* shard -> exit 1, stderr names the miss.
# -----------------------------------------------------------------------------
rm -f -- "${SHARD}"
STDERR_NO_SHARD="$(mktemp)"
RC="$(run_verify "${STDERR_NO_SHARD}")"
if [[ "${RC}" != "1" ]]; then
  echo "----- stderr from verify (step 3, expected exit 1) -----" >&2
  cat "${STDERR_NO_SHARD}" >&2
  fail "tree missing *.tfrecord-* shard did not exit 1 (exit ${RC})"
fi
if ! grep -qE "FAIL:|${SUITE}|/datasets/rlds/" "${STDERR_NO_SHARD}"; then
  echo "----- stderr from verify (step 3) -----" >&2
  cat "${STDERR_NO_SHARD}" >&2
  fail "stderr for missing-shard case did not reference FAIL: or the missing path"
fi

# -----------------------------------------------------------------------------
# Step 4: also remove dataset_info.json -> exit 1.
# -----------------------------------------------------------------------------
rm -f -- "${DATASET_INFO}"
STDERR_NO_INFO="$(mktemp)"
RC="$(run_verify "${STDERR_NO_INFO}")"
if [[ "${RC}" != "1" ]]; then
  echo "----- stderr from verify (step 4, expected exit 1) -----" >&2
  cat "${STDERR_NO_INFO}" >&2
  fail "tree missing dataset_info.json did not exit 1 (exit ${RC})"
fi

echo "smoke: OK"
exit 0
