"""Unit test for Property 8: Training_Job fails fast on an empty or malformed PVC.

This module exercises the upstream :class:`prismatic.vla.datasets.RLDSDataset`
directly against an empty tmpdir rather than a populated ``Dataset_PVC``. It
resolves design open question #4 by picking option (b): a lightweight unit
test that instantiates ``RLDSDataset(data_root_dir=<empty_tmpdir>,
dataset_name=<suite>)`` and asserts that the first access raises quickly,
and that the resulting exception message names a missing data path.

The test is *optional* from the spec's perspective and is expected to
``pytest.skip`` in environments where the ``prismatic`` package is not
installed (that is the standard case for this test-case's repo-level CI,
which does not pull in the OpenVLA-OFT training stack just to lint
manifests). When ``prismatic`` IS importable (e.g. inside the project's
training container image), the test runs end-to-end.

**Validates: Requirements 10.1**

**Feature: openvla-oft-libero-recipe, Property 8: Training_Job fails fast
on an empty or malformed PVC**

Notes:
    * The ``prismatic`` import is deliberately placed INSIDE the test
      function. Import failures at module scope would be collected as
      errors rather than skips, which would make the test suite fail on
      any environment without the training stack.
    * The test adds an ``@pytest.mark.timeout(30)`` marker that is only
      honoured when ``pytest-timeout`` is installed. When it is not, the
      test still relies on ``RLDSDataset`` raising immediately on a
      missing data directory, so the upper bound is enforced implicitly
      by the library under test.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path fragment that MUST appear in the exception message, per Req 10.1 and
# Property 8. The training recipe mounts LIBERO RLDS shards under
# ``/data/datasets/rlds/<task_suite>/``; when that tree is missing, the
# Training_Job's stderr must name a path under ``/data/datasets/rlds/`` so
# operators can triage quickly.
#
# We accept ANY substring of the canonical path as evidence the exception
# refers to the dataset root. Matching only ``rlds`` keeps the test robust
# to different formatting (quoted vs unquoted, absolute vs tmpdir-relative)
# while still failing on messages that are silent about which path is
# missing.
_EXPECTED_PATH_FRAGMENTS: tuple[str, ...] = (
    "rlds",
    "datasets",
)

# Suite name used for the probe. Any of the four LIBERO suites works; we
# pick the canonical one to keep the test deterministic.
_PROBE_SUITE: str = "libero_spatial_no_noops"

# Upper bound on how long ``RLDSDataset`` may take to raise on a missing
# data directory. The spec's Req 10.1 allows ten minutes for a real pod
# start; at unit-test scope we expect *immediate* failure (sub-second on a
# warm Python interpreter). Thirty seconds gives generous headroom for
# cold-cache TFDS import on CI while still catching a regression where
# the library starts blocking on a network fetch.
_FAILFAST_BUDGET_SECONDS: float = 30.0


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_rlds_dataset_fails_fast_on_empty_tmpdir(tmp_path: Path) -> None:
    """RLDSDataset on an empty tmpdir raises quickly and names a missing path.

    Steps:
        1. Try to import ``RLDSDataset`` from ``prismatic.vla.datasets``.
           If the import fails, skip the test (expected outcome in any
           environment that does not ship the training stack).
        2. Build an empty tmpdir to stand in for an unprovisioned
           ``Dataset_PVC``.
        3. Invoke ``RLDSDataset(data_root_dir=<tmpdir>,
           dataset_name=<suite>, ...)`` and/or its first iterator access,
           measuring wall-clock time.
        4. Assert an exception is raised within
           :data:`_FAILFAST_BUDGET_SECONDS`.
        5. Assert the exception message mentions a path fragment from
           :data:`_EXPECTED_PATH_FRAGMENTS`, so the operator-facing log
           line satisfies Req 10.1.

    Args:
        tmp_path: pytest-provided empty tmpdir; role is played by
            ``data_root_dir``.
    """
    # Step 1: import inside the test so module collection stays green on
    # environments without ``prismatic``.
    try:
        datasets_mod = importlib.import_module("prismatic.vla.datasets")
    except ImportError:
        pytest.skip("prismatic not installed in this environment")

    rlds_cls = getattr(datasets_mod, "RLDSDataset", None)
    if rlds_cls is None:
        pytest.skip("prismatic.vla.datasets.RLDSDataset not found in this prismatic install")

    # Step 2: the tmpdir is intentionally empty, so no
    # ``<tmpdir>/<suite>/1.0.0/`` TFDS subtree exists. This mirrors a
    # Dataset_PVC that was mounted but never populated by the Download_Job.
    data_root_dir = tmp_path
    assert data_root_dir.is_dir()
    assert not any(data_root_dir.iterdir()), "tmp_path should start empty"

    # Step 3 & 4: time the constructor + first access. Different
    # prismatic releases do the TFDS open in __init__ vs. on first
    # __iter__ / __getitem__; wrap both in the same try/except so the
    # test is agnostic to that detail.
    start = time.monotonic()
    error: BaseException | None = None
    try:
        # Minimal positional/keyword surface that matches the published
        # RLDSDataset signature. Extra required kwargs that the upstream
        # constructor has added in later releases are tolerated: we only
        # rely on ``data_root_dir`` + ``dataset_name`` pointing at a path
        # that does not exist, which MUST raise regardless of the other
        # arguments' values.
        dataset = rlds_cls(
            data_root_dir=str(data_root_dir),
            data_mix=_PROBE_SUITE,
            image_transform=lambda x: x,
            tokenizer=None,
            prompt_builder_fn=None,
        )
        # If construction somehow succeeded, force a read to surface the
        # missing-path error.
        iter(dataset).__next__()
    except TypeError:
        # Fallback to the simpler published signature
        # ``RLDSDataset(data_root_dir, dataset_name, ...)`` used by older
        # revisions. A TypeError here means the keyword set above no
        # longer matches; retry with positionals.
        try:
            dataset = rlds_cls(str(data_root_dir), _PROBE_SUITE)
            iter(dataset).__next__()
        except Exception as exc:  # noqa: BLE001 - we want the broadest catch
            error = exc
    except Exception as exc:  # noqa: BLE001 - we want the broadest catch
        error = exc
    elapsed = time.monotonic() - start

    # Step 4: assert we got an error within the budget.
    assert error is not None, (
        "RLDSDataset did not raise when data_root_dir was empty; " "Property 8 (fail-fast on an empty PVC) is violated."
    )
    assert elapsed < _FAILFAST_BUDGET_SECONDS, (
        f"RLDSDataset took {elapsed:.2f}s to raise, " f"exceeding the {_FAILFAST_BUDGET_SECONDS:.0f}s fail-fast budget."
    )

    # Step 5: assert the error message names a dataset path. We check
    # against both ``str(error)`` and ``repr(error)`` so wrapper
    # exceptions (TFDS ``DatasetNotFoundError`` wrapping an OSError, for
    # example) that format the path in their repr rather than their str
    # still pass.
    msg = f"{error!s}\n{error!r}"
    matched = [frag for frag in _EXPECTED_PATH_FRAGMENTS if frag in msg]
    assert matched, (
        f"RLDSDataset raised {type(error).__name__} but its message did not "
        f"mention any of {_EXPECTED_PATH_FRAGMENTS!r}. "
        f"Req 10.1 requires the log line to name the missing path under "
        f"/data/datasets/rlds/. Actual message: {msg!r}"
    )
