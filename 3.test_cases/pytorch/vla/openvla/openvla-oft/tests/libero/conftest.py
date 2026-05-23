"""Shared pytest fixtures and helpers for the LIBERO recipe tests.

This module backs the property-based test suite under
``3.test_cases/pytorch/vla/openvla/openvla-oft/tests/libero/``. The primary
export is :func:`render`, which shells out to the ``envsubst`` binary with a
caller-supplied environment dict so tests are decoupled from the user's
ambient shell. Supporting helpers and path constants make it easy for tests
to locate the recipe's manifests on disk.

Properties supported: P2 (fail-loud on unset ``TASK_SUITE``),
P3 (distinct ``metadata.name`` per suite), P4 (``verify-tfds-layout.sh``
accepts well-formed TFDS trees), P5 (``verify-tfds-layout.sh`` rejects
malformed trees), P7 (render + apply round-trip).

The :func:`tfds_tree_wellformed` and :func:`tfds_tree_malformed` fixtures
(Task 6.3) build controllable ``/data/datasets/rlds/<suite>/1.0.0/`` trees
under a pytest tmpdir so P4 and P5 can exercise ``verify-tfds-layout.sh``
without touching a real ``Dataset_PVC``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest
import yaml

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

# ``conftest.py`` lives at <test-case>/tests/libero/conftest.py, so the test
# case root (the directory that holds ``Dockerfile`` + ``kubernetes/``) is
# three parents up.
TESTS_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = TESTS_DIR.parent.parent
MANIFEST_DIR: Path = REPO_ROOT / "kubernetes" / "libero"


# ---------------------------------------------------------------------------
# envsubst render helper
# ---------------------------------------------------------------------------


def _locate_envsubst() -> str:
    """Return the absolute path to the ``envsubst`` executable.

    Raises:
        RuntimeError: when ``envsubst`` is not on the ``PATH``. Tests that
            depend on the real binary should either rely on this error or
            call :func:`shutil.which` themselves and ``pytest.skip`` with a
            message tailored to the caller's CI environment.
    """
    path = shutil.which("envsubst")
    if path is None:
        raise RuntimeError(
            "envsubst not found on PATH. Install GNU gettext "
            "(`brew install gettext` on macOS, "
            "`apt-get install gettext-base` on Debian/Ubuntu) "
            "and ensure the binary is on PATH before running these tests."
        )
    return path


def render(env: Mapping[str, str], manifest_path: Path) -> str:
    """Render a manifest through ``envsubst`` with a clean environment.

    The subprocess receives only the caller-supplied ``env`` dict plus
    ``PATH`` (so the ``envsubst`` binary itself can be found). This keeps
    the test outcome independent of whatever variables happen to be exported
    in the developer's shell when ``pytest`` is invoked.

    Args:
        env: Mapping of environment variable names to values that
            ``envsubst`` should substitute into the manifest. Keys in this
            mapping take precedence over the inherited ``PATH``.
        manifest_path: Path to a YAML manifest that contains ``${VAR}`` /
            ``$VAR`` placeholders to be substituted.

    Returns:
        The rendered manifest text (stdout from ``envsubst``).

    Raises:
        RuntimeError: when the ``envsubst`` binary is unavailable.
        FileNotFoundError: when ``manifest_path`` does not exist.
        subprocess.CalledProcessError: when ``envsubst`` exits non-zero.
    """
    envsubst = _locate_envsubst()
    manifest_path = Path(manifest_path)
    manifest_text = manifest_path.read_text()

    # Build the hermetic environment: start with PATH (so the binary is
    # reachable) and layer the caller's dict on top so tests can override
    # PATH when they need to.
    clean_env: dict[str, str] = {"PATH": os.environ.get("PATH", "")}
    clean_env.update({str(k): str(v) for k, v in env.items()})

    result = subprocess.run(
        [envsubst],
        input=manifest_text,
        capture_output=True,
        text=True,
        env=clean_env,
        check=True,
    )
    return result.stdout


def render_and_load(env: Mapping[str, str], manifest_path: Path) -> list[dict[str, Any]]:
    """Render a manifest via :func:`render` and parse it as multi-doc YAML.

    ``kubernetes/libero/`` ships single-document and (potentially)
    multi-document manifests. ``yaml.safe_load_all`` handles both cases;
    this helper filters out the ``None`` documents that can result from
    a trailing ``---`` separator.

    Args:
        env: Mapping passed through to :func:`render`.
        manifest_path: Path to the manifest to render and parse.

    Returns:
        A list of parsed YAML documents (each document is a ``dict``).
        Empty ``---`` separators are dropped.
    """
    rendered = render(env, manifest_path)
    docs = list(yaml.safe_load_all(rendered))
    return [doc for doc in docs if doc is not None]


# ---------------------------------------------------------------------------
# TFDS tmpdir fixtures (Task 6.3, Properties P4 and P5)
# ---------------------------------------------------------------------------

# The four LIBERO task suites the recipe supports, plus the deterministic
# default used when a test does not care which suite name it exercises.
LIBERO_SUITES: tuple[str, ...] = (
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops",
)
DEFAULT_SUITE: str = "libero_spatial_no_noops"

# The set of allowed defect tokens accepted by :func:`tfds_tree_malformed`.
# Kept as a module-level constant so tests can import and parametrize over
# it without duplicating the list.
MALFORMED_DEFECTS: tuple[str, ...] = (
    "no_suite_dir",
    "no_info",
    "no_features",
    "no_shard",
)


@dataclass(frozen=True)
class TfdsTree:
    """Handle describing a TFDS tree built under a pytest tmpdir.

    Attributes:
        root: Path that tests pass as ``LIBERO_DATA_ROOT`` to
            ``verify-tfds-layout.sh``. Always ``<tmp>/datasets/rlds``.
        suite: Name of the LIBERO task suite that was (or should have been)
            built under ``root``.
        suite_dir: Full path ``<root>/<suite>``. May not exist on disk for
            the ``no_suite_dir`` malformed variant.
    """

    root: Path
    suite: str
    suite_dir: Path


def _build_wellformed_tree(base: Path, suite: str) -> TfdsTree:
    """Build a minimal well-formed TFDS tree under ``base``.

    Creates ``<base>/datasets/rlds/<suite>/1.0.0/`` populated with
    ``dataset_info.json``, ``features.json``, and one
    ``<suite>-train.tfrecord-00000-of-00001`` shard file. The contents are
    deliberately tiny — the recipe's ``verify-tfds-layout.sh`` only checks
    for file presence and the ``*.tfrecord-*`` name pattern, not for valid
    TFDS serialisation.

    Args:
        base: The tmpdir under which the ``datasets/rlds/`` subtree is
            created. Caller owns the lifetime of this directory.
        suite: LIBERO task-suite name; becomes a directory under
            ``datasets/rlds/``.

    Returns:
        A :class:`TfdsTree` describing the built layout.
    """
    root = base / "datasets" / "rlds"
    suite_dir = root / suite
    version_dir = suite_dir / "1.0.0"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Minimal TFDS-shaped metadata. verify-tfds-layout.sh only checks
    # presence, so empty JSON documents are sufficient.
    (version_dir / "dataset_info.json").write_text("{}\n")
    (version_dir / "features.json").write_text("{}\n")

    # One shard file matching the *.tfrecord-* glob the verify script
    # searches for.
    shard = version_dir / f"{suite}-train.tfrecord-00000-of-00001"
    shard.write_bytes(b"")

    return TfdsTree(root=root, suite=suite, suite_dir=suite_dir)


@pytest.fixture
def tfds_tree_wellformed(
    tmp_path: Path,
) -> Callable[..., TfdsTree]:
    """Pytest fixture returning a factory that builds a well-formed TFDS tree.

    The factory signature is ``factory(suite: str = DEFAULT_SUITE) -> TfdsTree``.
    Each call builds an independent subdirectory under the same pytest
    ``tmp_path`` so a single test can materialize multiple suites side by
    side (e.g. to exercise ``TASK_SUITE=all`` behavior).

    Example:

        def test_something(tfds_tree_wellformed):
            tree = tfds_tree_wellformed()
            # tree.root can be passed as LIBERO_DATA_ROOT.
            assert (tree.suite_dir / "1.0.0" / "dataset_info.json").is_file()

    Drives: Property P4 (``verify-tfds-layout.sh`` accepts well-formed
    trees).
    """
    counter = {"n": 0}

    def _factory(suite: str = DEFAULT_SUITE) -> TfdsTree:
        # Each invocation gets its own subtree so the fixture is safe to
        # call multiple times per test.
        counter["n"] += 1
        base = tmp_path / f"wellformed-{counter['n']}"
        base.mkdir(parents=True, exist_ok=True)
        return _build_wellformed_tree(base, suite)

    return _factory


@pytest.fixture
def tfds_tree_malformed(
    tmp_path: Path,
) -> Callable[..., TfdsTree]:
    """Pytest fixture returning a factory that builds a malformed TFDS tree.

    The factory signature is
    ``factory(defect: str = "no_shard", suite: str = DEFAULT_SUITE) -> TfdsTree``.

    ``defect`` selects which part of the well-formed layout is intentionally
    left out. Accepted values (exposed as ``MALFORMED_DEFECTS``):

    - ``"no_suite_dir"``: ``<root>/<suite>/`` is not created at all. The
      suite directory simply does not exist.
    - ``"no_info"``: the suite directory and shard exist, but
      ``dataset_info.json`` is absent.
    - ``"no_features"``: the suite directory, ``dataset_info.json``, and
      shard exist, but ``features.json`` is absent. (Not strictly required
      by ``verify-tfds-layout.sh``; provided so tests that want to probe
      the finer-grained TFDS schema have a knob.)
    - ``"no_shard"``: the suite directory and ``dataset_info.json`` exist,
      but no ``*.tfrecord-*`` shard file is present.

    An unknown ``defect`` raises :class:`ValueError` to make test typos
    fail fast rather than silently producing a well-formed tree.

    Example:

        def test_missing_shard(tfds_tree_malformed):
            tree = tfds_tree_malformed(defect="no_shard")
            # Running verify-tfds-layout.sh against tree.root must exit 1.

    Drives: Property P5 (``verify-tfds-layout.sh`` rejects malformed trees).
    """
    counter = {"n": 0}

    def _factory(defect: str = "no_shard", suite: str = DEFAULT_SUITE) -> TfdsTree:
        if defect not in MALFORMED_DEFECTS:
            raise ValueError(f"Unknown defect {defect!r}; expected one of " f"{MALFORMED_DEFECTS!r}")

        counter["n"] += 1
        base = tmp_path / f"malformed-{counter['n']}"
        base.mkdir(parents=True, exist_ok=True)

        root = base / "datasets" / "rlds"
        suite_dir = root / suite

        if defect == "no_suite_dir":
            # Create only the data root so verify-tfds-layout.sh finds the
            # parent but fails on the missing suite directory.
            root.mkdir(parents=True, exist_ok=True)
            return TfdsTree(root=root, suite=suite, suite_dir=suite_dir)

        # For the remaining defects, start from a well-formed tree and
        # selectively remove one piece.
        tree = _build_wellformed_tree(base, suite)
        version_dir = tree.suite_dir / "1.0.0"

        if defect == "no_info":
            (version_dir / "dataset_info.json").unlink()
        elif defect == "no_features":
            (version_dir / "features.json").unlink()
        elif defect == "no_shard":
            for shard in version_dir.glob("*.tfrecord-*"):
                shard.unlink()

        return tree

    return _factory
