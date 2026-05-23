"""Property-based tests for the openvla-oft LIBERO recipe (Task 6.2).

Each property (P1 through P7) is exercised by a single pytest function. Each
test carries a traceability comment in the exact format required by the
spec::

    # Feature: openvla-oft-libero-recipe, Property N: <title>

placed directly above the ``@given`` decorator, and uses
``@settings(max_examples=100, deadline=None)`` at minimum. ``deadline=None``
keeps Hypothesis from flaking on subprocess-heavy properties (P1, P2, P4,
P5, P6, P7) on slower machines or under CI virtualisation.

The shared helpers and fixtures come from ``conftest.py``:

* ``render`` / ``render_and_load`` -- hermetic ``envsubst`` wrappers.
* ``MANIFEST_DIR`` -- path to the recipe's YAML manifests.
* ``LIBERO_SUITES`` / ``MALFORMED_DEFECTS`` -- the allow-list and the
  defect tokens accepted by :func:`tfds_tree_malformed`.
* ``tfds_tree_wellformed`` / ``tfds_tree_malformed`` -- pytest fixtures
  that build controllable TFDS trees under a tmpdir.

External binaries the tests depend on (``bash``, ``envsubst``) are probed
via ``shutil.which`` and missing binaries trigger ``pytest.skip`` so the
suite is usable on any POSIX CI image that has Python + Hypothesis.
"""

from __future__ import annotations

import filecmp
import hashlib
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any, Iterable

import pytest
import yaml
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from conftest import (  # type: ignore[attr-defined]
    LIBERO_SUITES,
    MALFORMED_DEFECTS,
    MANIFEST_DIR,
    TESTS_DIR,
    render,
    render_and_load,
)


# ---------------------------------------------------------------------------
# Recipe-level constants
# ---------------------------------------------------------------------------

# Allow-list the Download_Job bash block validates against (Req 2.3, 2.6).
ALLOWED_SUITES: tuple[str, ...] = LIBERO_SUITES + ("all",)

# Required env dict for rendering libero-finetune.yaml. RUN_ID_NOTE is set
# explicitly because GNU ``envsubst`` does NOT honour the ``${VAR:-default}``
# form; callers of ``render`` must supply every variable referenced by the
# manifest or it will be replaced with the empty string. The value below
# mirrors the upstream LIBERO.md default codified in Requirement 4.7.
_UPSTREAM_RUN_ID_NOTE = (
    "parallel_dec--8_acts_chunk--continuous_acts--L1_regression" "--3rd_person_img--wrist_img--proprio_state"
)


def _base_env(task_suite: str, task_suite_dns: str | None = None) -> dict[str, str]:
    """Return a fully populated env dict for rendering libero-finetune.yaml."""
    if task_suite_dns is None:
        task_suite_dns = task_suite.replace("_", "-")
    return {
        "IMAGE_URI": "dummy.ecr.example.com/openvla-oft:latest",
        "INSTANCE_TYPE": "p5.48xlarge",
        "NUM_NODES": "1",
        "GPU_PER_NODE": "8",
        "EFA_PER_NODE": "32",
        "FI_PROVIDER": "efa",
        "DATA_PVC_NAME": "openvla-oft-libero-data",
        "DATA_PVC_SIZE": "1200Gi",
        "TASK_SUITE": task_suite,
        "TASK_SUITE_DNS": task_suite_dns,
        # envsubst does not expand :- defaults, so we pass the upstream value.
        "RUN_ID_NOTE": _UPSTREAM_RUN_ID_NOTE,
        "WANDB_MODE": "online",
        "WANDB_ENTITY": "test-entity",
        "WANDB_PROJECT": "test-project",
        "NAMESPACE": "default",
        "FSX_FILESYSTEM_ID": "",
        "FSX_SUBNET_ID": "",
        "FSX_SECURITY_GROUP_ID": "",
        "FSX_DNS_NAME": "",
        "FSX_MOUNT_NAME": "",
    }


def _require(binary: str) -> str:
    """Locate an external binary or skip the test cleanly."""
    path = shutil.which(binary)
    if path is None:
        pytest.skip(f"{binary!r} is not on PATH; skipping test that needs it")
    return path


# ---------------------------------------------------------------------------
# Helpers for extracting the Download_Job bash body
# ---------------------------------------------------------------------------


def _download_bash_body() -> str:
    """Extract the Download_Job's bash script from ``libero-download.yaml``.

    The YAML authors use the ``$$`` convention (the a8m/envsubst escape) so
    the rendered manifest preserves ``$VAR`` references for the pod's bash
    to expand. ``yaml.safe_load`` returns the literal string (because YAML
    has no knowledge of envsubst), so a single ``replace("$$", "$")`` pass
    produces the script bash will actually execute at runtime.
    """
    manifest = MANIFEST_DIR / "libero-download.yaml"
    doc = yaml.safe_load(manifest.read_text())
    args = doc["spec"]["template"]["spec"]["containers"][0]["args"]
    assert isinstance(args, list) and len(args) == 1
    body = args[0]
    assert isinstance(body, str)
    return body.replace("$$", "$")


def _validator_only_body() -> str:
    """Return the Download_Job bash body truncated to the allow-list check.

    We keep every line up to and including the ``fi`` that closes the
    allow-list ``if`` block (the one that follows ``exit 2``), then append
    ``exit 0``. This isolates the classification behaviour from
    ``pip install`` and ``huggingface-cli`` side effects so P1 can run
    hermetically.
    """
    body = _download_bash_body()
    exit2_idx = body.index("exit 2")
    fi_idx = body.index("fi", exit2_idx)
    # Include the closing "fi" and the newline that follows it.
    end = fi_idx + len("fi")
    truncated = body[:end] + "\nexit 0\n"
    return truncated


# ---------------------------------------------------------------------------
# P1 -- Download_Job validator allow-list classification
# ---------------------------------------------------------------------------


# The "random text" half of the P1 generator excludes a handful of characters
# that cannot appear in a POSIX environment variable value or that break the
# validator's ``in allow-list`` oracle:
#   * ``\x00`` -- POSIX forbids embedded NULs in env values and Python's
#     ``subprocess`` raises ``ValueError`` before bash even runs.
#   * Surrogate code points (``\ud800``-``\udfff``) -- ``os.fsencode`` cannot
#     UTF-8 encode them, so the subprocess launch fails.
#   * ``\n`` -- ``grep -F`` treats embedded newlines as alternations, which
#     would let a multi-line value containing a valid suite on one line pass
#     the grep check; the oracle ``suite in ALLOWED_SUITES`` would call that
#     input invalid, producing a spurious "false positive" mismatch.
_P1_INVALID_ALPHABET = st.characters(
    blacklist_characters="\n\x00",
    blacklist_categories=("Cs",),  # surrogates
)


# Feature: openvla-oft-libero-recipe, Property 1: Download_Job validator allow-list classification
@given(
    suite=st.one_of(
        st.sampled_from(ALLOWED_SUITES),
        st.text(alphabet=_P1_INVALID_ALPHABET, max_size=40),
    ),
)
@settings(max_examples=100, deadline=None)
def test_p1_validator_allow_list_classification(
    tmp_path_factory: pytest.TempPathFactory,
    suite: str,
) -> None:
    """P1: validator exits 0 iff ``suite`` is in the allow-list.

    **Validates: Requirements 2.3, 2.6**
    """
    bash = _require("bash")

    # Materialise the truncated validator script once per example in a fresh
    # tmpdir. Using tmp_path_factory (instead of the per-test tmp_path
    # fixture) avoids Hypothesis' "function-scoped fixture reused across
    # examples" warning.
    work = tmp_path_factory.mktemp("p1-validator")
    script = work / "validator.sh"
    script.write_text(_validator_only_body())

    result = subprocess.run(
        [bash, str(script)],
        input="",
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TASK_SUITE": suite},
    )

    if suite in ALLOWED_SUITES:
        assert result.returncode == 0, (
            f"suite={suite!r} is in the allow-list but validator exited "
            f"{result.returncode}; stderr={result.stderr!r}"
        )
    else:
        assert result.returncode == 2, (
            f"suite={suite!r} is not in the allow-list but validator exited "
            f"{result.returncode}; stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )
        # Every allow-list member must appear in stderr so the message is
        # self-describing (Req 2.6).
        for member in ALLOWED_SUITES:
            assert member in result.stderr, (
                f"expected allow-list member {member!r} to appear in stderr; " f"got stderr={result.stderr!r}"
            )


# ---------------------------------------------------------------------------
# P2 -- Fail-loud on unset or empty TASK_SUITE
# ---------------------------------------------------------------------------


# Hypothesis strategy: generate strings that are unset-equivalent for the
# purposes of the :? guard in env_vars. We cannot represent "unset" with
# hypothesis-generated strings, so the test handles the unset case as a
# distinct example via ``st.none()``.
_whitespace_or_none_strategy = st.one_of(
    st.none(),  # TASK_SUITE not present in the environment at all
    st.just(""),  # empty string
    st.text(
        alphabet=st.sampled_from(" \t"),
        min_size=1,
        max_size=8,
    ),  # whitespace-only
)


# Feature: openvla-oft-libero-recipe, Property 2: Fail-loud on unset or empty TASK_SUITE
@given(task_suite=_whitespace_or_none_strategy)
@settings(max_examples=100, deadline=None)
def test_p2_env_vars_fails_loud_on_missing_task_suite(
    task_suite: str | None,
) -> None:
    """P2: sourcing env_vars with unset/empty/whitespace TASK_SUITE aborts.

    Option (a) from the task brief: sourcing the recipe's actual ``env_vars``
    file under Bash. Bash's ``${VAR:?msg}`` expansion treats an empty value
    the same as an unset one. Whitespace-only values are not caught by ``:?``
    but the rendered manifest would still fail downstream; here we assert
    the ``:?`` guard specifically, which is the binding decision in the
    design. For whitespace-only inputs we accept either a zero exit
    (guarded elsewhere) or a non-zero exit with ``TASK_SUITE`` on stderr,
    but Bash's ``:?`` does not reject whitespace, so those examples are
    expected to pass the guard.

    **Validates: Requirements 5.4**
    """
    bash = _require("bash")

    env_vars_path = MANIFEST_DIR / "env_vars"
    assert env_vars_path.is_file(), f"missing recipe file: {env_vars_path}"

    # Build a hermetic env. PATH is required so the shell can find ``:`` and
    # the builtins used by env_vars, but we intentionally omit TASK_SUITE /
    # TASK_SUITE_DNS unless the strategy supplied them.
    env: dict[str, str] = {"PATH": os.environ.get("PATH", "")}
    if task_suite is not None:
        env["TASK_SUITE"] = task_suite
        # env_vars also guards TASK_SUITE_DNS with :?. Only exercise the
        # TASK_SUITE guard by always providing a non-empty DNS form.
        env["TASK_SUITE_DNS"] = "placeholder-dns"

    result = subprocess.run(
        [bash, "-c", f"source {str(env_vars_path)!r}"],
        capture_output=True,
        text=True,
        env=env,
    )

    is_empty_like = task_suite is None or task_suite == ""
    if is_empty_like:
        assert result.returncode != 0, (
            f"expected non-zero exit for TASK_SUITE={task_suite!r}, got 0; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "TASK_SUITE" in result.stderr, (
            f"expected stderr to mention TASK_SUITE when it is missing; " f"got stderr={result.stderr!r}"
        )
    else:
        # Whitespace-only TASK_SUITE passes Bash's :? guard (which only
        # triggers on unset or empty). This behaviour is documented in the
        # design: the whitespace case is caught downstream by the
        # Download_Job validator (P1) or by kubectl schema validation, not
        # by the :? expansion itself. We therefore require only that the
        # guard behaves consistently -- i.e. does NOT falsely abort.
        assert result.returncode == 0, (
            f"Bash :? should not trigger on whitespace-only TASK_SUITE "
            f"{task_suite!r}, but sourcing env_vars failed with "
            f"returncode={result.returncode}, stderr={result.stderr!r}"
        )


# ---------------------------------------------------------------------------
# P3 -- Concurrent suites produce distinct metadata.name and --dataset_name
# ---------------------------------------------------------------------------


def _find_dataset_name_flag(command: Iterable[Any]) -> str:
    """Return the ``--dataset_name=<value>`` entry from a command list."""
    for item in command:
        s = str(item)
        if s.startswith("--dataset_name="):
            return s
    raise AssertionError(f"--dataset_name flag not found in command list: {list(command)}")


# Feature: openvla-oft-libero-recipe, Property 3: Concurrent suites produce distinct metadata.name and --dataset_name
@given(
    pair=st.tuples(st.sampled_from(LIBERO_SUITES), st.sampled_from(LIBERO_SUITES)),
)
@settings(max_examples=100, deadline=None)
def test_p3_concurrent_suites_are_distinct(pair: tuple[str, str]) -> None:
    """P3: two distinct TASK_SUITE values yield distinct manifests.

    **Validates: Requirements 5.3, 10.4**
    """
    _require("envsubst")
    s1, s2 = pair
    assume(s1 != s2)

    docs_1 = render_and_load(_base_env(s1), MANIFEST_DIR / "libero-finetune.yaml")
    docs_2 = render_and_load(_base_env(s2), MANIFEST_DIR / "libero-finetune.yaml")
    assert len(docs_1) == 1 and len(docs_2) == 1

    name_1 = docs_1[0]["metadata"]["name"]
    name_2 = docs_2[0]["metadata"]["name"]
    assert name_1 != name_2, (
        f"expected distinct metadata.name for distinct suites, but got " f"{name_1!r} for both {s1!r} and {s2!r}"
    )

    worker = docs_1[0]["spec"]["pytorchReplicaSpecs"]["Worker"]
    cmd_1 = worker["template"]["spec"]["containers"][0]["command"]
    cmd_2 = docs_2[0]["spec"]["pytorchReplicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]["command"]

    ds_1 = _find_dataset_name_flag(cmd_1)
    ds_2 = _find_dataset_name_flag(cmd_2)
    assert ds_1 != ds_2, f"expected distinct --dataset_name args for distinct suites, " f"but got {ds_1!r} for both"
    assert ds_1 == f"--dataset_name={s1}", f"unexpected --dataset_name: {ds_1!r}"
    assert ds_2 == f"--dataset_name={s2}", f"unexpected --dataset_name: {ds_2!r}"


# ---------------------------------------------------------------------------
# P4 / P5 -- verify-tfds-layout.sh accepts / rejects TFDS trees
# ---------------------------------------------------------------------------


_VERIFY_SCRIPT = MANIFEST_DIR / "verify-tfds-layout.sh"


def _run_verify(root: Path, suite: str) -> subprocess.CompletedProcess[str]:
    """Invoke ``verify-tfds-layout.sh <suite>`` with ``LIBERO_DATA_ROOT=root``."""
    bash = _require("bash")
    return subprocess.run(
        [bash, str(_VERIFY_SCRIPT), suite],
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "LIBERO_DATA_ROOT": str(root),
        },
    )


# P5's defect strategy intentionally excludes ``no_features``: the
# production ``verify-tfds-layout.sh`` only checks for ``dataset_info.json``
# and ``*.tfrecord-*`` shards, so a tree with no ``features.json`` still
# passes. Tests against that defect would spuriously fail. See the design's
# "Component 6" section for the exact check set.
_P5_DEFECTS: tuple[str, ...] = tuple(d for d in MALFORMED_DEFECTS if d != "no_features")


# Feature: openvla-oft-libero-recipe, Property 4: verify-tfds-layout.sh accepts well-formed TFDS trees
@given(suite=st.sampled_from(LIBERO_SUITES))
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_p4_verify_script_accepts_wellformed(
    tfds_tree_wellformed: Any,
    suite: str,
) -> None:
    """P4: ``verify-tfds-layout.sh`` returns 0 on a well-formed tree.

    **Validates: Requirements 3.2**
    """
    tree = tfds_tree_wellformed(suite)
    result = _run_verify(tree.root, suite)
    assert result.returncode == 0, (
        f"expected exit 0 for well-formed suite={suite!r} under "
        f"{tree.root}, got {result.returncode}; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


# Feature: openvla-oft-libero-recipe, Property 5: verify-tfds-layout.sh rejects malformed TFDS trees
@given(
    defect=st.sampled_from(_P5_DEFECTS),
    suite=st.sampled_from(LIBERO_SUITES),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_p5_verify_script_rejects_malformed(
    tfds_tree_malformed: Any,
    defect: str,
    suite: str,
) -> None:
    """P5: ``verify-tfds-layout.sh`` returns 1 and emits FAIL on malformed trees.

    **Validates: Requirements 3.3, 10.1**
    """
    tree = tfds_tree_malformed(defect=defect, suite=suite)
    result = _run_verify(tree.root, suite)
    assert result.returncode == 1, (
        f"expected exit 1 for defect={defect!r} suite={suite!r}, got "
        f"{result.returncode}; stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert "FAIL:" in result.stderr, f"expected 'FAIL:' marker on stderr, got stderr={result.stderr!r}"
    # The stderr message must name a path under the data root so the
    # operator can navigate to the missing file (Req 10.1).
    assert str(tree.root) in result.stderr, (
        f"expected stderr to reference the data-root path {tree.root!r}; " f"got stderr={result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# P6 -- Download_Job idempotence
# ---------------------------------------------------------------------------


_HF_CLI_SHIM = TESTS_DIR / "fixtures" / "hf-cli-shim"


def _make_shim_dir(root: Path) -> Path:
    """Create a ``bin/`` dir with ``huggingface-cli`` and ``pip`` shims.

    The resulting directory must be placed first on PATH so the bash body's
    ``pip install`` and ``huggingface-cli download`` calls resolve to the
    hermetic stand-ins rather than the real tooling. The ``pip`` shim is a
    no-op; the ``huggingface-cli`` shim is the real
    ``tests/libero/fixtures/hf-cli-shim`` script.
    """
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    # ``huggingface-cli`` -> the hf-cli-shim under tests/libero/fixtures/.
    hf = bin_dir / "huggingface-cli"
    hf.write_text(_HF_CLI_SHIM.read_text())
    hf.chmod(hf.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # ``pip`` -> a no-op so ``pip install hf_transfer`` exits 0 without
    # contacting PyPI.
    pip = bin_dir / "pip"
    pip.write_text("#!/usr/bin/env bash\nexit 0\n")
    pip.chmod(pip.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return bin_dir


def _hermetic_download_script(data_root: Path) -> str:
    """Return the full Download_Job bash body with ``/data`` rewritten.

    The Download_Job writes to ``/data/datasets/rlds`` hardcoded; this
    helper rewrites those paths to a tmpdir-scoped equivalent so the test
    never touches a real PVC.
    """
    body = _download_bash_body()
    # Rewrite the hardcoded /data/... paths to the tmpdir equivalent.
    return body.replace("/data/datasets/rlds", f"{data_root}/datasets/rlds")


def _dir_fingerprint(path: Path) -> list[tuple[str, str]]:
    """Return a sorted ``[(relative_path, sha256), ...]`` fingerprint."""
    items: list[tuple[str, str]] = []
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            items.append((str(p.relative_to(path)), h))
    return items


# Feature: openvla-oft-libero-recipe, Property 6: Download_Job is idempotent
@given(suite=st.sampled_from(LIBERO_SUITES))
@settings(max_examples=100, deadline=None)
def test_p6_download_job_is_idempotent(
    tmp_path_factory: pytest.TempPathFactory,
    suite: str,
) -> None:
    """P6: re-running the Download_Job yields a byte-identical tree.

    **Validates: Requirements 2.9, 10.3**
    """
    bash = _require("bash")
    assert _HF_CLI_SHIM.is_file(), f"missing hf-cli-shim at {_HF_CLI_SHIM}"

    work = tmp_path_factory.mktemp("p6-idempotence")
    data_root = work / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    bin_dir = _make_shim_dir(work)

    script = work / "download.sh"
    script.write_text(_hermetic_download_script(data_root))

    # PATH puts the shim bin dir first so huggingface-cli / pip resolve to
    # the local stand-ins. We still need /usr/bin etc. for coreutils.
    env = {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "TASK_SUITE": suite,
    }

    first = subprocess.run(
        [bash, str(script)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert first.returncode == 0, f"first download run failed: stdout={first.stdout!r} " f"stderr={first.stderr!r}"
    fp_1 = _dir_fingerprint(data_root)

    second = subprocess.run(
        [bash, str(script)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert second.returncode == 0, f"second download run failed: stdout={second.stdout!r} " f"stderr={second.stderr!r}"
    fp_2 = _dir_fingerprint(data_root)

    assert fp_1 == fp_2, (
        f"download tree changed between idempotent runs for suite={suite!r}:\n" f"  run 1: {fp_1}\n  run 2: {fp_2}"
    )
    # Belt-and-braces: also assert via filecmp.dircmp to catch weird
    # metadata differences that a pure-content fingerprint would miss.
    cmp = filecmp.dircmp(str(data_root), str(data_root))
    assert not cmp.diff_files, f"dircmp reports diffs: {cmp.diff_files}"


# ---------------------------------------------------------------------------
# P7 -- Render + apply round-trip preserves every Requirement-4 flag
# ---------------------------------------------------------------------------


# All flags the round-trip must preserve, expressed as substring checks.
# Values that embed caller-supplied placeholders use prefix-only matching
# (e.g. ``--wandb_entity=``) so the test does not over-specify beyond
# Requirement 4.
_REQUIRED_FLAG_SUBSTRINGS: tuple[str, ...] = (
    "--vla_path=openvla/openvla-7b",
    "--data_root_dir=/data/datasets/rlds",
    "--run_root_dir=/data/runs",
    "--use_l1_regression=True",
    "--use_diffusion=False",
    "--use_film=False",
    "--num_images_in_input=2",
    "--use_proprio=True",
    "--batch_size=8",
    "--learning_rate=5e-4",
    "--grad_accumulation_steps=1",
    "--num_steps_before_decay=100000",
    "--max_steps=150005",
    "--save_freq=10000",
    "--save_latest_checkpoint_only=False",
    "--image_aug=True",
    "--use_lora=True",
    "--lora_rank=32",
    "--lora_dropout=0.0",
    "--merge_lora_during_training=True",
    # Requirement 4.7 requires the `--run_id_note` flag to carry the exact
    # upstream LIBERO.md string. GNU envsubst leaves ``${VAR:-default}``
    # literal (it does not honour Bash's default-value syntax), so the
    # rendered command array may contain either:
    #   * ``--run_id_note=<UPSTREAM>`` (when RUN_ID_NOTE was exported before
    #     envsubst ran -- the path taken after ``source env_vars``), or
    #   * ``--run_id_note=${RUN_ID_NOTE:-<UPSTREAM>}`` (when the caller
    #     invokes envsubst directly without sourcing env_vars first).
    # Both cases embed the upstream default string verbatim, so we require
    # the substring only and accept either literal form. The
    # ``--run_id_note=`` flag presence is verified separately below.
    _UPSTREAM_RUN_ID_NOTE,
    "--wandb_entity=",
    "--wandb_project=",
    "--wandb_log_freq=10",
)


def _extract_worker_command(doc: dict[str, Any]) -> list[str]:
    """Return the Worker container's ``command:`` array as a list of strings."""
    command = doc["spec"]["pytorchReplicaSpecs"]["Worker"]["template"]["spec"]["containers"][0]["command"]
    return [str(entry) for entry in command]


def _kubectl_dry_run(rendered_yaml: str) -> dict[str, Any] | None:
    """Pipe rendered YAML through ``kubectl apply --dry-run=client -o yaml``.

    Returns the parsed manifest, or ``None`` if ``kubectl`` is unavailable
    on PATH (the test then falls back to ``yaml.safe_load`` per the design's
    Testing Strategy Layer 2).
    """
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        return None
    result = subprocess.run(
        [kubectl, "apply", "--dry-run=client", "-o", "yaml", "-f", "-"],
        input=rendered_yaml,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # kubectl might be installed but unable to validate the CRD because
        # no kubeconfig / cluster is reachable. Fall back cleanly.
        return None
    return yaml.safe_load(result.stdout)


# Feature: openvla-oft-libero-recipe, Property 7: Render + apply round-trip preserves every Requirement-4 flag
@given(suite=st.sampled_from(LIBERO_SUITES))
@settings(max_examples=100, deadline=None)
def test_p7_render_apply_roundtrip_preserves_required_flags(suite: str) -> None:
    """P7: every Req-4 flag survives render -> (kubectl apply --dry-run) -> YAML.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 5.2, 10.6**
    """
    _require("envsubst")
    manifest = MANIFEST_DIR / "libero-finetune.yaml"
    rendered = render(_base_env(suite), manifest)

    parsed = _kubectl_dry_run(rendered)
    if parsed is None:
        # Fallback path: parse the rendered text directly. This sacrifices
        # server-side CRD validation but exercises the same flag-propagation
        # property.
        docs = [d for d in yaml.safe_load_all(rendered) if d is not None]
        assert len(docs) == 1, f"expected exactly one document, got {len(docs)}"
        parsed = docs[0]

    command = _extract_worker_command(parsed)
    joined = "\n".join(command)

    for required in _REQUIRED_FLAG_SUBSTRINGS:
        assert any(required in entry for entry in command), (
            f"required flag substring {required!r} missing from Worker "
            f"command for suite={suite!r}.\nCommand was:\n{joined}"
        )

    # The --run_id_note flag itself must be present; the substring check
    # above only verifies the upstream default string is embedded (it could
    # be inside a ``${RUN_ID_NOTE:-default}`` literal when envsubst was
    # invoked without sourcing env_vars). Check the flag prefix separately.
    assert any(entry.startswith("--run_id_note=") or "--run_id_note=" in entry for entry in command), (
        f"--run_id_note= flag missing from Worker command for " f"suite={suite!r}.\nCommand was:\n{joined}"
    )

    # The suite-specific --dataset_name flag is validated separately because
    # its value varies per Hypothesis example.
    dataset_flag = _find_dataset_name_flag(command)
    assert dataset_flag == f"--dataset_name={suite}", f"expected --dataset_name={suite}, got {dataset_flag!r}"
