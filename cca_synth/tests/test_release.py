"""Tests for the reproducibility artifact (P19, limitation L19).

The property that matters: the release checklist must **refuse a simulated
run**. A bundle built from synthetic records demonstrates that the pipeline
executes; calling it a research artifact is the failure `--strict-real` exists
to prevent, and the bundle assembler is the last place that mistake can be
caught before publication.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from release import (  # noqa: E402
    RELEASE_CHECKLIST,
    assemble_release,
    check_release,
    environment_lockfile,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGISTICS = os.path.join(os.path.dirname(_REPO), "logistics")


def _fake_run(directory: str, provenance: str) -> str:
    """Write a minimal run directory with the given provenance."""
    run = os.path.join(directory, "run")
    os.makedirs(run, exist_ok=True)
    for name in ("manifest.json", "evaluation.json", "ablation.json"):
        with open(os.path.join(run, name), "w", encoding="utf-8") as handle:
            json.dump({"provenance": provenance}, handle)
    for name in ("audit_records.jsonl", "REPORT.md", "replication.json"):
        with open(os.path.join(run, name), "w", encoding="utf-8") as handle:
            handle.write("{}\n")
    return run


def test_environment_lockfile_captures_the_interpreter():
    """The lockfile records what actually ran, not what pip reports."""
    lock = environment_lockfile()
    assert lock["python_version"]
    assert lock["platform"]
    assert "torch" in lock["packages"] or "version" in lock.get("torch", {})
    # Read from the live interpreter, which is the point.
    assert "importlib.metadata" in lock["note"]


def test_checklist_refuses_a_simulated_run():
    """A simulated bundle is a demonstration, not a research artifact."""
    with tempfile.TemporaryDirectory() as directory:
        run = _fake_run(directory, "simulated")
        result = check_release(run, _REPO, _LOGISTICS)
        assert result["passed"] is False
        assert "simulated" in result["reason"]


def test_checklist_names_what_is_missing():
    """A partial release says which item is absent, not merely 'incomplete'."""
    with tempfile.TemporaryDirectory() as directory:
        run = _fake_run(directory, "real")
        os.unlink(os.path.join(run, "REPORT.md"))
        result = check_release(run, _REPO, _LOGISTICS)
        assert result["passed"] is False
        assert "cost_and_latency_report" in result["missing"]


def test_checklist_passes_on_a_complete_real_run():
    """The happy path must actually be reachable."""
    with tempfile.TemporaryDirectory() as directory:
        run = _fake_run(directory, "real")
        result = check_release(run, _REPO, _LOGISTICS)
        assert result["passed"] is True, result["reason"]
        assert result["n_present"] == result["n_total"] == len(RELEASE_CHECKLIST)


def test_assemble_refuses_an_incomplete_bundle_by_default():
    """Publication must not be one accidental call away."""
    with tempfile.TemporaryDirectory() as directory:
        run = _fake_run(directory, "simulated")
        with pytest.raises(RuntimeError, match="release checklist failed"):
            assemble_release(
                run, _REPO, _LOGISTICS, os.path.join(directory, "bundle")
            )


def test_assemble_marks_an_interim_bundle_as_incomplete():
    """An override is available, but the bundle says so in its manifest."""
    with tempfile.TemporaryDirectory() as directory:
        run = _fake_run(directory, "simulated")
        destination = os.path.join(directory, "bundle")
        manifest = assemble_release(
            run, _REPO, _LOGISTICS, destination, allow_incomplete=True
        )
        assert manifest["complete"] is False
        assert "INCOMPLETE BUNDLE" in manifest["warning"]
        assert os.path.exists(os.path.join(destination, "environment.json"))
        assert os.path.exists(os.path.join(destination, "bundle_manifest.json"))
        # Every copied file is hashed, so the bundle is checkable.
        assert manifest["file_hashes"]
        assert all(len(h) == 64 for h in manifest["file_hashes"].values())
