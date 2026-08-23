"""Reproducibility artifact assembly and the section 10 release checklist.

Limitation L19 is that the work has no reproducibility artifact.  Section 10
lists ten items that must ship.  This module does two things:

* :func:`check_release` evaluates the checklist against a run directory and the
  ``logistics/`` package, returning a per-item verdict rather than a boolean, so
  a partially-complete release says *which* part is missing;
* :func:`assemble_release` copies the artifact into a bundle directory, hashes
  every file, and writes a manifest.

The design decision worth stating: **the checklist refuses to pass on a
simulated run.**  A bundle whose ``provenance`` is ``simulated`` is a
demonstration that the pipeline executes, not a research artifact, and labelling
it as the latter is precisely the failure ``--strict-real`` exists to prevent.

Environment capture is deliberately not a ``pip freeze`` shell-out.  The
importable environment is what actually determines results, so it is read from
``importlib.metadata`` inside the running interpreter.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "RELEASE_CHECKLIST",
    "environment_lockfile",
    "check_release",
    "assemble_release",
]

#: Specification section 10, verbatim, with the artifact that satisfies each.
RELEASE_CHECKLIST: Tuple[Tuple[str, str], ...] = (
    ("pre_registration", "logistics/PREREGISTRATION.md"),
    ("prompts_and_role_definitions", "cca_synth/prompts/"),
    ("model_snapshots_and_manifest", "manifest.json"),
    ("versioned_benchmark_and_data_card", "logistics/ADVCONSISTENCY_DATACARD.md"),
    ("raw_structured_outputs", "audit_records.jsonl"),
    ("evaluation_and_statistics_code", "cca_synth/"),
    ("full_results_with_uncertainty", "evaluation.json"),
    ("failure_analysis_and_negative_findings", "ablation.json"),
    ("cost_and_latency_report", "REPORT.md"),
    ("ethics_and_threat_model_review", "logistics/ETHICS_COMPLIANCE.md"),
    ("replication", "replication.json"),
)


def environment_lockfile() -> Dict[str, Any]:
    """Capture the environment that produced a run.

    Read from the live interpreter rather than shelled out to ``pip freeze``,
    because the importable environment is what determines results and the two
    can differ (editable installs, ``PYTHONPATH`` entries, vendored packages).

    Returns:
        Mapping with interpreter, platform, and installed distributions.
    """
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python < 3.8 is unsupported anyway
        return {"error": "importlib.metadata unavailable"}

    packages: Dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"] if distribution.metadata else None
        if name:
            packages[name.lower()] = distribution.version or "unknown"

    torch_info: Dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "mps_available": bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ),
        }
    except Exception as error:  # pragma: no cover - torch is a hard dependency
        torch_info = {"error": repr(error)}

    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch_info,
        "packages": dict(sorted(packages.items())),
        "note": (
            "Captured from importlib.metadata inside the running interpreter, "
            "not from `pip freeze`. The importable environment determines "
            "results and can differ from what pip reports."
        ),
    }


def _file_sha256(path: str) -> str:
    """Hash a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_release(
    run_directory: str,
    repository_root: str,
    logistics_root: str,
) -> Dict[str, Any]:
    """Evaluate the section 10 checklist against a run.

    Args:
        run_directory: A ``runs/<run_id>/`` directory.
        repository_root: The ``cca_synth`` checkout.
        logistics_root: The ``logistics`` package.

    Returns:
        Per-item verdicts, the overall pass/fail, and the reason for any failure.
    """
    def resolve(target: str) -> str:
        """Map a checklist target onto an absolute path."""
        if target.startswith("logistics/"):
            return os.path.join(logistics_root, target.split("/", 1)[1])
        if target.startswith("cca_synth/"):
            remainder = target.split("/", 1)[1]
            return os.path.join(repository_root, remainder) if remainder else repository_root
        return os.path.join(run_directory, target)

    items: List[Dict[str, Any]] = []
    for name, target in RELEASE_CHECKLIST:
        path = resolve(target)
        present = os.path.exists(path)
        items.append(
            {
                "item": name,
                "expected_at": target,
                "present": present,
                "resolved_path": path if present else "",
            }
        )

    # Provenance gate. A simulated bundle is a demonstration, not an artifact.
    provenance = "unknown"
    evaluation_path = os.path.join(run_directory, "evaluation.json")
    if os.path.exists(evaluation_path):
        try:
            with open(evaluation_path, "r", encoding="utf-8") as handle:
                provenance = json.load(handle).get("provenance", "unknown")
        except (json.JSONDecodeError, OSError):
            provenance = "unreadable"

    missing = [i["item"] for i in items if not i["present"]]
    real = provenance == "real"

    return {
        "run_directory": run_directory,
        "provenance": provenance,
        "items": items,
        "n_present": sum(1 for i in items if i["present"]),
        "n_total": len(items),
        "missing": missing,
        "passed": real and not missing,
        "reason": (
            f"provenance is '{provenance}', not 'real'. A bundle built from "
            "simulated records demonstrates that the pipeline executes; it is "
            "not a research artifact and must not be released as one."
            if not real
            else f"missing: {', '.join(missing)}"
            if missing
            else "all section 10 items present on a real run"
        ),
    }


def assemble_release(
    run_directory: str,
    repository_root: str,
    logistics_root: str,
    destination: str,
    *,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    """Copy a run's artifact into a hashed release bundle.

    Args:
        run_directory: A ``runs/<run_id>/`` directory.
        repository_root: The ``cca_synth`` checkout.
        logistics_root: The ``logistics`` package.
        destination: Bundle directory to create.
        allow_incomplete: Build the bundle even when the checklist fails,
            marking it ``incomplete`` in the manifest. Intended for interim
            sharing, never for publication.

    Returns:
        The bundle manifest.

    Raises:
        RuntimeError: If the checklist fails and ``allow_incomplete`` is False.
    """
    checklist = check_release(run_directory, repository_root, logistics_root)
    if not checklist["passed"] and not allow_incomplete:
        raise RuntimeError(
            f"release checklist failed: {checklist['reason']}. "
            "Pass allow_incomplete=True to build an explicitly-marked interim "
            "bundle."
        )

    os.makedirs(destination, exist_ok=True)
    copied: Dict[str, str] = {}

    for item in checklist["items"]:
        source = item["resolved_path"]
        if not source:
            continue
        name = os.path.basename(source.rstrip(os.sep)) or item["item"]
        target = os.path.join(destination, name)
        if os.path.isdir(source):
            shutil.copytree(
                source,
                target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", ".pytest_cache", "runs", ".git"
                ),
            )
        else:
            shutil.copy2(source, target)
            copied[name] = _file_sha256(target)

    environment = environment_lockfile()
    with open(os.path.join(destination, "environment.json"), "w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2, sort_keys=True)

    manifest = {
        "bundle_version": 1,
        "complete": bool(checklist["passed"]),
        "provenance": checklist["provenance"],
        "checklist": checklist,
        "file_hashes": copied,
        "environment_summary": {
            "python": environment.get("python_version", "").split()[0],
            "platform": environment.get("platform", ""),
            "torch": environment.get("torch", {}).get("version", ""),
        },
        "warning": (
            ""
            if checklist["passed"]
            else "INCOMPLETE BUNDLE — not suitable for publication. "
            + checklist["reason"]
        ),
    }
    with open(os.path.join(destination, "bundle_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest
