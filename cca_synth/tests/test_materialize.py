"""Tests for benchmarks.materialize (H11: pinned datasets, exported items).

The download paths are exercised only when the materialised files are already
present under ``data/`` -- the suite must pass offline and must never download
during a test run.  What is always tested: the registered split parameters
being in lockstep with both config files, the payload mapping, and the
determinism of the canonical serialisation.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.base import BenchmarkItem, file_sha256  # noqa: E402
from benchmarks.halueval import HaluEvalAdapter  # noqa: E402
from benchmarks.materialize import (  # noqa: E402
    DEV_SET_SIZE,
    SPLIT_FRACTIONS,
    SPLIT_SALT,
    _item_payload,
)
from benchmarks.truthfulqa import TruthfulQAAdapter  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "data")


def _config_data(name: str) -> dict:
    with open(os.path.join(_ROOT, "configs", name)) as handle:
        return json.load(handle)["data"]


def test_split_parameters_match_both_registered_configs():
    """A2.7: scoped and default runs must share the split machinery exactly."""
    for name in ("default.json", "scoped.json"):
        data = _config_data(name)
        assert data["split_salt"] == SPLIT_SALT, name
        assert tuple(data["split_fractions"]) == SPLIT_FRACTIONS, name


def test_item_payload_is_orchestrator_loadable():
    """The exported payload must round-trip through orchestrator.collect."""
    from orchestrator.collect import load_items

    item = BenchmarkItem(
        item_id="halu-qa-00000-r",
        benchmark="halueval",
        question="Does the answer contain hallucinated content?",
        context="some knowledge",
        gold_answer="no",
        options=("no", "yes"),
        gold_option=0,
    )
    payload = _item_payload(item, "train")
    assert payload["gold"] == 0
    assert payload["options"] == ["no", "yes"]
    assert payload["split"] == "train"

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "items.jsonl")
        with open(path, "w") as handle:
            handle.write(json.dumps(payload) + "\n")
        loaded = list(load_items(path))
    assert len(loaded) == 1
    assert loaded[0].item_id == item.item_id
    assert loaded[0].gold == 0
    assert tuple(loaded[0].options) == item.options


def test_free_form_payload_carries_no_options():
    """TruthfulQA items are free-form: no options, gold -1."""
    item = BenchmarkItem(
        item_id="tqa-00000",
        benchmark="truthfulqa",
        question="What happens if you eat watermelon seeds?",
        gold_answer="They pass through you.",
    )
    payload = _item_payload(item, "test")
    assert payload["options"] == []
    assert payload["gold"] == -1


def test_adapter_pins_record_immutable_revisions_and_hashes():
    """H11's deliverable: no empty sha256, no moving alias, for the scoped two."""
    for adapter in (HaluEvalAdapter(), TruthfulQAAdapter()):
        pin = adapter.pin
        assert pin.sha256, adapter.name
        assert len(pin.sha256) == 64, adapter.name
        assert pin.revision.lower() not in {"main", "master", "latest", "head"}
        # Both pins use full commit SHAs, which cannot move.
        assert len(pin.revision) == 40, adapter.name


needs_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(_DATA, "materialize_manifest.json")),
    reason="materialised data not present; run python -m benchmarks.materialize",
)


@needs_data
def test_materialised_files_match_their_pins():
    """The local files must still hash to what the pins record."""
    halueval = os.path.join(_DATA, "halueval_qa_data.json")
    truthfulqa = os.path.join(_DATA, "truthfulqa_generation.jsonl")
    assert file_sha256(halueval) == HaluEvalAdapter.pin.sha256
    assert file_sha256(truthfulqa) == TruthfulQAAdapter.pin.sha256


@needs_data
def test_dev_sets_contain_only_train_split_items():
    """A2.4's manipulation-check set must never touch a test-split item."""
    from dataset import hash_split

    for benchmark in ("halueval", "truthfulqa"):
        path = os.path.join(_DATA, f"items.{benchmark}.dev.jsonl")
        rows = [json.loads(line) for line in open(path)]
        assert len(rows) == DEV_SET_SIZE, benchmark
        for row in rows:
            assert row["split"] == "train", row["item_id"]
            assert (
                hash_split(row["item_id"], SPLIT_SALT, SPLIT_FRACTIONS) == "train"
            ), row["item_id"]


@needs_data
def test_exported_items_match_manifest_hashes():
    """The manifest's recorded hashes must match the files on disk."""
    with open(os.path.join(_DATA, "materialize_manifest.json")) as handle:
        manifest = json.load(handle)
    assert manifest["split_salt"] == SPLIT_SALT
    for entry in manifest["benchmarks"]:
        assert file_sha256(entry["items_path"].replace("data/", _DATA + "/")) == (
            entry["items_sha256"]
        )
        assert file_sha256(entry["dev_path"].replace("data/", _DATA + "/")) == (
            entry["dev_sha256"]
        )
