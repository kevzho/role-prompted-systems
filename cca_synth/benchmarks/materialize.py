"""Materialise the scoped-study benchmarks and export orchestrator items.

H11 in ``logistics/HANDOVER.md``: every :class:`~benchmarks.base.DatasetPin`
shipped with an empty ``sha256``, and ``verify_pin`` refuses to evaluate until
the hash of the materialised local file is recorded.  This module closes that
for the two benchmarks the scoped study (``PREREGISTRATION.md`` §11 A2) runs:

* **HaluEval QA** -- the upstream ``data/qa_data.json`` is already JSON lines,
  so the materialised file is the upstream bytes verbatim, fetched at a pinned
  commit.  The recorded hash therefore identifies the upstream content itself.
* **TruthfulQA generation** -- upstream ships parquet, which the adapters do not
  read.  The parquet is fetched at a pinned repo commit, its own hash checked,
  then converted row-for-row to canonical JSONL (sorted keys, no whitespace,
  ``ensure_ascii=False``).  The conversion is deterministic, so pinned parquet
  plus this module reproduce the recorded JSONL hash exactly.

Downloads happen at pinned immutable revisions only; there is no ``main``
anywhere in this file.  Both adapters' pins record the hashes produced here,
so a later ``load(verify=True)`` fails loudly if the local copy drifts.

Item export writes the JSONL that ``orchestrator.collect.load_items`` reads,
via the adapters' own ``load``, with each item's split assigned by
``dataset.hash_split`` under the registered salt and fractions.  The dev file
contains the first 200 train-split items in corpus order -- the A2.4
manipulation-check set -- so no test-split item is ever read during it.

Usage::

    python -m benchmarks.materialize --data-dir data
    python -m benchmarks.materialize --data-dir data --skip-download  # verify only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import get_adapter  # noqa: E402
from benchmarks.base import BenchmarkItem, file_sha256  # noqa: E402
from utils import get_logger  # noqa: E402

__all__ = [
    "HALUEVAL_COMMIT",
    "TRUTHFULQA_COMMIT",
    "materialize_halueval",
    "materialize_truthfulqa",
    "export_items",
    "main",
]

#: Last upstream commit touching ``data/qa_data.json`` (2023-05-26).
HALUEVAL_COMMIT = "207f479c555a68b8d2431631a95083b84b2accbe"
HALUEVAL_URL = (
    "https://raw.githubusercontent.com/RUCAIBox/HaluEval/"
    f"{HALUEVAL_COMMIT}/data/qa_data.json"
)
#: Hash of the upstream file verbatim; also recorded in ``HaluEvalAdapter.pin``.
HALUEVAL_SHA256 = "89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88"

#: Hugging Face repo commit for ``truthfulqa/truthful_qa`` (2024-01-04).
TRUTHFULQA_COMMIT = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"
TRUTHFULQA_URL = (
    "https://huggingface.co/datasets/truthfulqa/truthful_qa/resolve/"
    f"{TRUTHFULQA_COMMIT}/generation/validation-00000-of-00001.parquet"
)
#: Hash of the parquet as downloaded, checked before conversion.
TRUTHFULQA_PARQUET_SHA256 = (
    "dfb1004b8ab83b22e8e476c76d5ac6074ff35c43946a724e810de3d83c3e21a5"
)

#: Registered split parameters (``PREREGISTRATION.md`` §9; identical in
#: ``configs/default.json`` and ``configs/scoped.json`` per A2.7).
SPLIT_SALT = "cca-synth-v1"
SPLIT_FRACTIONS = (0.5, 0.15, 0.15, 0.2)
#: A2.4: the manipulation-check dev set size.
DEV_SET_SIZE = 200


def _download(url: str, path: str, expected_sha256: str, logger: Any) -> None:
    """Fetch ``url`` to ``path`` and require the expected hash.

    Args:
        url: Pinned-revision URL.
        path: Destination.
        expected_sha256: Required content hash.

    Raises:
        ValueError: When the downloaded bytes do not match.
    """
    logger.info("downloading %s", url)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 -- https, pinned revision
    got = file_sha256(tmp)
    if got != expected_sha256:
        os.unlink(tmp)
        raise ValueError(
            f"download of {url} hashed {got}, expected {expected_sha256}; "
            "refusing to materialise unverified bytes"
        )
    os.replace(tmp, path)


def materialize_halueval(data_dir: str, *, download: bool = True) -> str:
    """Materialise HaluEval QA as the upstream JSONL verbatim.

    Args:
        data_dir: Destination directory.
        download: Fetch when the file is absent.

    Returns:
        Path to the materialised file.
    """
    logger = get_logger("materialize")
    path = os.path.join(data_dir, "halueval_qa_data.json")
    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(path)
        _download(HALUEVAL_URL, path, HALUEVAL_SHA256, logger)
    got = file_sha256(path)
    if got != HALUEVAL_SHA256:
        raise ValueError(
            f"{path} hashed {got}, expected {HALUEVAL_SHA256}; delete and re-run"
        )
    logger.info("halueval materialised: %s sha256=%s", path, got)
    return path


def materialize_truthfulqa(data_dir: str, *, download: bool = True) -> str:
    """Materialise TruthfulQA generation as canonical JSONL.

    The parquet is verified against its own pinned hash before conversion, and
    the conversion is deterministic: rows in upstream order, keys sorted,
    compact separators, ``ensure_ascii=False``, one ``\\n``-terminated object
    per line.

    Args:
        data_dir: Destination directory.
        download: Fetch the parquet when absent.

    Returns:
        Path to the materialised JSONL.
    """
    logger = get_logger("materialize")
    parquet = os.path.join(data_dir, "truthfulqa_generation.parquet")
    path = os.path.join(data_dir, "truthfulqa_generation.jsonl")
    if not os.path.exists(parquet):
        if not download:
            raise FileNotFoundError(parquet)
        _download(TRUTHFULQA_URL, parquet, TRUTHFULQA_PARQUET_SHA256, logger)
    got = file_sha256(parquet)
    if got != TRUTHFULQA_PARQUET_SHA256:
        raise ValueError(
            f"{parquet} hashed {got}, expected {TRUTHFULQA_PARQUET_SHA256}; "
            "delete and re-run"
        )

    import pandas as pd  # Local import: only this converter needs it.

    frame = pd.read_parquet(parquet)
    expected_columns = {
        "type", "category", "question", "best_answer",
        "correct_answers", "incorrect_answers", "source",
    }
    missing = expected_columns - set(frame.columns)
    if missing:
        raise ValueError(f"parquet is missing columns {sorted(missing)}")

    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in frame.itertuples(index=False):
            record = {
                "type": str(row.type),
                "category": str(row.category),
                "question": str(row.question),
                "best_answer": str(row.best_answer),
                "correct_answers": [str(a) for a in row.correct_answers],
                "incorrect_answers": [str(a) for a in row.incorrect_answers],
                "source": str(row.source),
            }
            handle.write(
                json.dumps(record, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
                + "\n"
            )
    os.replace(tmp, path)
    logger.info(
        "truthfulqa materialised: %s rows=%d sha256=%s",
        path, len(frame), file_sha256(path),
    )
    return path


def _item_payload(item: BenchmarkItem, split: str) -> Dict[str, Any]:
    """Orchestrator-format payload for one item.

    Args:
        item: Adapter-loaded item.
        split: Split name from ``dataset.hash_split``.

    Returns:
        A mapping matching ``orchestrator.collect.load_items``; the extra
        ``split`` key is carried for filtering and ignored by the loader.
    """
    return {
        "item_id": item.item_id,
        "benchmark": item.benchmark,
        "question": item.question,
        "options": list(item.options),
        "gold": item.gold_option,
        "context": item.context,
        "split": split,
    }


def export_items(
    benchmark: str,
    raw_path: str,
    data_dir: str,
    *,
    adapter_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Export adapter-loaded items as orchestrator JSONL, split-annotated.

    Writes ``items.<benchmark>.jsonl`` (every item, with a ``split`` field) and
    ``items.<benchmark>.dev.jsonl`` (the first :data:`DEV_SET_SIZE` train-split
    items in corpus order -- the A2.4 manipulation-check set).  Test-split items
    are written only into the full file, never selected here.

    Args:
        benchmark: Adapter name.
        raw_path: Materialised dataset file, verified against the adapter pin.
        data_dir: Destination directory.
        adapter_kwargs: Adapter constructor options.

    Returns:
        Summary with per-split counts and output paths.
    """
    from dataset import hash_split  # noqa: E402  Local import avoids a cycle.

    logger = get_logger("materialize")
    adapter = get_adapter(benchmark, **(adapter_kwargs or {}))
    items = adapter.load(raw_path, verify=True)

    all_path = os.path.join(data_dir, f"items.{benchmark}.jsonl")
    dev_path = os.path.join(data_dir, f"items.{benchmark}.dev.jsonl")
    counts: Dict[str, int] = {}
    n_dev = 0
    with open(all_path, "w", encoding="utf-8") as full, \
            open(dev_path, "w", encoding="utf-8") as dev:
        for item in items:
            split = hash_split(item.item_id, SPLIT_SALT, SPLIT_FRACTIONS)
            counts[split] = counts.get(split, 0) + 1
            line = json.dumps(
                _item_payload(item, split), sort_keys=True, ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n"
            full.write(line)
            if split == "train" and n_dev < DEV_SET_SIZE:
                dev.write(line)
                n_dev += 1

    summary = {
        "benchmark": benchmark,
        "n_items": len(items),
        "splits": counts,
        "n_dev": n_dev,
        "items_path": all_path,
        "dev_path": dev_path,
        "items_sha256": file_sha256(all_path),
        "dev_sha256": file_sha256(dev_path),
        "pin": adapter.pin.as_dict(),
    }
    logger.info(
        "%s: %d items, splits=%s, dev=%d", benchmark, len(items), counts, n_dev
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Materialise, verify, and export both scoped-study benchmarks.

    Args:
        argv: CLI arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Destination directory.")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Verify and export from already-present files only.",
    )
    args = parser.parse_args(argv)
    os.makedirs(args.data_dir, exist_ok=True)
    download = not args.skip_download

    summaries: List[Dict[str, Any]] = []
    halueval_raw = materialize_halueval(args.data_dir, download=download)
    summaries.append(export_items("halueval", halueval_raw, args.data_dir))
    truthfulqa_raw = materialize_truthfulqa(args.data_dir, download=download)
    summaries.append(export_items("truthfulqa", truthfulqa_raw, args.data_dir))

    manifest_path = os.path.join(args.data_dir, "materialize_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "split_salt": SPLIT_SALT,
                "split_fractions": list(SPLIT_FRACTIONS),
                "dev_set_size": DEV_SET_SIZE,
                "benchmarks": summaries,
            },
            handle, indent=2, sort_keys=True,
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
