"""Fresh / temporal QA adapter.

Primary outcome per specification section 5: accuracy against *timestamped
authoritative sources*. Harm outcomes: unsupported-answer rate and retrieval
dependence.

This is the benchmark where the ``rare_hypothesis_checker``'s predicted
advantage lives, and it is also the benchmark most exposed to contamination:
an item whose answer was stable before a model's training cutoff tests memory,
not reasoning. Every item therefore carries an ``as_of`` timestamp, and
``contamination.py`` uses it to build the post-cutoff subset that specification
section 4 requires.

Closed-book and retrieval-assisted performance are scored *separately*, never
pooled. Pooling them is how a retrieval result gets reported as a knowledge
result.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .base import BenchmarkAdapter, BenchmarkItem, DatasetPin, read_jsonl, verify_pin
from .metrics import (
    abstention_rate,
    exact_match,
    retrieval_dependence,
    unsupported_answer_rate,
)

__all__ = ["TemporalQAAdapter"]


class TemporalQAAdapter(BenchmarkAdapter):
    """Time-sensitive question answering against timestamped sources."""

    name = "temporal_qa"
    primary_metric = "accuracy"
    secondary_metrics = ("closed_book_score", "retrieval_assisted_score")
    harm_metrics = (
        "unsupported_answer_rate",
        "retrieval_dependence",
        "abstention_rate",
    )
    requires_human_judgement = False
    sesoi = 0.03

    pin = DatasetPin(
        source="constructed (timestamped authoritative-source questions)",
        revision="cca-temporal-v1",
        sha256="",
        licence="CC-BY-4.0 for constructed items; source licences recorded per item",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        provenance=(
            "Questions constructed against sources carrying an explicit "
            "publication timestamp, so that an item's answer can be dated. "
            "Items are tagged with as_of; the post-cutoff subset is the "
            "contamination-resistant portion. No personal data."
        ),
        cutoff_sensitive=True,
        personal_data=False,
    )

    def load(self, path: str, *, verify: bool = True) -> List[BenchmarkItem]:
        """Load timestamped questions from a local JSONL copy.

        Args:
            path: Local file with ``question``, ``answer``, ``as_of`` (ISO
                date), and ``source_url``.
            verify: Check the file hash against :attr:`pin`.

        Returns:
            Items carrying their timestamp and authoritative source.

        Raises:
            ValueError: If a record carries an unparseable ``as_of``.
        """
        if verify:
            verify_pin(path, self.pin)

        items: List[BenchmarkItem] = []
        for index, row in enumerate(read_jsonl(path)):
            as_of = str(row.get("as_of", ""))
            if as_of:
                try:
                    _dt.date.fromisoformat(as_of)
                except ValueError as error:
                    raise ValueError(
                        f"item {index}: as_of={as_of!r} is not an ISO date; "
                        "temporal validity cannot be established without one"
                    ) from error
            items.append(
                BenchmarkItem(
                    item_id=f"temporal-{index:05d}",
                    benchmark=self.name,
                    question=str(row["question"]),
                    gold_answer=str(row["answer"]),
                    metadata={
                        "as_of": as_of,
                        "source_url": row.get("source_url", ""),
                        "source_name": row.get("source_name", ""),
                        "answer_changes_over_time": bool(
                            row.get("answer_changes_over_time", True)
                        ),
                    },
                )
            )
        return items

    def normalize(self, text: str) -> str:
        """Apply free-form normalisation.

        Args:
            text: Raw answer text.

        Returns:
            Canonical form.
        """
        from normalize import normalize_answer

        return normalize_answer(text, self.name)

    def score(
        self,
        items: Sequence[BenchmarkItem],
        predictions: Sequence[str],
        *,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score accuracy, support, and retrieval dependence.

        Args:
            items: Evaluated items.
            predictions: Predicted answer texts (closed-book arm).
            extras: May carry ``abstained``, ``had_support`` (whether a citation
                or named check backed each answer), and
                ``retrieval_predictions`` for the retrieval-assisted arm.

        Returns:
            Primary, secondary, and harm outcomes.

        Raises:
            ValueError: If items and predictions are not aligned.
        """
        if len(items) != len(predictions):
            raise ValueError(f"length mismatch: {len(items)} vs {len(predictions)}")
        extras = dict(extras or {})
        golds = [i.gold_answer for i in items]

        out: Dict[str, Any] = dict(exact_match(predictions, golds, self.normalize))
        out["primary"] = out["accuracy"]

        if "abstained" in extras:
            out["abstention_rate"] = abstention_rate(extras["abstained"])
            answered = [not a for a in extras["abstained"]]
        else:
            answered = [bool(self.normalize(p)) for p in predictions]

        if "had_support" in extras:
            out["unsupported_answer_rate"] = unsupported_answer_rate(
                answered, extras["had_support"]
            )

        if "retrieval_predictions" in extras:
            retrieval = extras["retrieval_predictions"]
            if len(retrieval) != len(items):
                raise ValueError("retrieval_predictions is not aligned with items")
            closed_scores = [
                float(self.normalize(p) == self.normalize(g))
                for p, g in zip(predictions, golds)
            ]
            assisted_scores = [
                float(self.normalize(p) == self.normalize(g))
                for p, g in zip(retrieval, golds)
            ]
            out.update(retrieval_dependence(closed_scores, assisted_scores))
        else:
            out["retrieval_dependence"] = float("nan")
            out["note_retrieval"] = (
                "closed-book only; retrieval dependence unmeasured. "
                "Specification section 4 requires both arms before a knowledge "
                "claim is made."
            )

        post_cutoff = sum(1 for i in items if i.metadata.get("post_training_cutoff"))
        out["n_post_cutoff_items"] = float(post_cutoff)
        out["post_cutoff_fraction"] = post_cutoff / len(items) if items else float("nan")
        out["n_items"] = float(len(items))
        return out
