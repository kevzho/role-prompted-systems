"""AdvConsistency adapter.

Specification section 6 is a construction protocol, and until it has been
executed with expert annotators this benchmark does not exist -- ``advconsistency``
is currently a string in ``config.BENCHMARKS`` and nothing more. The construction
toolkit lives in ``cca_synth/advconsistency/``; this module is only the
*evaluation* half, and it deliberately refuses to score an unfrozen corpus.

Primary outcome per section 5: accuracy on the seeded local falsehood, with
hidden labels. Secondary and harm outcomes: long-range integration accuracy and
position bias.

The phenomenon is defined **independently of the local-claim-checker role**, per
the first bullet of section 6. Defining it as "what that role detects" would make
the benchmark circular and the role's measured advantage tautological. The
definition used here is:

    A *seeded local contradiction* is a single clause in a passage that
    contradicts a fact stated elsewhere in the same passage, where the
    contradiction is decidable from the passage alone by a reader who has read
    it in full, and where the contradicting clause is locally plausible -- that
    is, a reader who saw only the surrounding sentence would have no reason to
    doubt it.

That definition mentions no agent, no prompt, and no model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .base import BenchmarkAdapter, BenchmarkItem, DatasetPin, read_jsonl, verify_pin
from .metrics import abstention_rate, binary_classification

__all__ = ["AdvConsistencyAdapter", "PHENOMENON_DEFINITION"]

PHENOMENON_DEFINITION = (
    "A seeded local contradiction is a single clause in a passage that "
    "contradicts a fact stated elsewhere in the same passage, where the "
    "contradiction is decidable from the passage alone by a reader who has read "
    "it in full, and where the contradicting clause is locally plausible: a "
    "reader seeing only the surrounding sentence would have no reason to doubt "
    "it. Defined without reference to any agent, prompt, or model."
)


class AdvConsistencyAdapter(BenchmarkAdapter):
    """Paired passages with seeded local contradictions and positive controls."""

    name = "advconsistency"
    primary_metric = "accuracy"
    secondary_metrics = ("long_range_integration_accuracy", "positive_control_accuracy")
    harm_metrics = (
        "over_flagging_rate",
        "position_bias",
        "false_positive_rate",
        "abstention_rate",
    )
    requires_human_judgement = False
    sesoi = 0.03

    pin = DatasetPin(
        source="constructed (see cca_synth/advconsistency/ and ADVCONSISTENCY_DATACARD.md)",
        revision="UNBUILT",
        sha256="",
        licence="CC-BY-4.0 for constructed passages; source licences per item",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        provenance=(
            "Not yet constructed. Specification section 6 requires expert "
            "label creation, an ambiguity pilot, a frozen hidden test split, "
            "and a dedup pass before this may be called a benchmark."
        ),
        cutoff_sensitive=False,
        personal_data=False,
    )

    def load(self, path: str, *, verify: bool = True) -> List[BenchmarkItem]:
        """Load constructed items from a local JSONL copy.

        Args:
            path: Local file produced by the construction toolkit.
            verify: Check the file hash against :attr:`pin`.

        Returns:
            Items carrying contradiction position and control status.

        Raises:
            RuntimeError: If the pin still reads ``UNBUILT``. Scoring an
                unfrozen corpus would produce a number that cannot be
                reproduced, and section 6 requires a frozen hidden split.
        """
        if self.pin.revision == "UNBUILT":
            raise RuntimeError(
                "AdvConsistency has not been constructed. The toolkit in "
                "cca_synth/advconsistency/ implements every mechanical step; "
                "what remains is human: an ethics determination, paid domain "
                "experts, ~150 authored pilot passages, the ambiguity pilot, "
                "and adjudication. Then advconsistency.freeze_split() produces "
                "the release revision and sha256 to record in this pin. "
                "See logistics/ADVCONSISTENCY_DATACARD.md section 11."
            )
        if verify:
            verify_pin(path, self.pin)

        items: List[BenchmarkItem] = []
        for index, row in enumerate(read_jsonl(path)):
            contains = bool(row["contains_contradiction"])
            items.append(
                BenchmarkItem(
                    item_id=str(row.get("item_id") or f"advc-{index:05d}"),
                    benchmark=self.name,
                    question=(
                        "Does this passage contain an internal contradiction? "
                        "If so, quote the contradicting clause."
                    ),
                    context=str(row["passage"]),
                    gold_answer="yes" if contains else "no",
                    options=("no", "yes"),
                    gold_option=1 if contains else 0,
                    metadata={
                        "pair_id": row.get("pair_id", ""),
                        "is_positive_control": bool(row.get("is_positive_control", False)),
                        "contradiction_position": row.get("contradiction_position"),
                        "passage_length_bucket": row.get("passage_length_bucket", ""),
                        "topic": row.get("topic", ""),
                        "requires_long_range": bool(row.get("requires_long_range", False)),
                    },
                )
            )
        return items

    def normalize(self, text: str) -> str:
        """Reduce a response to ``yes``, ``no``, or the empty string.

        Args:
            text: Raw answer text.

        Returns:
            ``"yes"``, ``"no"``, or ``""``.
        """
        lowered = text.strip().lower()
        for token in lowered.replace(",", " ").replace(".", " ").split():
            if token in ("yes", "true", "contradiction", "contradictory"):
                return "yes"
            if token in ("no", "false", "consistent"):
                return "no"
        return ""

    def score(
        self,
        items: Sequence[BenchmarkItem],
        predictions: Sequence[str],
        *,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score detection, over-flagging, and position bias.

        Positive controls carry the harm outcome: they are passages where global
        context legitimately resolves an apparent local ambiguity, so flagging
        them is a false alarm. Without them, a model that answers "yes" to
        everything scores perfectly on the primary outcome.

        Args:
            items: Evaluated items.
            predictions: Predicted answer texts.
            extras: May carry ``abstained``.

        Returns:
            Primary, secondary, and harm outcomes.

        Raises:
            ValueError: If items and predictions are not aligned, or if the
                corpus carries no positive controls.
        """
        if len(items) != len(predictions):
            raise ValueError(f"length mismatch: {len(items)} vs {len(predictions)}")
        extras = dict(extras or {})

        controls = [i for i in items if i.metadata.get("is_positive_control")]
        if not controls:
            raise ValueError(
                "no positive controls present. Section 6 requires them so the "
                "benchmark can detect over-flagging; without them a "
                "flag-everything system scores perfectly."
            )

        predicted: List[bool] = []
        gold: List[bool] = []
        for item, prediction in zip(items, predictions):
            predicted.append(self.normalize(prediction) == "yes")
            gold.append(item.gold_option == 1)

        out: Dict[str, Any] = dict(binary_classification(predicted, gold))
        out["primary"] = out["accuracy"]

        def subset(predicate) -> Dict[str, float]:
            """Accuracy over the items matching a predicate."""
            pairs = [
                (p, g)
                for item, p, g in zip(items, predicted, gold)
                if predicate(item)
            ]
            if not pairs:
                return {"accuracy": float("nan"), "n": 0.0}
            return {
                "accuracy": sum(int(p == g) for p, g in pairs) / len(pairs),
                "n": float(len(pairs)),
            }

        control_scores = subset(lambda i: i.metadata.get("is_positive_control"))
        out["positive_control_accuracy"] = control_scores["accuracy"]
        out["n_positive_controls"] = control_scores["n"]
        out["over_flagging_rate"] = (
            1.0 - control_scores["accuracy"]
            if control_scores["accuracy"] == control_scores["accuracy"]
            else float("nan")
        )

        long_range = subset(lambda i: i.metadata.get("requires_long_range"))
        out["long_range_integration_accuracy"] = long_range["accuracy"]
        out["n_long_range_items"] = long_range["n"]

        # Position bias: accuracy by where in the passage the contradiction sits.
        # A model that only catches early contradictions has a position bias
        # that a pooled accuracy hides entirely.
        by_position: Dict[str, Dict[str, float]] = {}
        for bucket in ("early", "middle", "late"):
            by_position[bucket] = subset(
                lambda i, b=bucket: i.metadata.get("contradiction_position") == b
            )
        available = [
            v["accuracy"] for v in by_position.values() if v["accuracy"] == v["accuracy"]
        ]
        out["accuracy_by_position"] = {k: v["accuracy"] for k, v in by_position.items()}
        out["position_bias"] = (
            max(available) - min(available) if len(available) >= 2 else float("nan")
        )

        if "abstained" in extras:
            out["abstention_rate"] = abstention_rate(extras["abstained"])
        out["n_items"] = float(len(items))
        out["phenomenon_definition"] = PHENOMENON_DEFINITION
        return out
