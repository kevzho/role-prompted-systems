"""TruthfulQA adapter.

Primary outcome per specification section 5: the dataset-native
truthful-and-informative score. That metric is *judged*, not computed: the
official evaluation fine-tunes GPT-judge models, and the honest position is that
this adapter cannot produce a primary number on its own. It consumes judge
labels supplied by ``cca_synth/judging/`` and refuses to guess when they are
absent, because limitation L16 forbids using an LLM judge as a primary outcome
without a human audit sample quantifying its error rate.

The generative split is used rather than the multiple-choice one. MC1/MC2 are
easier to score but change the task: a council that deliberates over free text
and then picks from four options is not the system the paper describes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .base import BenchmarkAdapter, BenchmarkItem, DatasetPin, read_jsonl, verify_pin
from .metrics import (
    abstention_rate,
    judge_disagreement,
    truthful_and_informative,
    verbosity,
)

__all__ = ["TruthfulQAAdapter"]


class TruthfulQAAdapter(BenchmarkAdapter):
    """TruthfulQA, generation split."""

    name = "truthfulqa"
    primary_metric = "truthful_and_informative"
    secondary_metrics = ("truthful", "informative", "mean_output_tokens")
    harm_metrics = ("abstention_rate", "judge_disagreement_rate", "p95_output_tokens")
    requires_human_judgement = True
    sesoi = 0.03

    pin = DatasetPin(
        source="truthfulqa/truthful_qa",
        # Hugging Face repo commit (2024-01-04).  The materialised file is the
        # generation-split parquet at this commit converted deterministically
        # to JSONL by benchmarks/materialize.py, which also verifies the
        # parquet's own hash before converting.
        revision="741b8276f2d1982aa3d5b832d3ee81ed3b896490",
        sha256="6b525955feaf571993854d5234041824d6906614892635992a4497b82c20235d",
        licence="Apache-2.0",
        licence_url="https://github.com/sylinrl/TruthfulQA/blob/main/LICENSE",
        provenance=(
            "Lin, Hilton, and Evans, TruthfulQA: Measuring How Models Mimic "
            "Human Falsehoods, ACL 2022. Generation split, 817 questions."
        ),
        cutoff_sensitive=False,
        personal_data=False,
    )

    def load(self, path: str, *, verify: bool = True) -> List[BenchmarkItem]:
        """Load TruthfulQA generation items from a local JSONL copy.

        Args:
            path: Local file with one record per line carrying ``question``,
                ``best_answer``, ``correct_answers``, ``incorrect_answers``, and
                ``category``.
            verify: Check the file hash against :attr:`pin`.

        Returns:
            Items with the gold set carried in metadata for the judge.
        """
        if verify:
            verify_pin(path, self.pin)
        items: List[BenchmarkItem] = []
        for index, row in enumerate(read_jsonl(path)):
            items.append(
                BenchmarkItem(
                    item_id=f"tqa-{index:05d}",
                    benchmark=self.name,
                    question=str(row["question"]),
                    gold_answer=str(row.get("best_answer", "")),
                    metadata={
                        "category": row.get("category", ""),
                        # The judge needs both sets; the agents never see them.
                        "correct_answers": list(row.get("correct_answers", [])),
                        "incorrect_answers": list(row.get("incorrect_answers", [])),
                    },
                )
            )
        return items

    def normalize(self, text: str) -> str:
        """Apply free-form normalisation.

        TruthfulQA has no official string normalisation because its metric is
        judged rather than matched. Free-form normalisation is used only for
        clustering candidate answers into a shared option space, and
        ``normalize.clustering_risk_report`` gates whether that clustering is
        accurate enough for this benchmark to support a conclusion.

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
        """Score against judge labels.

        Args:
            items: Evaluated items.
            predictions: Predicted answer texts.
            extras: Must carry ``truthful`` and ``informative`` label sequences.
                May carry ``judge_labels`` (per-item, per-judge truthfulness),
                ``abstained``, and ``output_tokens``.

        Returns:
            Primary, secondary, and harm outcomes.

        Raises:
            ValueError: If judge labels are absent. TruthfulQA's primary outcome
                is not computable without them, and returning a placeholder
                would put an uncomputed number in a results table.
        """
        extras = dict(extras or {})
        if "truthful" not in extras or "informative" not in extras:
            raise ValueError(
                "TruthfulQA's primary outcome is judged. Supply 'truthful' and "
                "'informative' label sequences from cca_synth/judging/. Per "
                "limitation L16 an LLM judge may not stand in as the primary "
                "outcome without a human audit sample quantifying its error rate."
            )

        out: Dict[str, Any] = dict(
            truthful_and_informative(extras["truthful"], extras["informative"])
        )
        out["primary"] = out["truthful_and_informative"]

        if "abstained" in extras:
            out["abstention_rate"] = abstention_rate(extras["abstained"])
        if "output_tokens" in extras:
            out.update(verbosity(extras["output_tokens"]))
        if "judge_labels" in extras:
            disagreement = judge_disagreement(extras["judge_labels"])
            out["judge_disagreement_rate"] = disagreement["disagreement_rate"]
            out["judge_mean_pairwise_agreement"] = disagreement["mean_pairwise_agreement"]
            out["n_items_multiply_judged"] = disagreement["n_items_multiply_judged"]

        out["n_items"] = float(len(items))
        return out
