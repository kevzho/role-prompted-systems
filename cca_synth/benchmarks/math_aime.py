"""MATH and AIME adapter.

Primary outcome per specification section 5: exact final-answer accuracy using
the *official* normalisation. That normalisation already exists in
``cca_synth/normalize.py`` -- ``_normalize_mathematical`` implements the
boxed-answer extraction, LaTeX-wrapper stripping, and numeric canonicalisation
the official evaluators use -- so this adapter delegates rather than
reimplementing it. Two independent normalisers would drift, and the vote matrix
must be built with the same function that scores the answer.

Harm outcomes are invalid format and reasoning-token cost. Both matter for
``consistency_checker``, whose predicted regression in ``PREREGISTRATION.md``
section 3 is exactly token and latency inflation from re-checking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .base import BenchmarkAdapter, BenchmarkItem, DatasetPin, read_jsonl, verify_pin
from .metrics import abstention_rate, exact_match, invalid_format_rate, verbosity

__all__ = ["MathAimeAdapter"]


class MathAimeAdapter(BenchmarkAdapter):
    """Competition mathematics with exact-match scoring."""

    name = "math_aime"
    primary_metric = "accuracy"
    secondary_metrics = ("mean_output_tokens",)
    harm_metrics = ("invalid_format_rate", "p95_output_tokens", "abstention_rate")
    requires_human_judgement = False
    sesoi = 0.03

    pin = DatasetPin(
        source="AIME (constructed release split) + MATH test",
        revision="aime-1983-2024+math-test-v1",
        sha256="",
        licence="MIT (MATH); AIME problems used under fair use for research",
        licence_url="https://github.com/hendrycks/math/blob/main/LICENSE",
        provenance=(
            "Hendrycks et al., Measuring Mathematical Problem Solving With the "
            "MATH Dataset, NeurIPS 2021 (test split), combined with AIME "
            "problems. AIME items after a model's training cutoff are the "
            "contamination-resistant subset and are flagged in metadata."
        ),
        cutoff_sensitive=True,
        personal_data=False,
    )

    def load(self, path: str, *, verify: bool = True) -> List[BenchmarkItem]:
        """Load problems from a local JSONL copy.

        Args:
            path: Local file with ``problem``, ``answer``, and optionally
                ``level``, ``type``, and ``year``.
            verify: Check the file hash against :attr:`pin`.

        Returns:
            Items carrying the canonical gold answer.
        """
        if verify:
            verify_pin(path, self.pin)
        items: List[BenchmarkItem] = []
        for index, row in enumerate(read_jsonl(path)):
            answer = row.get("answer")
            if answer is None:
                # A MATH record stores the answer inside \boxed{} in solution.
                answer = row.get("solution", "")
            items.append(
                BenchmarkItem(
                    item_id=f"math-{index:05d}",
                    benchmark=self.name,
                    question=str(row.get("problem") or row.get("question") or ""),
                    gold_answer=str(answer),
                    metadata={
                        "level": row.get("level", ""),
                        "type": row.get("type", ""),
                        # Used by contamination.py to build the post-cutoff
                        # subset; specification section 4 requires temporal
                        # validity where it matters.
                        "year": row.get("year"),
                        "source": row.get("source", ""),
                    },
                )
            )
        return items

    def normalize(self, text: str) -> str:
        """Apply the official MATH/AIME normalisation.

        Args:
            text: Raw answer text.

        Returns:
            Canonical form.
        """
        from normalize import normalize_answer

        return normalize_answer(text, self.name)

    def is_valid_format(self, text: str) -> bool:
        """Whether a response yields a non-empty canonical answer.

        Args:
            text: Raw answer text.

        Returns:
            ``True`` when normalisation produces something to compare.
        """
        return bool(self.normalize(text))

    def score(
        self,
        items: Sequence[BenchmarkItem],
        predictions: Sequence[str],
        *,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score exact final-answer accuracy.

        Unlike HaluEval, an unparseable answer here *is* scored as wrong rather
        than excluded: the task is to produce a final answer, and producing no
        parseable answer is a failure at the task rather than a harness defect.
        The format-failure rate is reported alongside so the two are separable.

        Args:
            items: Evaluated items.
            predictions: Predicted answer texts.
            extras: May carry ``abstained`` and ``output_tokens``.

        Returns:
            Primary, secondary, and harm outcomes.

        Raises:
            ValueError: If items and predictions are not aligned.
        """
        if len(items) != len(predictions):
            raise ValueError(f"length mismatch: {len(items)} vs {len(predictions)}")
        extras = dict(extras or {})

        out: Dict[str, Any] = dict(
            exact_match(predictions, [i.gold_answer for i in items], self.normalize)
        )
        out["primary"] = out["accuracy"]
        out["invalid_format_rate"] = invalid_format_rate(predictions, self.is_valid_format)

        if "abstained" in extras:
            out["abstention_rate"] = abstention_rate(extras["abstained"])
        if "output_tokens" in extras:
            out.update(verbosity(extras["output_tokens"]))
            out["reasoning_token_cost"] = out["mean_output_tokens"]

        post_cutoff = [i for i in items if i.metadata.get("post_training_cutoff")]
        out["n_post_cutoff_items"] = float(len(post_cutoff))
        out["n_items"] = float(len(items))
        return out
