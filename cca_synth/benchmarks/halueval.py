"""HaluEval adapter.

Primary outcome per specification section 5: dataset-native accuracy or F1 for
the selected task and split. HaluEval ships several tasks (QA, dialogue,
summarisation, and a general user-query set); the task is a *choice* that must be
frozen before the run, not decided after seeing which one looks best. It is
declared here and recorded in the manifest.

The task is binary: given a sample, decide whether it contains hallucinated
content. That makes HaluEval one of the two benchmarks with unambiguous answer
normalisation, which is why ``REMAINING_PROMPTS.md`` P13 nominates it for the
pilot -- it sidesteps the clustering risk while that risk is being measured.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import BenchmarkAdapter, BenchmarkItem, DatasetPin, read_jsonl, verify_pin
from .metrics import abstention_rate, binary_classification, claim_level_rates

__all__ = ["HaluEvalAdapter", "HALUEVAL_TASKS"]

#: HaluEval's task splits. The selected task is frozen before the run.
HALUEVAL_TASKS: Tuple[str, ...] = ("qa", "dialogue", "summarization", "general")

#: The two answer strings the benchmark accepts. Anything else is a format
#: failure, not a wrong answer, and is counted separately.
_YES = "yes"
_NO = "no"


class HaluEvalAdapter(BenchmarkAdapter):
    """HaluEval hallucination detection, one frozen task split."""

    name = "halueval"
    primary_metric = "accuracy"
    secondary_metrics = ("f1", "precision", "recall")
    harm_metrics = (
        "false_positive_rate",
        "false_negative_rate",
        "invalid_format_rate",
        "abstention_rate",
    )
    requires_human_judgement = False
    sesoi = 0.03

    pin = DatasetPin(
        source="RUCAIBox/HaluEval",
        # Last upstream commit touching data/qa_data.json (2023-05-26); the
        # materialised file is the upstream bytes verbatim.  See
        # benchmarks/materialize.py.
        revision="207f479c555a68b8d2431631a95083b84b2accbe",
        sha256="89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88",
        licence="MIT",
        licence_url="https://github.com/RUCAIBox/HaluEval/blob/main/LICENSE",
        provenance=(
            "Li, Cheng, Zhao, Nie, and Wen, HaluEval: A Large-Scale "
            "Hallucination Evaluation Benchmark for Large Language Models, "
            "EMNLP 2023. QA split; 10,000 items."
        ),
        cutoff_sensitive=False,
        personal_data=False,
    )

    def __init__(self, task: str = "qa") -> None:
        """Freeze the task split.

        Args:
            task: One of :data:`HALUEVAL_TASKS`.

        Raises:
            ValueError: On an unknown task.
        """
        if task not in HALUEVAL_TASKS:
            raise ValueError(f"task must be one of {HALUEVAL_TASKS}; got {task!r}")
        self.task = task

    def describe(self) -> Dict[str, Any]:
        """Manifest entry, including the frozen task selection.

        Returns:
            JSON-serialisable description.
        """
        out = super().describe()
        out["task"] = self.task
        return out

    def load(self, path: str, *, verify: bool = True) -> List[BenchmarkItem]:
        """Load HaluEval items from a local JSONL copy.

        Each record is expanded into the two-way decision the benchmark poses:
        the model is shown one answer and asked whether it is hallucinated.

        Args:
            path: Local file with ``question``/``knowledge`` plus a right and a
                hallucinated answer.
            verify: Check the file hash against :attr:`pin`.

        Returns:
            Items, each carrying its gold yes/no decision.
        """
        if verify:
            verify_pin(path, self.pin)

        items: List[BenchmarkItem] = []
        for index, row in enumerate(read_jsonl(path)):
            question = str(row.get("question") or row.get("user_query") or "")
            knowledge = str(row.get("knowledge") or row.get("document") or "")
            right = row.get("right_answer") or row.get("chatgpt_response")
            hallucinated = row.get("hallucinated_answer")

            for suffix, answer, is_hallucinated in (
                ("r", right, False),
                ("h", hallucinated, True),
            ):
                if answer is None:
                    continue
                items.append(
                    BenchmarkItem(
                        item_id=f"halu-{self.task}-{index:05d}-{suffix}",
                        benchmark=self.name,
                        question=(
                            "Does the following answer contain hallucinated "
                            "content that is not supported by the knowledge?\n\n"
                            f"Question: {question}\n\nAnswer: {answer}"
                        ),
                        context=knowledge,
                        gold_answer=_YES if is_hallucinated else _NO,
                        options=(_NO, _YES),
                        gold_option=1 if is_hallucinated else 0,
                        metadata={"task": self.task, "candidate_answer": str(answer)},
                    )
                )
        return items

    def normalize(self, text: str) -> str:
        """Reduce a response to ``yes``, ``no``, or the empty string.

        The empty string denotes a format failure and is scored as such rather
        than being silently mapped onto one of the two classes -- mapping it to
        ``no`` would flatter any model that failed to answer.

        Args:
            text: Raw answer text.

        Returns:
            ``"yes"``, ``"no"``, or ``""``.
        """
        lowered = text.strip().lower()
        if not lowered:
            return ""
        # Take the first decisive token so that "Yes, the answer contains..."
        # and a bare "yes" normalise identically.
        for token in lowered.replace(",", " ").replace(".", " ").split():
            if token in (_YES, "true", "hallucinated"):
                return _YES
            if token in (_NO, "false", "supported", "faithful"):
                return _NO
        return ""

    def is_valid_format(self, text: str) -> bool:
        """Whether a response parses to a decision.

        Args:
            text: Raw answer text.

        Returns:
            ``True`` when the text yields ``yes`` or ``no``.
        """
        return self.normalize(text) in (_YES, _NO)

    def score(
        self,
        items: Sequence[BenchmarkItem],
        predictions: Sequence[str],
        *,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score the binary decision.

        Items whose prediction does not parse are excluded from the accuracy and
        F1 computation and reported as ``invalid_format_rate``. Section 8
        requires publishing exclusions rather than absorbing them, and scoring an
        unparseable response as wrong would confound two different failures.

        Args:
            items: Evaluated items.
            predictions: Predicted answer texts.
            extras: May carry ``abstained``, ``predicted_claim_flags``, and
                ``gold_claim_flags``.

        Returns:
            Primary, secondary, and harm outcomes.

        Raises:
            ValueError: If items and predictions are not aligned.
        """
        if len(items) != len(predictions):
            raise ValueError(f"length mismatch: {len(items)} vs {len(predictions)}")
        extras = dict(extras or {})

        parsed: List[bool] = []
        gold: List[bool] = []
        invalid = 0
        for item, prediction in zip(items, predictions):
            decision = self.normalize(prediction)
            if decision not in (_YES, _NO):
                invalid += 1
                continue
            parsed.append(decision == _YES)
            gold.append(self.normalize(item.gold_answer) == _YES)

        out: Dict[str, Any] = dict(binary_classification(parsed, gold))
        out["primary"] = out["accuracy"]
        out["invalid_format_rate"] = invalid / len(items) if items else float("nan")
        out["n_excluded_unparseable"] = float(invalid)
        out["n_items"] = float(len(items))
        out["task"] = self.task

        if "abstained" in extras:
            out["abstention_rate"] = abstention_rate(extras["abstained"])
        if "predicted_claim_flags" in extras and "gold_claim_flags" in extras:
            out.update(
                claim_level_rates(
                    extras["predicted_claim_flags"], extras["gold_claim_flags"]
                )
            )
        return out
