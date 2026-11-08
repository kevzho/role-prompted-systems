"""Refusal rates across the three council arms. EXPLORATORY, not pre-registered.

The manuscript's refusal paragraph (Appendix, ``app:refusal``) and the abstract's
refusal sentence were computed before this script existed. This script makes
them reproducible from the shipped records.

A refusal is an agent slot whose ``failure_flags`` contain ``"refusal"``. That is
the manuscript's definition. Absent records without that flag are errors, not
refusals, and are counted separately as ``absent_not_refusal``. On HaluEval
there are no refusals in any arm. On TruthfulQA, which is not scored for
accuracy, the rate is compared between arms on the common items: the paired
item-level bootstrap of the difference in per-item refusal share (2,000
resamples), with Benjamini-Hochberg across the three contrasts.

Usage::

    python3 analysis_refusal.py --out runs/refusal.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts.manipulation_check import _bootstrap_paired  # noqa: E402
from utils import benjamini_hochberg, write_json  # noqa: E402

ARMS = ["cca_council", "neutral_council", "persona_only_council"]
CONTRASTS = [("cca_council", "neutral_council"), ("persona_only_council", "cca_council"),
             ("persona_only_council", "neutral_council")]
SEED = 0


def _per_item(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            slots = record.get("candidates") or []
            refused = sum(1 for c in slots if "refusal" in (c.get("failure_flags") or []))
            absent_other = sum(1 for c in slots if c.get("present") is False
                               and "refusal" not in (c.get("failure_flags") or []))
            out[record["item_id"]] = {"benchmark": record.get("benchmark"), "slots": len(slots),
                                      "refused": refused, "absent_not_refusal": absent_other}
    return out


def analyse(runs_dir: str = "runs", seed: int = SEED) -> Dict[str, Any]:
    items = {arm: _per_item(os.path.join(runs_dir, "arms", f"records.{arm}.jsonl")) for arm in ARMS}
    rates: Dict[str, Any] = {}
    for arm in ARMS:
        for bench in ("halueval", "truthfulqa"):
            sel = [v for v in items[arm].values() if v["benchmark"] == bench]
            slots = sum(v["slots"] for v in sel)
            refused = sum(v["refused"] for v in sel)
            rates.setdefault(arm, {})[bench] = {
                "slots": slots, "refused": refused,
                "absent_not_refusal": sum(v["absent_not_refusal"] for v in sel),
                "rate": refused / slots if slots else float("nan")}
    common = sorted(set.intersection(*[
        {k for k, v in items[arm].items() if v["benchmark"] == "truthfulqa"} for arm in ARMS]))
    share = {arm: np.array([items[arm][k]["refused"] / items[arm][k]["slots"] for k in common])
             for arm in ARMS}
    contrasts = []
    for a, b in CONTRASTS:
        boot = _bootstrap_paired(share[a], share[b], seed=seed)
        contrasts.append({"contrast": f"{a} - {b}", "diff_points": 100 * boot["diff"],
                          "lo_points": 100 * boot["lo"], "hi_points": 100 * boot["hi"],
                          "p": boot["p_value"]})
    rejected, q_values = benjamini_hochberg([c["p"] for c in contrasts], alpha=0.05)
    for c, r, q in zip(contrasts, rejected, q_values):
        c["q"], c["fdr_rejected"] = float(q), bool(r)
    return {"status": "EXPLORATORY -- not pre-registered", "n_common_truthfulqa_items": len(common),
            "rates": rates, "truthfulqa_contrasts": contrasts}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    report = analyse(args.runs_dir)
    if args.out:
        write_json(report, args.out)
    for arm, r in report["rates"].items():
        print(f"{arm:22s} HaluEval {r['halueval']['refused']}/{r['halueval']['slots']}   "
              f"TruthfulQA {r['truthfulqa']['refused']}/{r['truthfulqa']['slots']} "
              f"= {100 * r['truthfulqa']['rate']:.2f}%")
    for c in report["truthfulqa_contrasts"]:
        print(f"{c['contrast']:45s} {c['diff_points']:+.2f} pts "
              f"[{c['lo_points']:+.2f}, {c['hi_points']:+.2f}]  q = {c['q']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
