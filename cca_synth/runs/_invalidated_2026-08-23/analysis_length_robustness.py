"""Is the manipulation-check effect an output-length effect?

The design length-matches the *prompt* -- control variants are padded to within
10% of the operation prompt's character count -- so "the operation helped" is
not confounded with "a longer instruction helped". Nothing in the design matches
the *output*. Three of the six checks are unbounded per-record counts
(counterevidence items, atomic claims, distinct claim texts), and a count rises
for free with a longer answer. If the operation prompt simply makes the model
write more, those checks move without the operation occurring, which is the one
inference the instrument exists to block.

This asks the question from the collected records, with no new calls. For each
role it regresses the paired check difference on the paired output-token
difference:

    (s_op - s_neutral) = a + b (t_op - t_neutral)

``a`` is the expected check difference between two answers of *equal output
length*, so it is the effect with the length channel removed. ``b`` is the price
in check units of one extra output token. Both are bootstrapped over the same
2,000 paired item resamples used everywhere else in this repository, and ``a``
is standardized by the same denominator as the published ``d`` so the two are
directly comparable.

Usage::

    python3 analysis_length_robustness.py --records runs/checks/check_records.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts import PROMPT_VARIANTS, ROLE_PROMPTS  # noqa: E402

N_BOOTSTRAP = 2000
SEED = 0
MIN_STANDARDIZED_EFFECT = 0.40


def _load(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fit(delta_s: np.ndarray, delta_t: np.ndarray) -> tuple:
    """OLS intercept and slope of delta_s on delta_t."""
    if delta_t.size < 2 or float(delta_t.std()) == 0.0:
        return float(delta_s.mean()), 0.0
    slope, intercept = np.polyfit(delta_t, delta_s, 1)
    return float(intercept), float(slope)


def analyse_role(role: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    spec = ROLE_PROMPTS[role]
    keep = [
        r for r in rows
        if role in r and all(
            r[role].get(v, {}).get("present", True) for v in PROMPT_VARIANTS
        )
    ]
    if not keep:
        return {"role": role, "n_items": 0, "error": "no complete items"}

    s = {v: np.array([spec.check([r[role][v]]) for r in keep], dtype=np.float64)
         for v in ("operation", "neutral")}
    t = {v: np.array([float(r[role][v].get("output_tokens", np.nan)) for r in keep],
                     dtype=np.float64)
         for v in ("operation", "neutral")}

    delta_s = s["operation"] - s["neutral"]
    delta_t = t["operation"] - t["neutral"]
    spread = float(delta_s.std(ddof=1)) if delta_s.size > 1 else 0.0

    raw_diff = float(delta_s.mean())
    raw_d = raw_diff / spread if spread > 0 else 0.0
    intercept, slope = _fit(delta_s, delta_t)
    adj_d = intercept / spread if spread > 0 else 0.0

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, delta_s.size, size=(N_BOOTSTRAP, delta_s.size))
    draws = np.array([_fit(delta_s[i], delta_t[i])[0] for i in idx])
    lo, hi = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))

    corr = (float(np.corrcoef(delta_s, delta_t)[0, 1])
            if delta_s.std() > 0 and delta_t.std() > 0 else float("nan"))

    return {
        "role": role,
        "check": spec.check_name,
        "n_items": int(delta_s.size),
        "mean_output_tokens_operation": float(t["operation"].mean()),
        "mean_output_tokens_neutral": float(t["neutral"].mean()),
        "mean_token_difference": float(delta_t.mean()),
        "corr_check_diff_vs_token_diff": corr,
        "tokens_per_check_unit_slope": slope,
        "raw_difference": raw_diff,
        "raw_d": raw_d,
        "length_adjusted_difference": intercept,
        "length_adjusted_difference_lo": lo,
        "length_adjusted_difference_hi": hi,
        "length_adjusted_d": adj_d,
        "share_of_effect_from_length": (
            float(1.0 - intercept / raw_diff) if raw_diff != 0 else float("nan")
        ),
        "passes_raw": bool(abs(raw_d) >= MIN_STANDARDIZED_EFFECT and raw_d > 0),
        "passes_adjusted": bool(abs(adj_d) >= MIN_STANDARDIZED_EFFECT and adj_d > 0),
        "verdict_changes": bool(
            (abs(raw_d) >= MIN_STANDARDIZED_EFFECT and raw_d > 0)
            != (abs(adj_d) >= MIN_STANDARDIZED_EFFECT and adj_d > 0)
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="runs/checks/check_records.jsonl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    rows = _load(args.records)
    results = [analyse_role(role, rows) for role in ROLE_PROMPTS]
    report = {
        "n_records": len(rows),
        "records_path": args.records,
        "n_bootstrap": N_BOOTSTRAP,
        "roles": results,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, sort_keys=True)
        print(f"wrote {args.out}")

    print(f"\n{'role':26} {'d raw':>8} {'d adj':>8} {'Δtokens':>9} {'r':>7}  verdict")
    for r in results:
        if r.get("n_items", 0) == 0:
            continue
        flag = "CHANGES" if r["verdict_changes"] else ""
        print(f"{r['role']:26} {r['raw_d']:+8.3f} {r['length_adjusted_d']:+8.3f} "
              f"{r['mean_token_difference']:+9.1f} {r['corr_check_diff_vs_token_diff']:+7.3f}  {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
