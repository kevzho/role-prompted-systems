"""Three exploratory robustness views on the collected manipulation checks.

None of this requires a new call. All three read the same per-item check scores
the registered analysis reads, and all three are EXPLORATORY: the registered
verdict is the conjunctive rule at d >= 0.40 under Benjamini-Hochberg, and
nothing here revises it.

1. ``threshold_sensitivity`` -- how many roles pass as the standardized
   threshold is swept. The registered 0.40 is one column of this table. The
   point is not to find a threshold at which more roles pass; it is to show
   that "one of six" is a property of the data rather than of 0.40.

2. ``classification_stability`` -- each role's qualitative verdict (passes /
   indistinguishable / reversed) recomputed over resampled and subsampled item
   sets. A verdict that survives resampling is a property of the construct; one
   that does not is a property of these 200 items.

3. ``component_decomposition`` -- five of the six checks are two-part
   composites, not only the two products already reported. A composite can sit
   at zero because nothing moved, or because both parts moved and cancelled,
   and those are different findings. This separates them.

Usage::

    python3 analysis_check_robustness.py --records runs/checks/check_records.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts import PROMPT_VARIANTS, ROLE_PROMPTS  # noqa: E402

N_BOOTSTRAP = 2000
SEED = 0
REGISTERED_THRESHOLD = 0.40
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
SUBSAMPLE_FRACTIONS = [0.50, 0.75]


def _mean(values: Sequence[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


# --- the two-part structure of each check, read off prompts/__init__.py ------
#: name -> (component name, extractor, component name, extractor, combiner)
COMPONENTS: Dict[str, Any] = {
    "counter_evidence_checker": (
        "counterevidence_items", lambda r: float(len(r.get("counterevidence", []))),
        "flagged_claims", lambda r: float(sum(
            1 for c in r.get("atomic_claims", [])
            if c.get("evidence_status") in ("unsupported", "contradicted"))),
        "sum",
    ),
    "consistency_checker": (
        "flag_rate", lambda r: float(bool(
            {"contradiction", "calculation_error"} & set(r.get("failure_flags", [])))),
        "abstention_rate", lambda r: float(bool(r.get("abstain"))),
        "difference",
    ),
    "local_claim_checker": (
        "atomic_claim_count", lambda r: float(len(r.get("atomic_claims", []))),
        "sourced_weight", lambda r: 1.0 + _mean(
            [float(bool(c.get("source_or_check"))) for c in r.get("atomic_claims", [])]),
        "product",
    ),
    "explore_then_prune": (
        "diversity", lambda r: float(len({c.get("claim", "") for c in r.get("atomic_claims", [])})),
        "concentration", lambda r: 1.0 - _mean(
            [float(c.get("evidence_status") == "uncertain") for c in r.get("atomic_claims", [])]),
        "product",
    ),
    "rare_hypothesis_checker": (
        "sourced_supported", lambda r: float(sum(
            1 for c in r.get("atomic_claims", [])
            if c.get("evidence_status") == "supported" and c.get("source_or_check"))),
        "unsourced_supported", lambda r: float(sum(
            1 for c in r.get("atomic_claims", [])
            if c.get("evidence_status") == "supported" and not c.get("source_or_check"))),
        "difference",
    ),
    # negative ECE is a binned aggregate over the whole record set, not a
    # two-part composite; decomposing it would mean inventing a structure the
    # check does not have.
    "confidence_calibrator": None,
}


def _d(delta: np.ndarray) -> float:
    if delta.size < 2:
        return 0.0
    spread = float(delta.std(ddof=1))
    mean = float(delta.mean())
    if spread > 0.0:
        return mean / spread
    return 0.0 if mean == 0.0 else float(np.sign(mean) * np.inf)


def _paired(op: np.ndarray, ne: np.ndarray, seed: int = SEED) -> Dict[str, float]:
    delta = op - ne
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, delta.size, size=(N_BOOTSTRAP, delta.size))].mean(axis=1)
    lo, hi = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    return {
        "mean_operation": float(op.mean()),
        "mean_neutral": float(ne.mean()),
        "diff": float(delta.mean()),
        "lo": lo, "hi": hi,
        "standardized_effect": _d(delta),
        "interval_excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "percent_change": (float(100.0 * delta.mean() / ne.mean())
                           if ne.mean() != 0 else float("nan")),
    }


def _classify(d: float, threshold: float = REGISTERED_THRESHOLD) -> str:
    if d >= threshold:
        return "passes"
    if d <= -threshold:
        return "reversed"
    return "indistinguishable"


def _per_item(rows, role, extractor=None):
    spec = ROLE_PROMPTS[role]
    keep = [r for r in rows if role in r and all(
        r[role].get(v, {}).get("present", True) for v in PROMPT_VARIANTS)]
    fn = extractor if extractor is not None else (lambda rec: spec.check([rec]))
    return (np.array([fn(r[role]["operation"]) for r in keep], dtype=np.float64),
            np.array([fn(r[role]["neutral"]) for r in keep], dtype=np.float64))


def analyse(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scores = {role: _per_item(rows, role) for role in ROLE_PROMPTS}

    # --- 1. threshold sensitivity ------------------------------------------
    threshold_rows = []
    for t in THRESHOLDS:
        verdicts = {role: _classify(_d(op - ne), t) for role, (op, ne) in scores.items()}
        threshold_rows.append({
            "threshold": t,
            "registered": t == REGISTERED_THRESHOLD,
            "n_passes": sum(v == "passes" for v in verdicts.values()),
            "n_reversed": sum(v == "reversed" for v in verdicts.values()),
            "n_indistinguishable": sum(v == "indistinguishable" for v in verdicts.values()),
            "passing_roles": sorted(r for r, v in verdicts.items() if v == "passes"),
        })

    # --- 2. classification stability ---------------------------------------
    stability = {}
    for role, (op, ne) in scores.items():
        delta = op - ne
        n = delta.size
        rng = np.random.default_rng(SEED)
        out: Dict[str, Any] = {"n_items": int(n), "point_verdict": _classify(_d(delta))}
        counts: Dict[str, int] = {}
        ds: List[float] = []
        for _ in range(N_BOOTSTRAP):
            di = _d(delta[rng.integers(0, n, size=n)])
            ds.append(di)
            v = _classify(di)
            counts[v] = counts.get(v, 0) + 1
        arr = np.array(ds, dtype=np.float64)
        out["resample_frequency"] = {k: v / N_BOOTSTRAP for k, v in sorted(counts.items())}
        out["resampled_d_p2.5"] = float(np.quantile(arr, 0.025))
        out["resampled_d_p97.5"] = float(np.quantile(arr, 0.975))
        out["distance_to_threshold_in_resample_sd"] = (
            float(abs(abs(_d(delta)) - REGISTERED_THRESHOLD) / arr.std(ddof=1))
            if arr.std(ddof=1) > 0 else float("inf"))
        out["agrees_with_point_verdict"] = counts.get(out["point_verdict"], 0) / N_BOOTSTRAP
        sub: Dict[str, Any] = {}
        for frac in SUBSAMPLE_FRACTIONS:
            m = max(2, int(round(frac * n)))
            rng2 = np.random.default_rng(SEED + 7)
            c2: Dict[str, int] = {}
            for _ in range(N_BOOTSTRAP):
                v = _classify(_d(delta[rng2.permutation(n)[:m]]))
                c2[v] = c2.get(v, 0) + 1
            sub[f"n={m}"] = {
                "frequency": {k: v / N_BOOTSTRAP for k, v in sorted(c2.items())},
                "agrees_with_point_verdict": c2.get(out["point_verdict"], 0) / N_BOOTSTRAP,
            }
        out["subsample"] = sub
        stability[role] = out

    # --- 3. component decomposition ----------------------------------------
    decomposition: Dict[str, Any] = {}
    for role, spec in COMPONENTS.items():
        if spec is None:
            decomposition[role] = {
                "decomposable": False,
                "reason": "negative ECE is a binned aggregate, not a two-part composite",
            }
            continue
        a_name, a_fn, b_name, b_fn, form = spec
        a_op, a_ne = _per_item(rows, role, a_fn)
        b_op, b_ne = _per_item(rows, role, b_fn)
        whole_op, whole_ne = scores[role]
        decomposition[role] = {
            "decomposable": True,
            "form": form,
            "whole": _paired(whole_op, whole_ne),
            "components": {
                a_name: _paired(a_op, a_ne, SEED + 1),
                b_name: _paired(b_op, b_ne, SEED + 2),
            },
        }
    return {
        "threshold_sensitivity": threshold_rows,
        "classification_stability": stability,
        "component_decomposition": decomposition,
        "exploratory": True,
        "registered_threshold": REGISTERED_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", default="runs/checks/check_records.jsonl")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    rows = [json.loads(l) for l in open(args.records, encoding="utf-8") if l.strip()]
    report = analyse(rows)
    report["records_path"] = args.records
    if args.out:
        with open(args.out, "w", encoding="utf-8") as h:
            json.dump(report, h, indent=1, sort_keys=True)
        print(f"wrote {args.out}\n")

    print("--- 1. threshold sensitivity (registered row marked *) ---")
    print(f"{'d threshold':>12} {'passes':>7} {'reversed':>9} {'null':>6}   passing roles")
    for r in report["threshold_sensitivity"]:
        star = "*" if r["registered"] else " "
        print(f"{star}{r['threshold']:>11.2f} {r['n_passes']:>7} {r['n_reversed']:>9} "
              f"{r['n_indistinguishable']:>6}   {', '.join(r['passing_roles']) or '-'}")

    print("\n--- 2. classification stability (2,000 resamples) ---")
    print(f"{'role':26} {'point verdict':>19} {'agree':>7} {'agree n=100':>12} {'resampled d 95%':>22} {'dist to 0.40':>13}")
    for role, s in report["classification_stability"].items():
        sub = list(s["subsample"].values())[0]["agrees_with_point_verdict"]
        print(f"{role:26} {s['point_verdict']:>19} {s['agrees_with_point_verdict']:>6.0%} {sub:>11.0%}"
              f"   [{s['resampled_d_p2.5']:+.3f}, {s['resampled_d_p97.5']:+.3f}] "
              f"{s['distance_to_threshold_in_resample_sd']:>10.1f} sd")

    print("\n--- 3. component decomposition ---")
    for role, d in report["component_decomposition"].items():
        if not d["decomposable"]:
            print(f"{role:26} not decomposable ({d['reason']})")
            continue
        w = d["whole"]
        print(f"{role:26} {d['form']:>10}  whole d={w['standardized_effect']:+.3f}")
        for name, c in d["components"].items():
            moved = "moved" if c["interval_excludes_zero"] else "flat "
            pc = c["percent_change"]
            pct = "n/a" if pc != pc else f"{pc:+.1f}%"
            print(f"{'':28} {name:24} {c['mean_neutral']:7.3f} -> {c['mean_operation']:7.3f} "
                  f"({pct:>7})  d={c['standardized_effect']:+.3f}  {moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
