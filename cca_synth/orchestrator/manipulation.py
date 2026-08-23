"""Collect the records the manipulation checks need, then score them.

This is contribution 2's measurement path. ``prompts.manipulation_check`` could
already *score* a variant-keyed records file; nothing wrote one, so the primary
outcome registered in ``PREREGISTRATION.md`` §11 A2.1 had no way to be measured.

The design is deliberately thin: run the existing :class:`~orchestrator.council.Council`
once per prompt variant over the *same* items with the *same* seed, then join the
three record streams by item and role. Every property the council enforces --
agent isolation, redaction before dispatch, schema rejection rather than repair,
snapshot pinning, provenance derived from providers -- is inherited rather than
reimplemented, and the only thing that differs between the three passes is the
prompt variant. That is what makes the contrast attributable to the operation
text.

Cost is ``3 * 6 * n_items`` calls: A2.4's 3,600 at 200 items.

Usage::

    python -m orchestrator.manipulation --items dev_items.jsonl \\
        --out-dir runs/checks --provider anthropic --model <snapshot> --limit 200

Writes into ``--out-dir``:

``records.<variant>.jsonl``
    Raw §7 records per variant, one file each, plus a manifest per variant.
``check_records.jsonl``
    The joined, variant-keyed file ``prompts.manipulation_check`` consumes.
``manipulation_report.json``
    The scored report, including the H6 verdict per role.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROLES  # noqa: E402
from orchestrator.collect import load_items  # noqa: E402
from orchestrator.council import Council  # noqa: E402
from orchestrator.providers import build_provider  # noqa: E402
from prompts import PROMPT_VARIANTS  # noqa: E402
from prompts.manipulation_check import run_checks  # noqa: E402
from utils import format_table, get_logger, write_json  # noqa: E402

__all__ = ["collect_variants", "join_variant_records", "main"]


def collect_variants(
    items_path: str,
    out_dir: str,
    provider_kind: str,
    model_snapshot: str,
    limit: Optional[int] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    seed: int = 1337,
    timeout_s: float = 120.0,
    max_retries: int = 2,
    max_workers: int = 1,
    resume: bool = False,
    azure_endpoint: Optional[str] = None,
    input_price_per_million: Optional[float] = None,
    output_price_per_million: Optional[float] = None,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the council once per prompt variant over the same items.

    The seed, model snapshot, sampling parameters, and item stream are identical
    across variants by construction; only ``Council.variant`` differs. A shared
    seed matters here in a way it does not for a single run: the three passes are
    a paired comparison, and varying the seed alongside the prompt would confound
    the operation effect with sampling noise.

    Args:
        items_path: Benchmark items JSONL.
        out_dir: Destination directory.
        provider_kind: ``stub``, ``anthropic``, or ``openai``.
        model_snapshot: Pinned snapshot; moving aliases are rejected upstream.
        limit: Process at most this many items.
        max_tokens: Output cap per call.
        temperature: Sampling temperature.
        seed: Seed, shared across variants.
        timeout_s: Per-call timeout.
        max_retries: Retries per call.
        max_workers: Items processed concurrently in each variant.
        resume: Skip IDs already present in each variant output.
        azure_endpoint: Azure resource endpoint for ``azure-openai``.
        input_price_per_million: Azure input-token price for accounting.
        output_price_per_million: Azure output-token price for accounting.
        logger: Optional logger.

    Returns:
        Mapping from variant name to that variant's run manifest, plus a
        ``provenance`` key derived from the providers.
    """
    log = logger or get_logger()
    os.makedirs(out_dir, exist_ok=True)

    manifests: Dict[str, Any] = {}
    provenance = "real"
    for variant in PROMPT_VARIANTS:
        # A fresh provider per variant, so per-variant call statistics and cost
        # are not pooled across passes.
        provider_kwargs: Dict[str, Any] = {
            "timeout_s": timeout_s,
            "max_retries": max_retries,
        }
        if provider_kind == "azure-openai":
            provider_kwargs.update(
                endpoint=azure_endpoint,
                input_price_per_million=input_price_per_million,
                output_price_per_million=output_price_per_million,
            )
        provider = build_provider(provider_kind, model_snapshot, **provider_kwargs)
        council = Council(
            providers={role: provider for role in ROLES},
            variant=variant,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            condition=f"manipulation_check/{variant}",
            max_workers=max_workers,
        )
        if council.provenance != "real":
            provenance = "simulated"

        path = os.path.join(out_dir, f"records.{variant}.jsonl")
        log.info("collecting variant=%s -> %s", variant, path)
        manifest = council.run(
            load_items(items_path, limit), path, logger=log, resume=resume
        )
        write_json(manifest, os.path.splitext(path)[0] + ".manifest.json")
        manifests[variant] = manifest

        rates = manifest["exclusion_rates"]
        log.info(
            "variant=%s items=%d refusal %.3f truncation %.3f malformed %.3f error %.3f",
            variant, manifest["n_items"], rates["refusal_rate"],
            rates["truncation_rate"], rates["malformed_rate"], rates["error_rate"],
        )

    return {"variants": manifests, "provenance": provenance}


def _index_by_item(path: str) -> Dict[str, Mapping[str, Any]]:
    """Load one variant's records, keyed by item id.

    Args:
        path: Records JSONL.

    Returns:
        Mapping from ``item_id`` to the record.

    Raises:
        ValueError: On a duplicate item id, which would make the join ambiguous.
    """
    indexed: Dict[str, Mapping[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            item_id = str(record["item_id"])
            if item_id in indexed:
                raise ValueError(f"duplicate item_id '{item_id}' in {path}")
            indexed[item_id] = record
    return indexed


def _with_outcome(
    candidate: Mapping[str, Any], record: Mapping[str, Any]
) -> Dict[str, Any]:
    """Copy a candidate, adding ``correct`` where the item is labelled.

    Args:
        candidate: One role's candidate record.
        record: The enclosing item record, carrying ``gold_option``.

    Returns:
        A new dict; the input is not mutated.
    """
    enriched = dict(candidate)
    gold = record.get("gold_option", -1)
    try:
        gold_index = int(gold)
    except (TypeError, ValueError):
        return enriched
    if gold_index < 0:
        # Unlabelled: no outcome exists, so assert none.
        return enriched
    if not candidate.get("present", True):
        # An absent record has no answer to be right or wrong about. check_role
        # drops it pairwise anyway; omitting the field keeps that decision in
        # one place.
        return enriched
    enriched["correct"] = bool(int(candidate.get("answer_option", -1)) == gold_index)
    return enriched


def join_variant_records(out_dir: str, output_path: str) -> Dict[str, Any]:
    """Join per-variant records into the variant-keyed file the checks consume.

    Only items present in *every* variant are emitted, and the count dropped is
    returned rather than logged and forgotten: an item that failed under one
    variant only is not a paired observation, and silently keeping it would let
    a variant-dependent failure rate masquerade as an operation effect.

    Missing role slots are carried through with their ``present`` flag intact.
    Excluding them is :func:`prompts.manipulation_check.check_role`'s job, done
    per role rather than per item, so one role's refusal does not discard the
    other five roles' observations for that item.

    Each emitted candidate gains a derived ``correct`` field, because
    ``prompts._check_calibration`` needs outcomes and returns 0.0 without them --
    which silently made the confidence calibrator's check unscoreable, and so
    unpassable, no matter what the model did. The value is computed from the
    item's gold label against the candidate's ``answer_option``; it is a join,
    not an assertion. Items whose ``gold_option`` is ``-1`` are unlabelled, so
    ``correct`` is omitted and the check masks them, matching how every
    coalition metric treats a ``-1`` gold.

    Args:
        out_dir: Directory holding ``records.<variant>.jsonl``.
        output_path: Destination JSONL.

    Returns:
        A join summary with per-variant counts and the intersection size.

    Raises:
        FileNotFoundError: If any variant's records file is missing.
    """
    indexed: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for variant in PROMPT_VARIANTS:
        path = os.path.join(out_dir, f"records.{variant}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing records for variant '{variant}': {path}")
        indexed[variant] = _index_by_item(path)

    common = set(indexed[PROMPT_VARIANTS[0]])
    for variant in PROMPT_VARIANTS[1:]:
        common &= set(indexed[variant])
    # Sorted, so the joined file and every bootstrap over it are reproducible.
    ordered = sorted(common)

    n_written = 0
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for item_id in ordered:
            payload: Dict[str, Any] = {"item_id": item_id}
            for role in ROLES:
                by_variant: Dict[str, Any] = {}
                for variant in PROMPT_VARIANTS:
                    record = indexed[variant][item_id]
                    candidates = record.get("candidates", [])
                    match = next((c for c in candidates if c.get("role") == role), None)
                    if match is not None:
                        by_variant[variant] = _with_outcome(match, record)
                if len(by_variant) == len(PROMPT_VARIANTS):
                    payload[role] = by_variant
            handle.write(json.dumps(payload))
            handle.write("\n")
            n_written += 1

    return {
        "n_per_variant": {v: len(indexed[v]) for v in PROMPT_VARIANTS},
        "n_intersection": len(ordered),
        "n_written": n_written,
        "n_dropped_not_in_all_variants": {
            v: len(indexed[v]) - len(ordered) for v in PROMPT_VARIANTS
        },
        "output_path": output_path,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Returns:
        ``0`` if every role passes its check, ``1`` otherwise, so this can gate
        the benchmark arms. A2.6 commits to reporting either outcome, so a
        non-zero exit is a signal to read the report, not a reason to discard it.
    """
    parser = argparse.ArgumentParser(
        description="Collect and score the six role manipulation checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--items", required=True, help="Dev-set items JSONL.")
    parser.add_argument("--out-dir", required=True, help="Destination directory.")
    parser.add_argument(
        "--provider", default="stub",
        choices=["stub", "anthropic", "anthropic-foundry", "openai", "azure-openai", "xai"],
        help="Provider used for all three variants.",
    )
    parser.add_argument(
        "--model", default="stub-provider-2026-07-28",
        help="Pinned model snapshot. Moving aliases are rejected.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N items.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Output cap per call.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=1337, help="Shared across variants.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-call timeout.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per call.")
    parser.add_argument("--max-workers", type=int, default=1, help="Concurrent items.")
    parser.add_argument("--resume", action="store_true", help="Resume each variant output.")
    parser.add_argument("--paid-item-cap", type=int, default=200)
    parser.add_argument("--allow-large-paid-run", action="store_true")
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--input-price-per-million", type=float, default=None)
    parser.add_argument("--output-price-per-million", type=float, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="Bootstrap seed.")
    parser.add_argument(
        "--strict-real", action="store_true",
        help="Refuse to write a report if any provider is synthetic.",
    )
    parser.add_argument(
        "--skip-collect", action="store_true",
        help="Score records already in --out-dir instead of calling any provider.",
    )
    args = parser.parse_args(argv)

    logger = get_logger()
    collection: Dict[str, Any] = {}

    if args.skip_collect:
        logger.info("--skip-collect: scoring existing records in %s", args.out_dir)
    else:
        if (
            args.provider != "stub"
            and (args.limit is None or args.limit > args.paid_item_cap)
            and not args.allow_large_paid_run
        ):
            parser.error(
                "refusing a large paid manipulation run: set --limit at or below "
                f"{args.paid_item_cap}, or pass --allow-large-paid-run"
            )
        collection = collect_variants(
            args.items, args.out_dir, args.provider, args.model,
            limit=args.limit, max_tokens=args.max_tokens,
            temperature=args.temperature, seed=args.seed,
            timeout_s=args.timeout, max_retries=args.max_retries, logger=logger,
            max_workers=args.max_workers, resume=args.resume,
            azure_endpoint=args.azure_endpoint,
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
        )
        # The same fail-safe direction as everywhere else: refuse the run rather
        # than emit an artifact that could be read as empirical. Checked after
        # collection so the records survive for inspection.
        if args.strict_real and collection["provenance"] != "real":
            logger.error(
                "--strict-real: provider provenance is '%s'; refusing to write a "
                "manipulation-check report. Records were kept in %s.",
                collection["provenance"], args.out_dir,
            )
            return 2

    joined_path = os.path.join(args.out_dir, "check_records.jsonl")
    join = join_variant_records(args.out_dir, joined_path)
    logger.info(
        "joined %d items (per-variant: %s)", join["n_intersection"], join["n_per_variant"]
    )
    dropped = join["n_dropped_not_in_all_variants"]
    if any(dropped.values()):
        logger.info("dropped, not present in all variants: %s", dropped)

    report = run_checks(joined_path, seed=args.bootstrap_seed)
    report["collection"] = collection
    report["join"] = join
    report_path = os.path.join(args.out_dir, "manipulation_report.json")
    write_json(report, report_path)

    rows: List[Dict[str, Any]] = [
        {
            "role": r["role"],
            "n": r.get("n_items", 0),
            "excl": r.get("n_excluded_absent", 0),
            "operation": r.get("mean_operation", float("nan")),
            "neutral": r.get("mean_neutral", float("nan")),
            "delta": r.get("vs_neutral", {}).get("diff", float("nan")),
            "d": r.get("vs_neutral", {}).get("standardized_effect", float("nan")),
            "p": r.get("p_value", float("nan")),
            "q": r.get("q_value", float("nan")),
            "ci!=0": r.get("vs_neutral", {}).get("interval_excludes_zero", False),
            "uncorr": r.get("passed_uncorrected", False),
            "passed": r.get("passed", False),
        }
        for r in report["results"]
    ]
    print(format_table(rows))
    print(f"\n{report['n_passed']}/{report['n_roles']} roles passed. Report: {report_path}")
    # A3.2: show the correction's effect rather than only its result.
    print(
        f"Rule: Benjamini-Hochberg q <= {report['thresholds']['fdr_q']} across the "
        f"{report['thresholds']['family_size']}-role family AND d >= "
        f"{report['thresholds']['min_standardized_effect']}. Without the "
        f"correction, {report['n_passed_uncorrected']}/{report['n_roles']} "
        "would pass."
    )
    if collection.get("provenance") == "simulated":
        print(
            "PROVENANCE: simulated. These numbers describe the stub provider's "
            "plumbing, not model behaviour, and support no claim about any role."
        )
    return 0 if report["n_passed"] == report["n_roles"] else 1


if __name__ == "__main__":
    sys.exit(main())
