"""CLI for collecting candidate records.

Reads benchmark items from JSONL, routes them through the council, and writes
§7 candidate records plus a run manifest.  The output is directly consumable::

    python -m orchestrator.collect --items items.jsonl --out records.jsonl \\
        --provider stub --model stub-provider-2026-07-28
    python main.py all --records records.jsonl --strict-real

Item input format, one JSON object per line::

    {"item_id": "...", "benchmark": "math_aime", "question": "...",
     "options": ["a", "b"], "gold": 0, "context": ""}

``--condition`` selects which of the ten conditions in
``RESEARCH_SPECIFICATION.md`` §3 to run.  Conditions differing only in prompt
variant (5, 6, 7) are one flag apart, which is the point: they must differ in
exactly one respect or the comparison is confounded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROLES
from orchestrator.conditions import DirectAnswer
from orchestrator.council import Council, Item
from orchestrator.providers import build_provider
from utils import get_logger, write_json

__all__ = [
    "load_items", "load_item_sources", "main", "CONDITION_VARIANTS",
    "SINGLE_AGENT_CONDITIONS",
]

#: Conditions served by one model with no council, dispatched to a
#: :class:`~orchestrator.conditions.Condition` rather than to
#: :class:`~orchestrator.council.Council`.  ``direct_answer`` is specification
#: section 3 condition 1 and is the declared source of ``base_answer_option``
#: (= ``v(empty)``); until it was wired here there was no way to run it, so the
#: empty coalition silently fell back to "no answer at all".
SINGLE_AGENT_CONDITIONS: Dict[str, type] = {
    "direct_answer": DirectAnswer,
}

#: Prompt variant per condition.  Conditions 5, 6, and 7 are the ones that
#: differ *only* by variant; keeping the mapping explicit here stops a later
#: edit from quietly confounding them.
CONDITION_VARIANTS: Dict[str, str] = {
    "cca_council": "operation",              # condition 5
    "persona_only_council": "persona",       # condition 6
    "neutral_council": "neutral",            # condition 2 control
    "declinicalised_council": "operation",   # condition 7 (prompts carry no clinical language by construction)
    "heterogeneous_council": "operation",    # condition 10
}


def load_items(path: str, limit: Optional[int] = None) -> Iterator[Item]:
    """Stream items from a JSONL file.

    Args:
        path: Source path.
        limit: Optional cap on items yielded.

    Yields:
        :class:`~orchestrator.council.Item` instances.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"items file not found: {path}")
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if limit is not None and count >= limit:
                return
            payload = json.loads(line)
            try:
                yield Item(
                    item_id=str(payload["item_id"]),
                    benchmark=str(payload["benchmark"]),
                    question=str(payload["question"]),
                    options=tuple(payload.get("options", ())),
                    gold=int(payload.get("gold", -1)),
                    context=str(payload.get("context", "")),
                )
            except (KeyError, ValueError) as error:
                raise ValueError(f"malformed item at line {line_number}: {error}") from error
            count += 1


def load_item_sources(
    paths: Sequence[str],
    limit: Optional[int] = None,
    limit_per_source: Optional[int] = None,
) -> Iterator[Item]:
    """Stream one or more benchmarks with explicit global/per-source caps.

    A separate per-source cap prevents a large first dataset from consuming a
    global limit before a smaller registered benchmark contributes any items.
    """
    yielded = 0
    for path in paths:
        for item in load_items(path, limit_per_source):
            if limit is not None and yielded >= limit:
                return
            yield item
            yielded += 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments and run a collection.

    Args:
        argv: Argument vector.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Collect structured candidate records from a model council.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--items", required=True, action="append",
        help="Benchmark items JSONL; repeat for a multi-benchmark run.",
    )
    parser.add_argument("--out", required=True, help="Destination records JSONL.")
    parser.add_argument(
        "--provider", default="stub",
        choices=["stub", "anthropic", "anthropic-foundry", "openai", "azure-openai", "xai"],
        help="Provider for every role, unless --role-provider overrides it.",
    )
    parser.add_argument(
        "--model", default="stub-provider-2026-07-28",
        help="Pinned model snapshot. Moving aliases are rejected.",
    )
    parser.add_argument(
        "--role-provider", action="append", default=[], metavar="ROLE=PROVIDER:MODEL",
        help="Per-role override, repeatable. Use for the heterogeneous council.",
    )
    parser.add_argument(
        "--condition", default="cca_council",
        choices=sorted(set(CONDITION_VARIANTS) | set(SINGLE_AGENT_CONDITIONS)),
        help="Which specification section 3 condition to run.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024, help="Output cap per call.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=1337, help="Seed where supported.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-call timeout, seconds.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per call.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N items.")
    parser.add_argument(
        "--limit-per-source", type=int, default=None,
        help="Process at most N items from each repeated --items source.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=1,
        help="Items processed concurrently; each item still makes six isolated role calls.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Append safely and skip item IDs already present in --out.",
    )
    parser.add_argument(
        "--paid-item-cap", type=int, default=1000,
        help="Refuse a real-provider run above this item limit unless explicitly overridden.",
    )
    parser.add_argument(
        "--allow-large-paid-run", action="store_true",
        help="Acknowledge a real-provider run with no limit or a limit above --paid-item-cap.",
    )
    parser.add_argument(
        "--foundry-resource", default=None,
        help="Foundry resource name, for a model provisioned onto the resource.",
    )
    parser.add_argument(
        "--foundry-base-url", default=None,
        help=("Endpoint of a serverless Foundry deployment, which has its own "
              "URL that the resource name cannot reach."),
    )
    parser.add_argument(
        "--azure-endpoint", default=None,
        help="Azure resource endpoint; defaults to AZURE_OPENAI_ENDPOINT.",
    )
    parser.add_argument(
        "--input-price-per-million", type=float, default=None,
        help="Azure run-time input-token price for cost accounting.",
    )
    parser.add_argument(
        "--output-price-per-million", type=float, default=None,
        help="Azure run-time output-token price for cost accounting.",
    )
    args = parser.parse_args(argv)

    logger = get_logger()

    def make_provider(kind: str, model: str):
        kwargs: Dict[str, Any] = {
            "timeout_s": args.timeout,
            "max_retries": args.max_retries,
        }
        if kind == "azure-openai":
            kwargs.update(
                endpoint=args.azure_endpoint,
                input_price_per_million=args.input_price_per_million,
                output_price_per_million=args.output_price_per_million,
            )
        elif kind == "xai":
            kwargs.update(
                input_price_per_million=args.input_price_per_million,
                output_price_per_million=args.output_price_per_million,
            )
        elif kind == "anthropic-foundry":
            kwargs.update(
                resource=args.foundry_resource,
                base_url=args.foundry_base_url,
                input_price_per_million=args.input_price_per_million,
                output_price_per_million=args.output_price_per_million,
            )
        return build_provider(kind, model, **kwargs)

    default = make_provider(args.provider, args.model)
    providers = {role: default for role in ROLES}
    for override in args.role_provider:
        if "=" not in override or ":" not in override:
            parser.error(f"malformed --role-provider '{override}'; expected ROLE=PROVIDER:MODEL")
        role, spec = override.split("=", 1)
        kind, model = spec.split(":", 1)
        if role not in ROLES:
            parser.error(f"unknown role '{role}'; expected one of {sorted(ROLES)}")
        providers[role] = make_provider(kind, model)

    single_agent = args.condition in SINGLE_AGENT_CONDITIONS
    if single_agent:
        # One model, no council: --role-provider does not apply, and silently
        # ignoring it would let a caller believe a per-role route was honoured.
        if args.role_provider:
            parser.error(
                f"--role-provider does not apply to condition '{args.condition}': "
                "it is served by a single model"
            )
        arm = SINGLE_AGENT_CONDITIONS[args.condition](
            provider=default,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            max_workers=args.max_workers,
        )
        provenance = "simulated" if getattr(default, "synthetic", True) else "real"
    else:
        arm = Council(
            providers=providers,
            variant=CONDITION_VARIANTS[args.condition],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            condition=args.condition,
            max_workers=args.max_workers,
        )
        provenance = arm.provenance

    requested_items = args.limit
    if requested_items is None and args.limit_per_source is not None:
        requested_items = args.limit_per_source * len(args.items)
    if (
        provenance == "real"
        and (requested_items is None or requested_items > args.paid_item_cap)
        and not args.allow_large_paid_run
    ):
        parser.error(
            "refusing a large paid run: set --limit at or below "
            f"{args.paid_item_cap}, or pass --allow-large-paid-run"
        )

    logger.info(
        "collecting condition=%s variant=%s heterogeneous=%s",
        args.condition,
        CONDITION_VARIANTS.get(args.condition, "n/a (single agent)"),
        "n/a" if single_agent else arm.manifest()["heterogeneous_council"],
    )

    items = load_item_sources(args.items, args.limit, args.limit_per_source)
    if single_agent:
        manifest = arm.run(items, args.out, resume=args.resume)
    else:
        manifest = arm.run(items, args.out, logger=logger, resume=args.resume)
    manifest_path = os.path.splitext(args.out)[0] + ".manifest.json"
    write_json(manifest, manifest_path)

    logger.info(
        "output now has %d records (%d new) at %s (manifest: %s)",
        manifest["n_items"], manifest["n_items_written"], args.out, manifest_path,
    )
    rates = manifest["exclusion_rates"]
    logger.info(
        "exclusions | refusal %.3f truncation %.3f malformed %.3f error %.3f retries %.3f",
        rates["refusal_rate"], rates["truncation_rate"], rates["malformed_rate"],
        rates["error_rate"], rates["retry_rate"],
    )
    if manifest["counts"]["injections_flagged"]:
        logger.info(
            "%d items carried injection-like content (flagged, not removed)",
            manifest["counts"]["injections_flagged"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
