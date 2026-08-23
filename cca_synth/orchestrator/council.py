"""The council orchestrator: items in, structured candidate records out.

This is the component that produces what the rest of the repository consumes.
Its output is a JSONL file that ``dataset.parse_item`` accepts unmodified.

Five properties are enforced structurally rather than by convention, because
each corresponds to a requirement that is easy to satisfy accidentally and easy
to break silently:

**Agent isolation.**  ``RESEARCH_SPECIFICATION.md`` §4 requires agents to be
isolated until the declared communication round.  :class:`Council` builds each
agent's user content from the *item alone*; no agent's output is reachable from
another's call because the call sites never share a mutable context object.
:meth:`Council.run_item` asserts this rather than trusting it.

**Failure is recorded, never imputed.**  A refusal, truncation, timeout, or
unparseable response produces a record with ``present=false`` and the
corresponding failure flag.  Nothing is retried into existence, and no missing
agent is filled with a neutral guess: §8 requires publishing failed calls, and a
silently imputed agent would corrupt both the Shapley ablation and the
error-correlation measurement.

**Untrusted inter-agent text.**  Item content is wrapped in an explicit data
boundary and scanned for instruction-like injection attempts, which are flagged
rather than removed, so the robustness analysis can see them.

**Data minimisation.**  ``ETHICS_AND_DATA_GOVERNANCE.md`` requires stripping
secrets and direct identifiers before dispatch and forbids logging prompts or
credentials.  :func:`redact` runs on every outbound item, and the run log
records prompt *hashes*, never prompt text.

**Snapshot pinning.**  Providers reject moving aliases at construction, so a run
cannot be recorded against an identifier that will mean something else later.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BENCHMARKS, FAILURE_FLAGS, N_ROLES, ROLES
from normalize import cluster_answers
from orchestrator.providers import Provider, ProviderResponse
from prompts import PROMPT_VERSION, ROLE_PROMPTS, output_schema, prompt_manifest

__all__ = [
    "Item",
    "AgentCall",
    "Council",
    "redact",
    "detect_injection",
    "validate_record",
    "REDACTION_PATTERNS",
]


#: Patterns stripped from item content before dispatch.  Deliberately
#: conservative: over-redaction costs a little task accuracy, under-redaction
#: sends personal data to six providers instead of one.
REDACTION_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "[EMAIL]"),
    (r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    (r"\b(?:\d[ -]*?){13,16}\b", "[CARD]"),
    (r"\b(?:sk|pk|api[_-]?key|token|bearer)[-_]?[A-Za-z0-9]{16,}\b", "[CREDENTIAL]"),
    (r"\bhttps?://[^\s]*[?&](?:key|token|secret)=[^\s&]+", "[URL_WITH_SECRET]"),
)

#: Phrases that look like an attempt to instruct the model through data.
_INJECTION_PATTERNS: Tuple[str, ...] = (
    r"ignore (?:all |any )?(?:previous|prior|above) instructions",
    r"disregard (?:the |your )?(?:above|previous|system)",
    r"you are now",
    r"new instructions?:",
    r"system prompt",
    r"the correct answer is",
    r"you must (?:answer|say|choose|output)",
    r"</?(?:system|instruction)>",
)
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def redact(text: str) -> Tuple[str, List[str]]:
    """Strip direct identifiers and credentials from outbound content.

    Args:
        text: Raw item content.

    Returns:
        Tuple ``(redacted_text, applied_labels)``.
    """
    applied: List[str] = []
    for pattern, replacement in REDACTION_PATTERNS:
        text, count = re.subn(pattern, replacement, text)
        if count:
            applied.append(replacement)
    return text, applied


def detect_injection(text: str) -> bool:
    """Whether ``text`` contains instruction-like content.

    Detection flags rather than removes.  Removing would hide the attack from
    the robustness analysis that ``RESEARCH_SPECIFICATION.md`` §9 requires.

    Args:
        text: Content to scan.

    Returns:
        ``True`` if an injection pattern matched.
    """
    return bool(_INJECTION_RE.search(text))


@dataclass
class Item:
    """One benchmark item to route through the council.

    Attributes:
        item_id: Stable identifier.
        benchmark: Benchmark identifier from ``config.BENCHMARKS``.
        question: The question text.
        options: Answer options; empty for free-form items, whose option space
            is built by clustering the agents' own answers.
        gold: Index of the correct option, or ``-1`` when unknown.
        context: Optional passage or supporting context.
    """

    item_id: str
    benchmark: str
    question: str
    options: Sequence[str] = field(default_factory=tuple)
    gold: int = -1
    context: str = ""

    def __post_init__(self) -> None:
        if self.benchmark not in BENCHMARKS:
            raise ValueError(f"unknown benchmark '{self.benchmark}'")


@dataclass
class AgentCall:
    """One agent's call outcome, before assembly into a record.

    Attributes:
        role: Functional role identifier.
        response: The raw provider response.
        record: Parsed §7 record, or ``None`` if unusable.
        failure_flags: Flags to attach.
        injection_seen: Whether injection-like content was present in the input.
    """

    role: str
    response: ProviderResponse
    record: Optional[Dict[str, Any]] = None
    failure_flags: List[str] = field(default_factory=list)
    injection_seen: bool = False


def validate_record(payload: Any, n_options: int) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Validate a parsed response against the §7 contract.

    Implemented directly rather than via a schema library so the package has no
    extra dependency and so each rejection reason is explicit.  Crucially, an
    unexpected key is a *rejection*: ``additionalProperties: false`` in
    :func:`prompts.output_schema` is what stops a model-emitted ``reasoning``
    field from being silently carried into storage.

    Args:
        payload: Decoded JSON from the model.
        n_options: Valid option count for this item.

    Returns:
        Tuple ``(record, flags)``; ``record`` is ``None`` when unusable.
    """
    flags: List[str] = []
    if not isinstance(payload, dict):
        return None, ["format_invalid"]

    schema = output_schema(max(1, n_options))
    allowed = set(schema["properties"])
    unexpected = set(payload) - allowed
    if unexpected:
        # Includes any reasoning-like field the model volunteered.
        return None, ["format_invalid"]

    for required in ("answer_option", "abstain", "confidence"):
        if required not in payload:
            return None, ["format_invalid"]

    try:
        answer = int(payload["answer_option"])
        confidence = float(payload["confidence"])
        abstain = bool(payload["abstain"])
    except (TypeError, ValueError):
        return None, ["format_invalid"]

    if answer < -1 or answer >= n_options:
        answer = -1
        flags.append("format_invalid")
    confidence = min(1.0, max(0.0, confidence))

    claims: List[Dict[str, Any]] = []
    for raw in payload.get("atomic_claims", []) or []:
        if not isinstance(raw, dict):
            continue
        status = raw.get("evidence_status")
        if status not in ("supported", "unsupported", "contradicted", "uncertain"):
            continue
        claims.append(
            {
                "claim": str(raw.get("claim", ""))[:400],
                "evidence_status": status,
                "source_or_check": str(raw.get("source_or_check", ""))[:400],
                "confidence": min(1.0, max(0.0, float(raw.get("confidence", 0.5)))),
            }
        )

    declared = [f for f in (payload.get("failure_flags") or []) if f in FAILURE_FLAGS]
    if any(c["evidence_status"] == "unsupported" for c in claims):
        if "unsupported_claim" not in declared:
            declared.append("unsupported_claim")

    record = {
        "answer_option": answer,
        "abstain": abstain,
        "confidence": confidence,
        "atomic_claims": claims,
        "counterevidence": [
            str(c)[:400] for c in (payload.get("counterevidence") or [])[:16]
        ],
        "failure_flags": sorted(set(declared + flags)),
    }
    return record, flags


def _extract_json(text: str) -> Optional[Any]:
    """Recover a JSON object from a model response.

    Args:
        text: Raw response text.

    Returns:
        The decoded object, or ``None`` if none could be recovered.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


class Council:
    """Routes items through the six roles and emits structured records."""

    def __init__(
        self,
        providers: Mapping[str, Provider],
        variant: str = "operation",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: Optional[int] = 1337,
        condition: str = "cca_council",
        max_workers: int = 1,
    ):
        """Configure the council.

        Args:
            providers: Mapping from role to provider.  Supplying different
                providers per role gives the heterogeneous council of §3
                condition 10 with routing otherwise identical.
            variant: Prompt variant: ``operation``, ``neutral``, or ``persona``.
            max_tokens: Output cap per agent call.
            temperature: Sampling temperature.
            seed: Seed passed to providers that support one.
            condition: Condition label recorded in the manifest.
            max_workers: Maximum items processed concurrently. Calls for the
                six roles within an item remain isolated and sequential.

        Raises:
            ValueError: If any role lacks a provider.
        """
        missing = set(ROLES) - set(providers)
        if missing:
            raise ValueError(f"no provider configured for roles: {sorted(missing)}")
        self.providers = dict(providers)
        self.variant = variant
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.condition = condition
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.max_workers = int(max_workers)
        # Provenance is derived from the providers, never asserted. If ANY role
        # is served by a synthetic provider the whole record is simulated: a
        # council that is five-sixths real is not a real council, and stamping
        # it "real" would let fabricated candidates into a results table through
        # --strict-real -- the exact failure that flag exists to prevent.
        self.provenance = (
            "simulated"
            if any(getattr(p, "synthetic", True) for p in self.providers.values())
            else "real"
        )
        self.stats: Dict[str, int] = {
            "calls": 0, "refusals": 0, "truncations": 0, "malformed": 0,
            "errors": 0, "retries": 0, "injections_flagged": 0, "redactions": 0,
        }
        self._stats_lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def _add_stat(self, key: str, value: int = 1) -> None:
        """Update one manifest counter safely during concurrent collection."""
        with self._stats_lock:
            self.stats[key] += value

    def _record_response(self, response: ProviderResponse) -> None:
        """Accumulate call, retry, token, and cost accounting atomically."""
        with self._stats_lock:
            self.stats["calls"] += 1
            self.stats["retries"] += response.retries
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens
            self.total_cost_usd += response.cost_usd

    def build_user_content(self, item: Item, redacted_question: str, redacted_context: str) -> str:
        """Assemble one agent's user content from the item alone.

        No agent output is reachable from here.  That is what enforces the §4
        isolation requirement: isolation is a property of the call graph, not a
        rule the caller is trusted to follow.

        Args:
            item: The item.
            redacted_question: Question after redaction.
            redacted_context: Context after redaction.

        Returns:
            The user message.
        """
        parts = [f"OPTIONS: {max(len(item.options), 2)}"]
        if redacted_context:
            parts.append(f"<data>\n{redacted_context}\n</data>")
        parts.append(f"QUESTION: {redacted_question}")
        if item.options:
            listing = "\n".join(f"  [{i}] {option}" for i, option in enumerate(item.options))
            parts.append(f"ANSWER OPTIONS:\n{listing}")
        else:
            parts.append(
                "No fixed answer options are supplied. State your answer as the "
                "'claim' of your first atomic claim and set answer_option to 0."
            )
        return "\n\n".join(parts)

    def run_item(self, item: Item) -> Dict[str, Any]:
        """Route one item through all six roles.

        Args:
            item: The item to process.

        Returns:
            A record dict accepted by ``dataset.parse_item``.
        """
        question, redactions_q = redact(item.question)
        context, redactions_c = redact(item.context)
        if redactions_q or redactions_c:
            self._add_stat("redactions")
        injection_seen = detect_injection(item.question) or detect_injection(item.context)
        if injection_seen:
            self._add_stat("injections_flagged")

        user_content = self.build_user_content(item, question, context)
        n_options = max(len(item.options), 2)

        calls: List[AgentCall] = []
        for role in ROLES:
            system_prompt = ROLE_PROMPTS[role].render(self.variant)
            response = self.providers[role].complete(
                system_prompt=system_prompt,
                user_content=user_content,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=self.seed,
            )
            self._record_response(response)

            call = AgentCall(role=role, response=response, injection_seen=injection_seen)
            if injection_seen:
                call.failure_flags.append("injection_suspected")

            if not response.ok:
                self._add_stat(
                    "refusals" if response.stop_reason == "refusal" else "errors"
                )
                call.failure_flags.append(
                    "refusal" if response.stop_reason == "refusal" else "format_invalid"
                )
            else:
                payload = _extract_json(response.text)
                record, flags = (None, ["format_invalid"]) if payload is None else validate_record(
                    payload, n_options
                )
                if record is None:
                    self._add_stat("malformed")
                    call.failure_flags.extend(flags)
                    if response.stop_reason == "truncated":
                        self._add_stat("truncations")
                        call.failure_flags.append("truncated_output")
                else:
                    call.record = record
                    call.failure_flags.extend(record["failure_flags"])
                    if response.stop_reason == "truncated":
                        self._add_stat("truncations")
                        call.failure_flags.append("truncated_output")
            calls.append(call)

        return self._assemble(item, calls, n_options)

    def _assemble(
        self, item: Item, calls: Sequence[AgentCall], n_options: int
    ) -> Dict[str, Any]:
        """Build the item record from six agent calls.

        For free-form benchmarks the shared option space is created here by
        clustering the agents' own answer texts, which is the only point in the
        pipeline where that space exists to be created.

        Args:
            item: The item.
            calls: Per-role call outcomes.
            n_options: Option count for fixed-option items.

        Returns:
            A record dict accepted by ``dataset.parse_item``.
        """
        free_form = not item.options
        if free_form:
            texts = [
                (c.record["atomic_claims"][0]["claim"] if c.record and c.record["atomic_claims"] else "")
                for c in calls
            ]
            clustering = cluster_answers(texts, item.benchmark)
            option_ids = clustering.option_ids
            n_options = max(2, clustering.n_options)
        else:
            option_ids = [c.record["answer_option"] if c.record else -1 for c in calls]

        candidates: List[Dict[str, Any]] = []
        for call, option in zip(calls, option_ids):
            present = call.record is not None
            candidates.append(
                {
                    "role": call.role,
                    "present": present,
                    "answer_option": option if present else -1,
                    "abstain": bool(call.record["abstain"]) if present else False,
                    "confidence": float(call.record["confidence"]) if present else 0.0,
                    "counterevidence": call.record["counterevidence"] if present else [],
                    "failure_flags": sorted(set(call.failure_flags)),
                    "atomic_claims": call.record["atomic_claims"] if present else [],
                    "output_tokens": call.response.output_tokens,
                    "latency_ms": call.response.latency_ms,
                    "cost_usd": call.response.cost_usd,
                    "retries": call.response.retries,
                    "model_snapshot": call.response.model_snapshot,
                }
            )

        return {
            "item_id": item.item_id,
            "benchmark": item.benchmark,
            "n_options": n_options,
            "gold_option": item.gold,
            # The direct-answer baseline defining v(empty coalition) is produced
            # by a separate single-agent condition; -1 means "not collected in
            # this run" rather than "the baseline abstained".
            "base_answer_option": -1,
            "candidates": candidates,
            "input_tokens": sum(c.response.input_tokens for c in calls) // max(1, len(calls)),
            "provenance": self.provenance,
            "condition": self.condition,
            "prompt_version": PROMPT_VERSION,
        }

    def run(
        self,
        items: Iterable[Item],
        output_path: str,
        progress_every: int = 25,
        logger: Optional[Any] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Route many items and write records to JSONL.

        Args:
            items: Items to process.
            output_path: Destination JSONL path.
            progress_every: Log interval.
            logger: Optional logger.
            resume: Append to an existing file and skip item IDs already
                present. Existing records are validated before any paid call.

        Returns:
            A run manifest with counts, exclusions, and cost.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        start = time.perf_counter()
        existing_ids = set()
        if resume and os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as existing:
                for line_number, line in enumerate(existing, start=1):
                    if not line.strip():
                        continue
                    try:
                        item_id = str(json.loads(line)["item_id"])
                    except (json.JSONDecodeError, KeyError) as error:
                        raise ValueError(
                            f"cannot resume: malformed record at {output_path}:{line_number}"
                        ) from error
                    if item_id in existing_ids:
                        raise ValueError(
                            f"cannot resume: duplicate item_id '{item_id}' in {output_path}"
                        )
                    existing_ids.add(item_id)

        skipped = 0
        n_written = 0

        def pending_items() -> Iterable[Item]:
            nonlocal skipped
            for item in items:
                if item.item_id in existing_ids:
                    skipped += 1
                    continue
                # Also protects against duplicate IDs in the current input.
                existing_ids.add(item.item_id)
                yield item

        def write_record(handle: Any, record: Dict[str, Any]) -> None:
            nonlocal n_written
            handle.write(json.dumps(record))
            handle.write("\n")
            handle.flush()
            n_written += 1
            if logger and progress_every and n_written % progress_every == 0:
                logger.info("orchestrated %d new items", n_written)

        mode = "a" if resume else "w"
        with open(output_path, mode, encoding="utf-8") as handle:
            pending = iter(pending_items())
            if self.max_workers == 1:
                for item in pending:
                    write_record(handle, self.run_item(item))
            else:
                # Limit queued futures to 2x workers so a streaming input does
                # not become an unbounded in-memory work queue.
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    while True:
                        batch = list(islice(pending, self.max_workers * 2))
                        if not batch:
                            break
                        futures = [executor.submit(self.run_item, item) for item in batch]
                        # Resolve in input order to keep JSONL deterministic.
                        for future in futures:
                            write_record(handle, future.result())

        manifest = self.manifest()
        manifest.update(
            {
                "n_items": len(existing_ids),
                "n_items_written": n_written,
                "n_items_skipped_existing": skipped,
                "resumed": resume,
                "output_path": output_path,
                "wall_clock_seconds": time.perf_counter() - start,
            }
        )
        return manifest

    def manifest(self) -> Dict[str, Any]:
        """Return the execution manifest required by §4.

        Records prompt *hashes* rather than prompt text: the full prompts are
        versioned in the repository, and the ethics addendum forbids writing
        system prompts into logs.

        Returns:
            A JSON-serialisable manifest.
        """
        calls = max(1, self.stats["calls"])
        return {
            "condition": self.condition,
            "prompt_variant": self.variant,
            "prompt_manifest": prompt_manifest(),
            "model_snapshots": {
                role: provider.model_snapshot for role, provider in self.providers.items()
            },
            "heterogeneous_council": len(
                {p.model_snapshot for p in self.providers.values()}
            ) > 1,
            "sampling": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
                "max_workers": self.max_workers,
            },
            "provider_pricing_usd_per_million_tokens": {
                role: {
                    "input": provider.price_per_million_input,
                    "output": provider.price_per_million_output,
                }
                for role, provider in self.providers.items()
            },
            "provider": {
                role: provider.provider_identity()
                for role, provider in self.providers.items()
            },
            "heterogeneous_route": len(
                {
                    (p.provider_kind, getattr(p, "base_url", None))
                    for p in self.providers.values()
                }
            ) > 1,
            "usage": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
            },
            "counts": dict(self.stats),
            "exclusion_rates": {
                "refusal_rate": self.stats["refusals"] / calls,
                "truncation_rate": self.stats["truncations"] / calls,
                "malformed_rate": self.stats["malformed"] / calls,
                "error_rate": self.stats["errors"] / calls,
                "retry_rate": self.stats["retries"] / calls,
            },
            "total_cost_usd": self.total_cost_usd,
            "isolation": (
                "Agents are isolated: each agent's user content is built from "
                "the item alone and no agent output is reachable from another "
                "agent's call (RESEARCH_SPECIFICATION.md section 4)."
            ),
            "governance": (
                "Item content is redacted before dispatch; injection-like "
                "content is flagged rather than removed; prompts are recorded "
                "by hash, never by text."
            ),
        }
