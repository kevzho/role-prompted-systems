"""Model providers: the boundary between this study and an external API.

Five implementations:

:class:`StubProvider`
    Deterministic, offline, no network.  Exists so the full pipeline is testable
    and so CI can run without credentials or spend.

:class:`AnthropicProvider` / :class:`OpenAIProvider` /
:class:`AzureOpenAIProvider` / :class:`AnthropicFoundryProvider`
    Thin adapters over the real SDKs, imported lazily so the package works
    without them installed.  The two platform adapters extend their first-party
    counterparts and override only client construction; note that Azure OpenAI
    and Foundry-served Anthropic are *different* API surfaces, so they are not
    interchangeable.

Every provider returns a :class:`ProviderResponse` carrying the fields
``RESEARCH_SPECIFICATION.md`` §4 requires in the execution manifest: the exact
model snapshot, token counts, latency, retries, and the reason generation
stopped.  A refusal or truncation is a *recorded outcome*, never an exception
that a caller might swallow into a silent imputation.

Snapshot pinning is enforced, not requested.  §4 says "Do not use a moving alias
as the only identifier", so :meth:`Provider.resolve_snapshot` must return a
pinned identifier and construction fails when only an alias is supplied.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ProviderResponse",
    "Provider",
    "StubProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "AnthropicFoundryProvider",
    "build_provider",
    "MOVING_ALIASES",
]

#: Identifiers that name a moving target rather than a frozen snapshot.  §4
#: forbids using one as the sole identifier for a run.
MOVING_ALIASES: Tuple[str, ...] = ("latest", "newest", "current", "stable", "default")


@dataclass
class ProviderResponse:
    """One model call's outcome.

    Attributes:
        text: Raw response text; empty on refusal or hard failure.
        ok: Whether a usable response was produced.
        stop_reason: ``"complete"``, ``"truncated"``, ``"refusal"``,
            ``"timeout"``, or ``"error"``.
        input_tokens: Prompt tokens billed.
        output_tokens: Generated tokens billed.
        latency_ms: Wall-clock latency of the final attempt.
        retries: Retries before this outcome.
        cost_usd: Estimated cost.
        model_snapshot: Exact pinned model identifier.
        error: Short error description; never contains prompt content or
            credentials, since it is written to the run log.
    """

    text: str = ""
    ok: bool = True
    stop_reason: str = "complete"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    cost_usd: float = 0.0
    model_snapshot: str = ""
    error: str = ""


class Provider:
    """Base class for model providers."""

    #: Registry key naming the serving route, recorded in the execution
    #: manifest.  Spec section 4 requires the route to be derivable from the
    #: artifacts rather than asserted in prose: without it a record attests
    #: that *a* real provider served it but not *which*, and a run that
    #: silently changed route would be undetectable after the fact.
    provider_kind: str = "unknown"

    #: Per-million-token prices, recorded at run time per limitation L18.
    price_per_million_input: float = 0.0
    price_per_million_output: float = 0.0

    def provider_identity(self) -> Dict[str, Any]:
        """Describe the serving route for the execution manifest.

        Never includes credentials: only the registry key, the resolved
        route, and the model snapshot as sent.

        Adapters name their route differently -- an OpenAI-compatible adapter
        resolves a ``base_url``, while the Foundry adapter is configured with a
        bare resource name and lets the SDK derive the URL. Both are recorded,
        so a Foundry manifest says where the calls went instead of carrying a
        null endpoint and no substitute.

        Returns:
            The route description written into the manifest.
        """
        return {
            "kind": self.provider_kind,
            "endpoint": getattr(self, "base_url", None),
            "resource": getattr(self, "resource", None),
            "model_snapshot": self.model_snapshot,
            "synthetic": self.synthetic,
        }

    #: Whether this provider fabricates responses rather than calling a model.
    #:
    #: Drives the ``provenance`` stamp on every emitted record, and through it
    #: ``--strict-real``.  It defaults to ``True`` deliberately: a provider that
    #: forgets to declare itself gets its output stamped *simulated*, which
    #: costs a refused run, whereas the opposite default would silently let
    #: fabricated records into a results table.  Only a provider that genuinely
    #: calls a model may set it ``False``.
    synthetic: bool = True

    def __init__(self, model_snapshot: str, timeout_s: float = 120.0, max_retries: int = 2):
        """Initialise the provider.

        Args:
            model_snapshot: Exact pinned model identifier.
            timeout_s: Per-call timeout.
            max_retries: Retries on transient failure.

        Raises:
            ValueError: If the identifier is a moving alias.
        """
        self.model_snapshot = self.resolve_snapshot(model_snapshot)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self._client_lock = threading.Lock()

    @staticmethod
    def resolve_snapshot(model: str) -> str:
        """Validate that ``model`` is a pinned snapshot.

        Args:
            model: Model identifier.

        Returns:
            The identifier, unchanged.

        Raises:
            ValueError: If it names a moving alias.
        """
        lowered = model.lower().strip()
        if not lowered:
            raise ValueError("model snapshot must not be empty")
        for alias in MOVING_ALIASES:
            if lowered.endswith(f"-{alias}") or lowered == alias:
                raise ValueError(
                    f"'{model}' is a moving alias. RESEARCH_SPECIFICATION.md "
                    "section 4 requires a frozen snapshot identifier, because a "
                    "run that cannot name its model cannot be reproduced."
                )
        return model

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate the dollar cost of one call.

        Args:
            input_tokens: Prompt tokens.
            output_tokens: Generated tokens.

        Returns:
            Cost in USD.
        """
        return (
            input_tokens * self.price_per_million_input
            + output_tokens * self.price_per_million_output
        ) / 1e6

    def complete(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> ProviderResponse:
        """Produce one completion.

        Args:
            system_prompt: The role's system prompt.
            user_content: The item content.
            max_tokens: Output cap.
            temperature: Sampling temperature.
            seed: Seed where the provider supports it.

        Returns:
            A :class:`ProviderResponse`.

        Raises:
            NotImplementedError: On the base class.
        """
        raise NotImplementedError


class StubProvider(Provider):
    """Deterministic offline provider for tests and pipeline validation.

    Produces schema-valid §7 records whose content is a reproducible function of
    the prompt and item, and injects a controlled rate of refusals, truncations,
    and malformed JSON so the orchestrator's failure handling is genuinely
    exercised rather than assumed.

    It is **not** a model simulator. Its answers carry no signal about task
    difficulty; anything trained on them measures plumbing, not behaviour.
    """

    provider_kind = "stub"

    price_per_million_input = 0.0
    price_per_million_output = 0.0

    def __init__(
        self,
        model_snapshot: str = "stub-provider-2026-07-28",
        timeout_s: float = 120.0,
        max_retries: int = 2,
        failure_rate: float = 0.03,
        malformed_rate: float = 0.02,
        truncation_rate: float = 0.02,
        seed: int = 0,
    ):
        """Initialise the stub.

        Args:
            model_snapshot: Pinned identifier for traceability.
            timeout_s: Ignored; present for interface parity.
            max_retries: Retries to report.
            failure_rate: Fraction of calls that refuse.
            malformed_rate: Fraction that emit unparseable JSON.
            truncation_rate: Fraction that truncate mid-object.
            seed: Base seed; outcomes are a deterministic function of it.
        """
        super().__init__(model_snapshot, timeout_s, max_retries)
        self.failure_rate = failure_rate
        self.malformed_rate = malformed_rate
        self.truncation_rate = truncation_rate
        self.seed = seed

    def _rng(self, key: str) -> random.Random:
        """Return a Random seeded deterministically from ``key``.

        Args:
            key: Text whose hash seeds the generator.

        Returns:
            A seeded :class:`random.Random`.
        """
        digest = hashlib.sha256(f"{self.seed}::{key}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def complete(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> ProviderResponse:
        """Produce a deterministic pseudo-response.

        Args:
            system_prompt: Role prompt; its hash steers the stub's answer so
                different roles disagree.
            user_content: Item content.
            max_tokens: Output cap, used to decide truncation.
            temperature: Ignored.
            seed: Additional seed material.

        Returns:
            A :class:`ProviderResponse`.
        """
        key = f"{system_prompt[:64]}::{user_content}::{seed}"
        rng = self._rng(key)
        start = time.perf_counter()

        n_options = 4
        match = re.search(r"OPTIONS:\s*(\d+)", user_content)
        if match:
            n_options = max(2, int(match.group(1)))

        if rng.random() < self.failure_rate:
            return ProviderResponse(
                ok=False, stop_reason="refusal", model_snapshot=self.model_snapshot,
                latency_ms=(time.perf_counter() - start) * 1000, error="stub refusal",
            )

        n_claims = rng.randint(1, 4)
        statuses = ["supported", "unsupported", "contradicted", "uncertain"]
        record = {
            "answer_option": rng.randrange(n_options),
            "abstain": rng.random() < 0.05,
            "confidence": round(rng.uniform(0.3, 0.95), 3),
            "atomic_claims": [
                {
                    "claim": f"stub claim {i}",
                    "evidence_status": rng.choice(statuses),
                    "source_or_check": "stub_check" if rng.random() < 0.5 else "",
                    "confidence": round(rng.uniform(0.2, 0.95), 3),
                }
                for i in range(n_claims)
            ],
            "counterevidence": [f"stub counterevidence {i}" for i in range(rng.randint(0, 3))],
            "failure_flags": [],
        }
        text = json.dumps(record)

        if rng.random() < self.malformed_rate:
            text = text[: len(text) // 2] + "  <<not json>>"
            stop_reason = "complete"
        elif rng.random() < self.truncation_rate:
            text = text[: len(text) // 2]
            stop_reason = "truncated"
        else:
            stop_reason = "complete"

        output_tokens = max(1, len(text) // 4)
        return ProviderResponse(
            text=text,
            ok=True,
            stop_reason=stop_reason,
            input_tokens=max(1, (len(system_prompt) + len(user_content)) // 4),
            output_tokens=min(output_tokens, max_tokens),
            latency_ms=(time.perf_counter() - start) * 1000,
            retries=0,
            cost_usd=0.0,
            model_snapshot=self.model_snapshot,
        )


class AnthropicProvider(Provider):
    """Adapter over the Anthropic Messages API.

    The SDK is imported lazily so this module loads without it.

    Prices are list prices for the default pinned snapshot and must be updated
    alongside it.  Prompt caching and the Batch API are **not** implemented, so
    realised cost is list price -- roughly double the "effective per call"
    column in ``logistics/HANDOVER.md`` section 2a, which assumes both.  At the
    scoped study's ~42,000 calls that is the difference between ~$85 and ~$170,
    which the available credit covers either way; it would not be immaterial at
    the full run's 513,000 calls.
    """

    provider_kind = "anthropic"

    price_per_million_input = 3.0
    price_per_million_output = 15.0
    synthetic = False

    #: Substring identifying a recoverable sampling-parameter rejection.
    _TEMPERATURE_REJECTED = ("temperature",)

    def __init__(self, model_snapshot: str, timeout_s: float = 120.0, max_retries: int = 2):
        """Initialise the client.

        Args:
            model_snapshot: Pinned model identifier.
            timeout_s: Per-call timeout.
            max_retries: Retries on transient failure.

        Raises:
            ImportError: If the ``anthropic`` package is unavailable.
        """
        super().__init__(model_snapshot, timeout_s, max_retries)
        try:
            import anthropic  # noqa: F401
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "AnthropicProvider requires the 'anthropic' package: pip install anthropic"
            ) from error
        self._client = None
        # Latched across calls, not per call: a model that rejects a non-default
        # temperature rejects it on every item, so a per-call flag would pay one
        # wasted round trip per item -- 12,600 of them on a full arm. Written
        # without a lock because the only transition is True -> False and a
        # concurrent worker that misses it merely repeats the discovery once.
        self._send_temperature = True

    def _get_client(self) -> Any:  # pragma: no cover - requires credentials
        """Return a lazily constructed SDK client."""
        if self._client is None:
            with self._client_lock:
                if self._client is not None:
                    return self._client
                import anthropic

                self._client = anthropic.Anthropic(timeout=self.timeout_s)
        return self._client

    def complete(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> ProviderResponse:  # pragma: no cover - requires credentials
        """Call the Messages API once, with retries.

        Args:
            system_prompt: Role prompt.
            user_content: Item content.
            max_tokens: Output cap.
            temperature: Sampling temperature.
            seed: Unused; the API does not expose a seed.

        Returns:
            A :class:`ProviderResponse`.
        """
        del seed
        client = self._get_client()
        retries = 0
        start = time.perf_counter()
        while True:
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model_snapshot,
                    "max_tokens": max_tokens,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_content}],
                }
                if self._send_temperature:
                    kwargs["temperature"] = temperature
                message = client.messages.create(**kwargs)
                text = "".join(
                    block.text for block in message.content if getattr(block, "type", "") == "text"
                )
                stop_reason = (
                    "truncated" if message.stop_reason == "max_tokens" else "complete"
                )
                usage_in = getattr(message.usage, "input_tokens", 0)
                usage_out = getattr(message.usage, "output_tokens", 0)
                return ProviderResponse(
                    text=text,
                    ok=bool(text),
                    stop_reason=stop_reason if text else "refusal",
                    input_tokens=usage_in,
                    output_tokens=usage_out,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    retries=retries,
                    cost_usd=self.estimate_cost(usage_in, usage_out),
                    model_snapshot=self.model_snapshot,
                )
            except Exception as error:
                # Newer Claude models reject a non-default temperature outright
                # rather than clamping it. That is a deterministic parameter
                # incompatibility, not a transient fault: retrying it unchanged
                # burns the retry budget and then reports "error" for every
                # call in the run. Drop the parameter and retry immediately
                # without counting it, once, so a genuinely broken request still
                # terminates. Detected from the error rather than from a model
                # allowlist, which would go stale silently -- the same reasoning
                # OpenAIProvider documents for its two rejections.
                message_text = str(error).lower()
                if self._send_temperature and all(
                    token in message_text for token in self._TEMPERATURE_REJECTED
                ):
                    self._send_temperature = False
                    continue
                retries += 1
                if retries > self.max_retries:
                    return ProviderResponse(
                        ok=False,
                        stop_reason="error",
                        retries=retries,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        model_snapshot=self.model_snapshot,
                        # Only the exception type: the message can echo prompt
                        # content, and this string reaches the run log.
                        error=type(error).__name__,
                    )
                time.sleep(min(2**retries, 30))


class OpenAIProvider(Provider):
    """Adapter over the OpenAI chat completions API.

    Two parameter incompatibilities are handled here rather than left to fail at
    the first paid call.  Newer OpenAI models reject ``max_tokens`` in favour of
    ``max_completion_tokens``, and reasoning models reject any ``temperature``
    other than the default.  Both are detected from the error and retried once
    with the offending parameter adjusted, because the alternative is a
    hard-coded model-name allowlist that goes stale silently.

    Prices are provider list prices and must be updated when the pinned snapshot
    changes; :meth:`Provider.estimate_cost` has no way to know they are wrong.
    Neither this adapter nor :class:`AnthropicProvider` implements prompt
    caching or the Batch API, so realised cost is list price rather than the
    discounted figure in ``logistics/HANDOVER.md`` section 2a.
    """

    provider_kind = "openai"

    price_per_million_input = 2.5
    price_per_million_output = 10.0
    synthetic = False

    #: Substrings identifying the two recoverable parameter rejections.
    _MAX_TOKENS_REJECTED = ("max_tokens", "max_completion_tokens")
    _TEMPERATURE_REJECTED = ("temperature",)

    def __init__(self, model_snapshot: str, timeout_s: float = 120.0, max_retries: int = 2):
        """Initialise the client.

        Args:
            model_snapshot: Pinned model identifier.
            timeout_s: Per-call timeout.
            max_retries: Retries on transient failure.

        Raises:
            ImportError: If the ``openai`` package is unavailable.
        """
        super().__init__(model_snapshot, timeout_s, max_retries)
        try:
            import openai  # noqa: F401
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "OpenAIProvider requires the 'openai' package: pip install openai"
            ) from error
        self._client = None
        # Latched across calls, not per call. Both rejections are properties of
        # the deployment, not of the request, so a model that rejects on the
        # first item rejects on every one. Held per call, the discovery is
        # thrown away and re-learned 12,600 times on a full arm -- one wasted
        # round trip each. Written without a lock because each transition is
        # one-way and a racing worker at worst repeats the discovery once.
        self._token_param = "max_tokens"
        self._send_temperature = True

    def _get_client(self) -> Any:  # pragma: no cover - requires credentials
        """Return a lazily constructed SDK client."""
        if self._client is None:
            with self._client_lock:
                if self._client is not None:
                    return self._client
                import openai

                self._client = openai.OpenAI(timeout=self.timeout_s)
        return self._client

    def complete(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> ProviderResponse:  # pragma: no cover - requires credentials
        """Call the chat completions API once, with retries.

        Args:
            system_prompt: Role prompt.
            user_content: Item content.
            max_tokens: Output cap.
            temperature: Sampling temperature.
            seed: Seed, which this API does support.

        Returns:
            A :class:`ProviderResponse`.
        """
        client = self._get_client()
        retries = 0
        start = time.perf_counter()
        while True:
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model_snapshot,
                    self._token_param: max_tokens,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                }
                if self._send_temperature:
                    kwargs["temperature"] = temperature
                if seed is not None:
                    kwargs["seed"] = seed
                completion = client.chat.completions.create(**kwargs)
                choice = completion.choices[0]
                text = choice.message.content or ""
                stop_reason = "truncated" if choice.finish_reason == "length" else "complete"
                usage_in = getattr(completion.usage, "prompt_tokens", 0)
                usage_out = getattr(completion.usage, "completion_tokens", 0)
                return ProviderResponse(
                    text=text,
                    ok=bool(text),
                    stop_reason=stop_reason if text else "refusal",
                    input_tokens=usage_in,
                    output_tokens=usage_out,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    retries=retries,
                    cost_usd=self.estimate_cost(usage_in, usage_out),
                    model_snapshot=self.model_snapshot,
                )
            except Exception as error:
                # Two rejections are deterministic parameter incompatibilities
                # rather than transient faults: retrying them unchanged burns
                # the retry budget and then reports "error" for every call. Fix
                # the parameter and retry immediately without counting it, but
                # only once each, so a genuinely broken request still
                # terminates.
                message = str(error).lower()
                if (
                    self._token_param == "max_tokens"
                    and all(token in message for token in self._MAX_TOKENS_REJECTED)
                ):
                    self._token_param = "max_completion_tokens"
                    continue
                if self._send_temperature and all(
                    token in message for token in self._TEMPERATURE_REJECTED
                ):
                    self._send_temperature = False
                    continue
                retries += 1
                if retries > self.max_retries:
                    return ProviderResponse(
                        ok=False,
                        stop_reason="error",
                        retries=retries,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        model_snapshot=self.model_snapshot,
                        error=type(error).__name__,
                    )
                time.sleep(min(2**retries, 30))


class AzureOpenAIProvider(OpenAIProvider):
    """OpenAI-compatible adapter for Azure OpenAI's versioned ``/openai/v1`` API.

    Azure's ``model`` argument is the deployment name.  For reproducibility the
    deployment should itself be named for the pinned model snapshot; moving
    names such as ``latest`` are rejected by :class:`Provider`.

    Azure prices vary by model, region, and provisioning mode, so this adapter
    never guesses them.  Pass the run-time prices explicitly (or set the two
    ``AZURE_OPENAI_*_PRICE_PER_MILLION`` environment variables) and they will be
    recorded in the execution manifest.
    """

    provider_kind = "azure-openai"

    def __init__(
        self,
        model_snapshot: str,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        input_price_per_million: Optional[float] = None,
        output_price_per_million: Optional[float] = None,
    ):
        """Configure an Azure OpenAI deployment without logging credentials."""
        super().__init__(model_snapshot, timeout_s, max_retries)
        raw_endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        if not raw_endpoint.strip():
            raise ValueError(
                "Azure OpenAI requires --azure-endpoint or AZURE_OPENAI_ENDPOINT"
            )
        base_url = raw_endpoint.rstrip("/")
        if not base_url.endswith("/openai/v1"):
            base_url += "/openai/v1"
        self.base_url = base_url + "/"
        self._api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")

        env_input = os.getenv("AZURE_OPENAI_INPUT_PRICE_PER_MILLION")
        env_output = os.getenv("AZURE_OPENAI_OUTPUT_PRICE_PER_MILLION")
        if input_price_per_million is None and env_input is None:
            raise ValueError(
                "Azure OpenAI requires --input-price-per-million or "
                "AZURE_OPENAI_INPUT_PRICE_PER_MILLION for auditable cost accounting"
            )
        if output_price_per_million is None and env_output is None:
            raise ValueError(
                "Azure OpenAI requires --output-price-per-million or "
                "AZURE_OPENAI_OUTPUT_PRICE_PER_MILLION for auditable cost accounting"
            )
        self.price_per_million_input = float(
            input_price_per_million
            if input_price_per_million is not None
            else env_input
        )
        self.price_per_million_output = float(
            output_price_per_million
            if output_price_per_million is not None
            else env_output
        )

    def _get_client(self) -> Any:  # pragma: no cover - requires credentials
        """Return an OpenAI client pointed at the Azure v1 base URL."""
        if self._client is None:
            with self._client_lock:
                if self._client is not None:
                    return self._client
                if not self._api_key:
                    raise ValueError(
                        "Azure OpenAI requires AZURE_OPENAI_API_KEY or an explicit api_key"
                    )
                import openai

                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url=self.base_url,
                    timeout=self.timeout_s,
                )
        return self._client


class XAIProvider(OpenAIProvider):
    """OpenAI-compatible adapter for xAI's Grok models.

    xAI serves the chat-completions shape, so this inherits
    :meth:`OpenAIProvider.complete` unchanged -- including its two adaptive
    parameter rejections -- and only client construction differs. That works
    here because nothing in this pipeline depends on provider-side structured
    output: prompts ask for JSON and the reply is parsed downstream, so a
    second family needs no ``response_format`` or tool-calling support.

    The point of this adapter is the cross-family replication the paper cannot
    otherwise make: a different pretraining lineage at a comparable capability
    tier, so a null is not confounded with a weaker second model.

    Prices are never guessed. xAI's rates are not derivable from the model name,
    so pass them explicitly (or set the two ``XAI_*_PRICE_PER_MILLION``
    variables) and they are recorded in the execution manifest.
    """

    provider_kind = "xai"

    #: xAI's OpenAI-compatible base URL.
    DEFAULT_BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        model_snapshot: str,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        input_price_per_million: Optional[float] = None,
        output_price_per_million: Optional[float] = None,
    ):
        """Configure an xAI client without logging credentials.

        Args:
            model_snapshot: Pinned model identifier, e.g. the Grok version.
            timeout_s: Per-call timeout.
            max_retries: Retries on transient failure.
            base_url: Override for the API root.
            api_key: Key; defaults to ``XAI_API_KEY``.
            input_price_per_million: Input price; defaults to
                ``XAI_INPUT_PRICE_PER_MILLION``.
            output_price_per_million: Output price; defaults to
                ``XAI_OUTPUT_PRICE_PER_MILLION``.

        Raises:
            ValueError: If either price is missing.
        """
        super().__init__(model_snapshot, timeout_s, max_retries)
        self.base_url = (base_url or os.getenv("XAI_BASE_URL", "")).strip().rstrip(
            "/"
        ) or self.DEFAULT_BASE_URL
        self._api_key = api_key or os.getenv("XAI_API_KEY")

        env_input = os.getenv("XAI_INPUT_PRICE_PER_MILLION")
        env_output = os.getenv("XAI_OUTPUT_PRICE_PER_MILLION")
        if input_price_per_million is None and env_input is None:
            raise ValueError(
                "xAI requires --input-price-per-million or "
                "XAI_INPUT_PRICE_PER_MILLION for auditable cost accounting"
            )
        if output_price_per_million is None and env_output is None:
            raise ValueError(
                "xAI requires --output-price-per-million or "
                "XAI_OUTPUT_PRICE_PER_MILLION for auditable cost accounting"
            )
        self.price_per_million_input = float(
            input_price_per_million
            if input_price_per_million is not None
            else env_input
        )
        self.price_per_million_output = float(
            output_price_per_million
            if output_price_per_million is not None
            else env_output
        )

    def _get_client(self) -> Any:  # pragma: no cover - requires credentials
        """Return an OpenAI client pointed at the xAI base URL."""
        if self._client is None:
            with self._client_lock:
                if self._client is not None:
                    return self._client
                if not self._api_key:
                    raise ValueError(
                        "xAI requires XAI_API_KEY or an explicit api_key"
                    )
                import openai

                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url=self.base_url,
                    timeout=self.timeout_s,
                )
        return self._client


class AnthropicFoundryProvider(AnthropicProvider):
    """Anthropic models served through Microsoft Foundry.

    Mirrors how :class:`AzureOpenAIProvider` extends :class:`OpenAIProvider`:
    the request shape, retry policy, cost accounting, and provenance stamp are
    inherited unchanged, and only client construction differs. Foundry serves
    the Messages API, so :meth:`AnthropicProvider.complete` applies as written --
    this is *not* the OpenAI-compatible surface :class:`AzureOpenAIProvider`
    targets, and pointing that adapter at a Claude deployment will not work.

    As with Azure, the deployment name is the model identifier of record, so it
    should name the pinned snapshot rather than a role ("claude-sonnet-4-6-v1",
    not "claude-prod"). Anthropic publishes no dated snapshot for current
    models, so deployment name plus Foundry version is the strongest identifier
    available and `ETHICS_COMPLIANCE.md` §5 records it as such.

    Prices are never guessed, for the same reason they are not guessed on Azure:
    Foundry bills through the Microsoft Marketplace and the effective rate
    depends on region and agreement. ``budget.matched_budgets`` derives the
    cost-matched arm's cap from these numbers, so a wrong price silently
    mis-sizes an experimental condition rather than merely misreporting one.

    .. warning::
        Requires ``pip install anthropic``, which is **not** in
        ``requirements.txt`` and is not installed in the environment this class
        was written in. The constructor call below follows the documented
        ``AnthropicFoundry(api_key=..., resource=...)`` form; verify it against
        the installed SDK before the first paid run, and add the resolved
        version to ``requirements.txt`` so the §4 execution manifest records it.
    """

    provider_kind = "anthropic-foundry"

    def __init__(
        self,
        model_snapshot: str,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        resource: Optional[str] = None,
        api_key: Optional[str] = None,
        input_price_per_million: Optional[float] = None,
        output_price_per_million: Optional[float] = None,
        base_url: Optional[str] = None,
    ):
        """Configure a Foundry-served Anthropic deployment.

        Args:
            model_snapshot: Foundry deployment name, which is the identifier of
                record for the run.
            timeout_s: Per-call timeout.
            max_retries: Retries on transient failure.
            resource: Foundry resource name; defaults to
                ``ANTHROPIC_FOUNDRY_RESOURCE``.
            api_key: Defaults to ``ANTHROPIC_FOUNDRY_API_KEY``.
            input_price_per_million: Input price; defaults to
                ``ANTHROPIC_FOUNDRY_INPUT_PRICE_PER_MILLION``.
            output_price_per_million: Output price; defaults to
                ``ANTHROPIC_FOUNDRY_OUTPUT_PRICE_PER_MILLION``.

        Raises:
            ValueError: If the resource or either price is missing.
        """
        super().__init__(model_snapshot, timeout_s, max_retries)
        # Foundry serves Claude two ways and they are configured differently.
        # A model provisioned onto the resource is addressed by resource name
        # and the SDK derives the URL; a serverless (Marketplace) deployment
        # gets its own endpoint, which no resource name can reach. Accept
        # either, require one, and prefer the explicit URL when both are given.
        self.resource = (resource or os.getenv("ANTHROPIC_FOUNDRY_RESOURCE", "")).strip() or None
        raw_base_url = (base_url or os.getenv("ANTHROPIC_FOUNDRY_BASE_URL", "")).strip()
        self.base_url = raw_base_url.rstrip("/") or None
        if not self.resource and not self.base_url:
            raise ValueError(
                "Foundry requires a resource name (--foundry-resource or "
                "ANTHROPIC_FOUNDRY_RESOURCE) or a deployment endpoint "
                "(--foundry-base-url or ANTHROPIC_FOUNDRY_BASE_URL). A "
                "serverless deployment has its own endpoint and the resource "
                "name will not reach it."
            )
        self._api_key = api_key or os.getenv("ANTHROPIC_FOUNDRY_API_KEY")

        env_input = os.getenv("ANTHROPIC_FOUNDRY_INPUT_PRICE_PER_MILLION")
        env_output = os.getenv("ANTHROPIC_FOUNDRY_OUTPUT_PRICE_PER_MILLION")
        if input_price_per_million is None and env_input is None:
            raise ValueError(
                "Foundry requires --input-price-per-million or "
                "ANTHROPIC_FOUNDRY_INPUT_PRICE_PER_MILLION for auditable cost "
                "accounting"
            )
        if output_price_per_million is None and env_output is None:
            raise ValueError(
                "Foundry requires --output-price-per-million or "
                "ANTHROPIC_FOUNDRY_OUTPUT_PRICE_PER_MILLION for auditable cost "
                "accounting"
            )
        self.price_per_million_input = float(
            input_price_per_million if input_price_per_million is not None else env_input
        )
        self.price_per_million_output = float(
            output_price_per_million if output_price_per_million is not None else env_output
        )

    def _get_client(self) -> Any:  # pragma: no cover - requires credentials
        """Return a lazily constructed Foundry client."""
        if self._client is None:
            with self._client_lock:
                if self._client is not None:
                    return self._client
                if not self._api_key:
                    raise ValueError(
                        "Foundry requires ANTHROPIC_FOUNDRY_API_KEY or an "
                        "explicit api_key"
                    )
                from anthropic import AnthropicFoundry

                client_kwargs: Dict[str, Any] = {
                    "api_key": self._api_key,
                    "timeout": self.timeout_s,
                }
                # An explicit endpoint wins: it addresses one deployment, while
                # the resource name is only a template the SDK expands.
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                else:
                    client_kwargs["resource"] = self.resource
                self._client = AnthropicFoundry(**client_kwargs)
        return self._client


#: Registry consulted by :func:`build_provider`.
_PROVIDERS: Dict[str, type] = {
    "stub": StubProvider,
    "anthropic": AnthropicProvider,
    "anthropic-foundry": AnthropicFoundryProvider,
    "openai": OpenAIProvider,
    "azure-openai": AzureOpenAIProvider,
    "xai": XAIProvider,
}


def build_provider(kind: str, model_snapshot: str, **kwargs: Any) -> Provider:
    """Construct a provider by name.

    Args:
        kind: One of ``stub``, ``anthropic``, ``openai``, or ``azure-openai``.
        model_snapshot: Pinned model identifier.
        **kwargs: Forwarded to the provider constructor.

    Returns:
        The provider instance.

    Raises:
        KeyError: If ``kind`` is unknown.
    """
    if kind not in _PROVIDERS:
        raise KeyError(f"unknown provider '{kind}'; expected one of {sorted(_PROVIDERS)}")
    return _PROVIDERS[kind](model_snapshot, **kwargs)
