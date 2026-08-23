"""Council orchestration: benchmark items in, structured candidate records out.

This package produces what the rest of the repository consumes.  See
``council.Council`` for the orchestrator and ``providers`` for the model
adapters, including an offline :class:`~orchestrator.providers.StubProvider`
that makes the whole pipeline testable without credentials or spend.
"""

from __future__ import annotations

from orchestrator.council import (
    AgentCall,
    Council,
    Item,
    detect_injection,
    redact,
    validate_record,
)
from orchestrator.providers import (
    AnthropicProvider,
    AzureOpenAIProvider,
    OpenAIProvider,
    Provider,
    ProviderResponse,
    XAIProvider,
    StubProvider,
    build_provider,
)
from orchestrator.safety import (
    SAFETY_FLAGS,
    CouncilHealth,
    DegradationState,
    KillSwitch,
    KillSwitchTripped,
    degradation_report,
    preserve_dissent,
    single_agent_fallback,
)

__all__ = [
    "AgentCall",
    "Council",
    "Item",
    "detect_injection",
    "redact",
    "validate_record",
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderResponse",
    "XAIProvider",
    "StubProvider",
    "build_provider",
    "SAFETY_FLAGS",
    "CouncilHealth",
    "DegradationState",
    "KillSwitch",
    "KillSwitchTripped",
    "degradation_report",
    "preserve_dissent",
    "single_agent_fallback",
]
