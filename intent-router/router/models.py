"""Core data models shared by every stage of the engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional

from errors import ConfigError, RoutingError  # noqa: F401  (re-exported)

AuthLevel = Literal["standard", "stepped_up"]
RiskTier = Literal["READ", "TRANSACT", "SENSITIVE"]
Band = Literal["HIGH", "MEDIUM", "LOW", "RULE"]

OUT_OF_SCOPE = "__out_of_scope__"


@dataclass(frozen=True)
class RequestContext:
    tenant_id: str
    participant_ref: str
    auth_level: AuthLevel
    tenant_capabilities: Mapping[str, Any] = field(default_factory=dict)
    tenant_frozen: bool = False
    ui_context: Mapping[str, Any] = field(default_factory=dict)
    conversation_state: Mapping[str, Any] = field(default_factory=dict)

    def with_conversation_state(self, state: Mapping[str, Any]) -> "RequestContext":
        return RequestContext(
            tenant_id=self.tenant_id,
            participant_ref=self.participant_ref,
            auth_level=self.auth_level,
            tenant_capabilities=self.tenant_capabilities,
            tenant_frozen=self.tenant_frozen,
            ui_context=self.ui_context,
            conversation_state=MappingProxyType(dict(state)),
        )


@dataclass(frozen=True)
class Interpretation:
    intent: str
    slots: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    top_alternatives: tuple = ()
    source: Literal["rule", "model"] = "model"


@dataclass(frozen=True)
class RoutingDecision:
    graph: str
    entry_node: Optional[str]
    budgets: Mapping[str, int]
    fired_row: int
    ladder_path: tuple
    cache_key: Optional[str]
    versions: Mapping[str, str]
    intent: str
    slots: Mapping[str, Any]
    missing_slots: tuple
    band: Band
    source: Literal["rule", "model"]
    confidence: float
    clarifying_question: Optional[Mapping[str, Any]] = None
    handoff_payload: Optional[Mapping[str, Any]] = None
    log_for_catalog_review: bool = False
    no_data_reads: bool = False
    note: Optional[str] = None
    conversation_state: Mapping[str, Any] = field(default_factory=dict)
