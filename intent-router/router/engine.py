"""The deterministic planner.

Stage 0 rules -> stage 1 classifier -> slot canonicalization -> stage 2 decision table ->
stage 3 ambiguity ladder. The engine selects exactly one pre-approved graph with budgets;
it assembles nothing and executes nothing.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from graphs.registry import GraphRegistry, load_registry

from .cache import derive_cache_key
from .catalog import Catalog, IntentSpec, load_catalog
from .models import (
    Band,
    ConfigError,
    Interpretation,
    RequestContext,
    RoutingDecision,
    RoutingError,
)
from .rules import match_rules
from .slots import resolve_slots
from .table import DecisionTable, capability_rung, capability_state, load_table
from .trace import TraceSink, build_trace

CONFIG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = CONFIG_ROOT / "catalog" / "v1.yaml"
DEFAULT_TABLE = CONFIG_ROOT / "table" / "v1.yaml"
DEFAULT_REGISTRY = CONFIG_ROOT / "graphs" / "registry.yaml"

CLARIFY_GRAPH = "G-CLARIFY"
FALLBACK_GRAPH = "G-FALLBACK"
PENDING_PROPOSAL = "pending_proposal"


class IntentRouter:
    def __init__(
        self,
        *,
        catalog: Catalog,
        table: DecisionTable,
        registry: GraphRegistry,
        classifier: Any,
        trace_sink: Optional[TraceSink] = None,
    ) -> None:
        if table.catalog_version != catalog.version:
            raise ConfigError(
                f"table {table.version} declares catalog {table.catalog_version}, got {catalog.version}"
            )
        self.catalog = catalog
        self.table = table
        self.registry = registry
        self.classifier = classifier
        self.trace_sink = trace_sink
        self.tool_calls: List[Tuple[str, dict]] = []

    # -- public API ---------------------------------------------------------

    def route(self, utterance: str, context: RequestContext) -> RoutingDecision:
        ladder: List[str] = []
        pending = context.conversation_state.get("pending_clarify")
        selected = None
        if pending is not None:
            ladder.append("clarify")
            utterance_for_classification = f"{pending['utterance']} {utterance}"
            selected = self._selected_option(utterance, pending)
        else:
            utterance_for_classification = utterance

        if selected is not None:
            # Picking one of the offered named options is a resolution, not an interpretation.
            ladder.extend(("option_selected", "reclassified"))
            interpretation = selected
        else:
            interpretation = self._interpret(
                utterance_for_classification, context, ladder, reclassified=pending is not None
            )
        intent = self.catalog.get(interpretation.intent)
        band = self._band(interpretation, intent)
        ladder.append(f"band={band}")

        if intent.requires_pending_proposal and PENDING_PROPOSAL not in context.conversation_state:
            # A confirmation with nothing to confirm is not a transaction, it is noise.
            ladder.append("no_pending_proposal")
            band = "LOW"

        slots = resolve_slots(intent, interpretation.slots, context)
        slot_error = slots.error
        if slot_error is not None:
            # A bad value is never guessed or clamped: it becomes a clarification.
            ladder.append("slot_validation_failed")
            band = "MEDIUM"

        if pending is not None and band not in ("RULE", "HIGH"):
            # Exactly one clarifying question per conversation: a second unresolved pass falls down.
            ladder.append("fall_down")
            band = "LOW"

        capability = capability_state(intent, context)
        ladder.append(f"entitlement={intent.permission}")
        rung = capability_rung(capability)
        try:
            row = self.table.match(
                intent=intent.id,
                band=band,
                capability=capability,
                risk=intent.risk_tier,
                auth=context.auth_level,
            )
        except ConfigError as exc:
            raise RoutingError(str(exc)) from None

        self.registry.assert_manifest_legal(
            row.graph,
            intent.risk_tier,
            capability_enabled=capability.startswith("enabled"),
            rung=rung,
            confirmed=PENDING_PROPOSAL in context.conversation_state,
        )

        clarifying_question = None
        conversation_state: Dict[str, Any] = {
            k: v for k, v in context.conversation_state.items() if k != "pending_clarify"
        }
        if row.graph == CLARIFY_GRAPH:
            ladder.append("clarify")
            clarifying_question = self._clarifying_question(intent, interpretation, slot_error)
            conversation_state["pending_clarify"] = {
                "utterance": utterance_for_classification,
                "options": [[label, candidate] for label, candidate in clarifying_question["options"]],
            }
            conversation_state["clarify_count"] = int(context.conversation_state.get("clarify_count", 0)) + 1
        conversation_state.update(slots.values)

        handoff_payload = None
        if row.graph == FALLBACK_GRAPH:
            handoff_payload = self._handoff_payload(interpretation, slots.values, context)

        cache_key = derive_cache_key(
            intent=intent,
            slots=slots.values,
            band=band,
            risk=intent.risk_tier,
            catalog_version=self.catalog.version,
            table_version=self.table.version,
        )
        versions = {"catalog": self.catalog.version, "table": self.table.version}

        if row.no_data_reads and self.tool_calls:
            raise RoutingError(f"row {row.index} forbids data reads but tools were called")

        decision = RoutingDecision(
            graph=row.graph,
            entry_node=row.entry_node,
            budgets=dict(row.budgets),
            fired_row=row.index,
            ladder_path=tuple(ladder),
            cache_key=cache_key,
            versions=versions,
            intent=intent.id,
            slots=dict(slots.values),
            missing_slots=tuple(slots.missing_required),
            band=band,
            source=interpretation.source,
            confidence=interpretation.confidence,
            clarifying_question=clarifying_question,
            handoff_payload=handoff_payload,
            log_for_catalog_review=row.log_for_catalog_review,
            no_data_reads=row.no_data_reads,
            note=row.note,
            conversation_state=conversation_state,
            capability=capability,
            rung=rung,
            permission=intent.permission,
            posture=self.registry.get(row.graph).posture,
            band_edges="catalog" if intent.bands else "table",
            evidence_policy=dict(row.evidence) if row.evidence else None,
        )
        self._trace(decision, context)
        return decision

    async def route_async(self, utterance: str, context: RequestContext) -> RoutingDecision:
        """Thin async wrapper: the core stays pure and synchronous."""
        return await asyncio.get_running_loop().run_in_executor(None, self.route, utterance, context)

    # -- stages -------------------------------------------------------------

    def _interpret(
        self, utterance: str, context: RequestContext, ladder: List[str], *, reclassified: bool
    ) -> Interpretation:
        rule_hit = match_rules(utterance, self.catalog)
        if rule_hit is not None:
            ladder.append("stage0_rule")
            if reclassified:
                ladder.append("reclassified")
            return rule_hit
        ladder.append("stage0_miss")
        interpretation = self.classifier.classify(utterance, context)
        if interpretation.intent not in self.catalog:
            raise RoutingError(f"classifier returned {interpretation.intent!r}, outside the closed catalog")
        ladder.append("reclassified" if reclassified else "classified")
        return interpretation

    def _selected_option(self, utterance: str, pending: Mapping[str, Any]) -> Optional[Interpretation]:
        answer = utterance.strip().casefold()
        for label, candidate in pending.get("options", ()):
            if answer == label.strip().casefold() and candidate in self.catalog:
                return Interpretation(intent=candidate, confidence=1.0, source="rule")
        return None

    def _band(self, interpretation: Interpretation, intent: IntentSpec) -> Band:
        """Band edges are measured per intent: a transaction wants a higher bar than a read."""
        if interpretation.source == "rule":
            return "RULE"
        return self.table.band_for(interpretation.confidence, intent.bands)

    def _clarifying_question(
        self, intent: IntentSpec, interpretation: Interpretation, slot_error: Optional[str]
    ) -> Mapping[str, Any]:
        if slot_error is not None:
            return {
                "text": f"I can't use that value ({slot_error}). What should I set instead?",
                "options": [(self.catalog.label(intent.id), intent.id)],
                "reason": "slot_validation_failed",
            }
        options = [
            (self.catalog.label(candidate), candidate)
            for candidate, _ in interpretation.top_alternatives
            if candidate in self.catalog
        ]
        return {
            "text": "Which of these did you mean?",
            "options": options,
            "reason": "ambiguous_intent",
        }

    def _handoff_payload(
        self, interpretation: Interpretation, slots: Mapping[str, Any], context: RequestContext
    ) -> Mapping[str, Any]:
        return {
            "transcript_ref": f"{context.tenant_id}:{context.participant_ref}",
            "slots": dict(slots),
            "top_alternatives": [list(alt) for alt in interpretation.top_alternatives],
            "intent": interpretation.intent,
            "confidence": interpretation.confidence,
        }

    def _trace(self, decision: RoutingDecision, context: RequestContext) -> None:
        if self.trace_sink is None:
            return
        self.trace_sink(
            build_trace(
                tenant=context.tenant_id,
                intent=decision.intent,
                source=decision.source,
                confidence=decision.confidence,
                band=decision.band,
                band_edges=decision.band_edges,
                fired_row=decision.fired_row,
                graph=decision.graph,
                posture=decision.posture,
                capability=decision.capability,
                rung=decision.rung,
                ladder_path=decision.ladder_path,
                versions=dict(decision.versions),
                cache_key=decision.cache_key,
            )
        )


def build_router(
    *,
    classifier: Any,
    catalog_path: Path = DEFAULT_CATALOG,
    table_path: Path = DEFAULT_TABLE,
    registry_path: Path = DEFAULT_REGISTRY,
    trace_sink: Optional[TraceSink] = None,
) -> IntentRouter:
    return IntentRouter(
        catalog=load_catalog(catalog_path),
        table=load_table(table_path),
        registry=load_registry(registry_path),
        classifier=classifier,
        trace_sink=trace_sink,
    )
