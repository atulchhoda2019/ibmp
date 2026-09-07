"""Behavioral details from §5: canonicalization, trace completeness, typed proposals."""
from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pytest

from app import evidence
from app.corridor import ConfirmationStore
from app.responder import respond
from conftest import (
    REGISTRY_PATH,
    T_AUTONOMOUS,
    T_DRAFT_ONLY,
    T_ENABLED,
    T_EXECUTE,
    context,
)
from graphs.registry import Proposal, load_registry
from router.catalog import SlotSpec
from router.models import Interpretation
from router.slots import canonicalize
from router.trace import TRACE_FIELDS


class FixedClassifier:
    """Pins the model's output so a band edge can be tested without a real model."""

    def __init__(self, interpretation: Interpretation) -> None:
        self.interpretation = interpretation

    def classify(self, utterance, context):
        return self.interpretation


@pytest.mark.parametrize("raw,expected", [("six percent", 6.0), ("6%", 6.0), (0.06, 6.0), (6, 6.0), ("6", 6.0)])
def test_percent_canonicalization(raw, expected):
    assert canonicalize(SlotSpec(name="rate_pct", type="percent"), raw) == expected


def test_uncanonicalizable_percent_raises(router):
    with pytest.raises(ValueError):
        canonicalize(SlotSpec(name="rate_pct", type="percent"), "a lot")


def test_trace_line_is_complete(router, trace):
    router.route("Increase my 401(k) to 8%", context())

    assert len(trace.lines) == 1
    assert set(trace.lines[0]) == set(TRACE_FIELDS)
    assert trace.lines[0]["graph"] == "G-CONTRIB-CHANGE"
    assert trace.lines[0]["versions"] == {"catalog": "catalog-v1", "table": "table-v1"}
    assert trace.as_jsonl()


def test_transactional_graph_ends_at_a_typed_proposal(registry):
    graph = registry.instantiate("G-CONTRIB-CHANGE")
    result = asyncio.run(graph.call("ProposeContributionChange", rate_pct=8.0))

    assert isinstance(result, Proposal)
    assert result.confirmation_nonce
    assert graph.calls == [("ProposeContributionChange", {"rate_pct": 8.0})]


def test_graph_rejects_tools_outside_its_manifest(registry):
    graph = registry.instantiate("G-DEDUCTIBLE")
    with pytest.raises(Exception):
        asyncio.run(graph.call("ProposeContributionChange", rate_pct=8.0))


def test_per_intent_band_edges_override_the_table(router, catalog):
    """0.88 is HIGH by the table's edges but only MEDIUM for a transaction."""
    assert router.table.band_for(0.88) == "HIGH"
    assert router.table.band_for(0.88, catalog.get("ret.contribution.change").bands) == "MEDIUM"

    router.classifier = FixedClassifier(
        Interpretation(
            intent="ret.contribution.change", slots={"rate_pct": 8.0}, confidence=0.88, source="model"
        )
    )
    decision = router.route("bump my deferral a bit", context(auth="stepped_up"))

    assert decision.band == "MEDIUM"
    assert decision.band_edges == "catalog"
    assert decision.graph == "G-CLARIFY"


def test_entitlement_is_decided_before_the_graph_can_read(router):
    decision = router.route(
        "How much of my deductible have I met?", context(entitlements={"VIEW": False, "CHANGE": False})
    )

    assert decision.capability == "unentitled"
    assert decision.graph == "G-EXPLAIN-ROUTE"
    turn = asyncio.run(
        respond(decision, load_registry(REGISTRY_PATH), context(), ConfirmationStore())
    )
    assert not any(call in turn.tool_calls for call in ("GetCoverage", "GetAccumulators"))


@pytest.mark.parametrize(
    "capabilities,graph",
    [
        (T_DRAFT_ONLY, "G-CONTRIB-DRAFT"),
        (T_ENABLED, "G-CONTRIB-CHANGE"),
        (T_EXECUTE, "G-CONTRIB-AUTO"),
        (T_AUTONOMOUS, "G-CONTRIB-AUTO"),
    ],
)
def test_the_rung_the_tenant_bought_picks_the_graph(router, capabilities, graph):
    decision = router.route(
        "Increase my 401(k) to 8%", context(capabilities=capabilities, auth="stepped_up")
    )
    assert decision.graph == graph


def test_a_confirmation_without_a_pending_proposal_falls_to_fallback(router):
    decision = router.route("yes, confirm it", context(auth="stepped_up"))

    assert decision.intent == "ret.contribution.confirm"
    assert "no_pending_proposal" in decision.ladder_path
    assert decision.graph == "G-FALLBACK"


def test_stale_evidence_abstains(registry):
    envelope = evidence.Envelope(
        citations=(evidence.Citation(source="spd", effective_date=date.today() - timedelta(days=40)),)
    )
    verdict = evidence.validate(envelope, {"max_age_days": 7, "abstain_if_stale": True})

    assert not verdict.ok
    assert verdict.reason == "stale"


def test_async_wrapper_matches_the_sync_core(router):
    ctx = context()
    assert asyncio.run(router.route_async("Increase my 401(k) to 8%", ctx)) == router.route(
        "Increase my 401(k) to 8%", ctx
    )
