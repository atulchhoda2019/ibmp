"""Retrieval is filtered before it is ranked, and stale evidence never silently answers."""
from app.mocks import evidence
from app.nodes import assemble, gate_plan
from tests.conftest import make_state, run_turn


def test_retrieval_is_filtered_by_tenant_before_ranking():
    result = evidence.retrieve("T-ACME", ["PL-401K-ACME"], "2026-06-01", "annual deferral limit")
    sources = {item["passage_id"] for item in result["payload"]}
    assert sources
    assert all(item["tenant_id"] == "T-ACME" for item in result["payload"])
    zen = evidence.retrieve("T-ZEN", ["PL-401K-ZEN"], "2026-06-01", "annual deferral limit")
    assert sources.isdisjoint({item["passage_id"] for item in zen["payload"]})


def test_retrieval_is_filtered_by_effective_date():
    current = evidence.retrieve("T-ACME", ["PL-401K-ACME"], "2026-06-01", "annual deferral limit")
    assert all(item["effective_from"] <= "2026-06-01" for item in current["payload"])
    past = evidence.retrieve("T-ACME", ["PL-401K-ACME"], "2024-06-01", "annual deferral limit")
    assert {i["passage_id"] for i in past["payload"]} != {i["passage_id"] for i in current["payload"]}


def test_a_plan_the_participant_is_not_in_is_never_retrieved():
    result = evidence.retrieve("T-ACME", ["PL-401K-ACME"], "2026-06-01", "physical therapy")
    assert all(item["plan_id"] == "PL-401K-ACME" for item in result["payload"])


def test_envelope_items_are_effective_dated_and_citable(graph):
    result = run_turn(graph, make_state("what is my 401k balance right now"))
    assert result["envelope"]
    for item in result["envelope"]:
        assert item.item_id.startswith("ev-")
        assert item.effective_from and item.observed_at
        assert item.source


def test_missing_required_evidence_abstains_instead_of_answering(graph, monkeypatch):
    monkeypatch.setenv("EVIDENCE_EMPTY", "1")
    result = run_turn(graph, make_state("how much of my deductible have I met this year"))
    response = result["response"]
    # Either the single refetch recovered, or we abstained: never a confident answer with no policy.
    if result.get("abstain_reason"):
        assert response["limitation"]
        assert response["citations"] == []
    else:
        assert any(item.evidence_type == "plan_rules" for item in result["envelope"])


def test_freshness_policy_governs_staleness():
    state = make_state("x", turn_started_at="2026-06-01T12:00:00+00:00", service_date="2026-06-01")
    fresh_item = _item("elections", "fp_live", observed_at="2026-06-01T11:59:30+00:00")
    stale_item = _item("elections", "fp_live", observed_at="2026-06-01T09:00:00+00:00")
    assert assemble.fresh(fresh_item, state) is True
    assert assemble.fresh(stale_item, state) is False


def _item(evidence_type: str, policy: str, observed_at: str):
    from app.state import EnvelopeItem
    return EnvelopeItem(
        item_id="ev-1", kind="fact", source="test", effective_from="2024-01-01",
        observed_at=observed_at, freshness_policy=policy, evidence_type=evidence_type, payload={},
    )


def test_capability_gate_blocks_a_tenant_without_the_bundle(graph):
    result = run_turn(graph, make_state("change my 401k contribution to 8 percent",
                                        tenant_id="T-LOCK", participant_ref="P-4001"))
    assert result["response"]["kind"] in ("handoff", "answer")
    assert result.get("proposal") is None


def test_rung_ceiling_is_the_lower_of_tenant_and_table():
    assert gate_plan.effective_rung("T-ZEN", "contribution_change", 4) == 1
    assert gate_plan.effective_rung("T-ACME", "contribution_change", 4) == 2
    assert gate_plan.effective_rung("T-NOVA", "contribution_change", 2) == 2
