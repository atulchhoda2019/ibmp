"""The write path: nothing commits without a live nonce, and nothing commits twice."""
import pytest
from langgraph.types import Command

from app.mocks import odl, sor
from tests.conftest import config_for, make_state, run_turn

CHANGE = "change my 401k contribution to 8 percent"


def propose(graph, **kwargs):
    state = make_state(CHANGE, **kwargs)
    config = config_for(state)
    result = run_turn(graph, state)
    return result, config


def test_rung_two_previews_and_waits_for_a_nonce(graph):
    result, config = propose(graph)
    assert result["response"]["kind"] == "preview"
    assert graph.get_state(config).next == ("corridor_wait_confirmation",)
    assert odl.get_elections("P-1001")["payload"]["rate"] == "0.06"   # nothing written yet


def test_confirmation_executes_verifies_and_receipts(graph):
    result, config = propose(graph)
    proposal = result["response"]["proposal"]
    final = graph.invoke(
        Command(resume={"proposal_id": proposal["proposal_id"], "nonce": proposal["nonce"]}), config)
    receipt = final["response"]["receipt"]
    assert final["response"]["kind"] == "receipt"
    assert receipt["idempotency_key"] == proposal["proposal_id"]
    assert receipt["verified_rate"] == "0.08"
    assert odl.get_elections("P-1001")["payload"]["rate"] == "0.08"


def test_a_wrong_nonce_never_executes(graph):
    result, config = propose(graph)
    proposal = result["response"]["proposal"]
    with pytest.raises(ValueError, match="nonce"):
        graph.invoke(Command(resume={"proposal_id": proposal["proposal_id"], "nonce": "not-it"}), config)
    assert odl.get_elections("P-1001")["payload"]["rate"] == "0.06"


def test_a_confirmation_for_another_proposal_never_executes(graph):
    result, config = propose(graph)
    proposal = result["response"]["proposal"]
    with pytest.raises(ValueError, match="different proposal"):
        graph.invoke(Command(resume={"proposal_id": "PRP-somebody-else", "nonce": proposal["nonce"]}), config)
    assert odl.get_elections("P-1001")["payload"]["rate"] == "0.06"


def test_an_expired_confirmation_never_executes(graph, monkeypatch):
    monkeypatch.setattr("app.nodes.corridor.CONFIRMATION_TTL_MINUTES", -1)
    result, config = propose(graph)
    proposal = result["response"]["proposal"]
    with pytest.raises(ValueError, match="expired"):
        graph.invoke(
            Command(resume={"proposal_id": proposal["proposal_id"], "nonce": proposal["nonce"]}), config)
    assert odl.get_elections("P-1001")["payload"]["rate"] == "0.06"


def test_replaying_the_same_command_commits_once(graph):
    result, config = propose(graph)
    proposal = result["response"]["proposal"]
    graph.invoke(Command(resume={"proposal_id": proposal["proposal_id"], "nonce": proposal["nonce"]}), config)
    first = sor.lookup_by_key(proposal["proposal_id"])
    replay = sor.submit({"idempotency_key": proposal["proposal_id"], "action": "ContributionChange",
                         "participant_ref": "P-1001", "params": {"rate": "0.08"}})
    assert replay["receipt_id"] == first["receipt_id"] and replay["replayed"] is True


def test_an_unknown_submit_outcome_reconciles_by_key_instead_of_resubmitting(graph, monkeypatch):
    monkeypatch.setenv("SOR_TIMEOUT_ONCE", "1")
    result, config = propose(graph)
    proposal = result["response"]["proposal"]
    final = graph.invoke(
        Command(resume={"proposal_id": proposal["proposal_id"], "nonce": proposal["nonce"]}), config)
    assert final["execution"]["outcome"] == "unknown"
    assert final["response"]["kind"] == "receipt"
    assert final["response"]["receipt"]["idempotency_key"] == proposal["proposal_id"]


def test_a_change_over_the_annual_limit_fails_validation_before_any_preview(graph):
    result = run_turn(graph, make_state("change my 401k contribution to 60 percent", participant_ref="P-1002"))
    assert result["response"]["kind"] == "answer"
    assert result["response"]["failed_validation"] is True
    assert result.get("proposal") is None


def test_an_ineligible_participant_cannot_propose(graph):
    result = run_turn(graph, make_state(CHANGE, participant_ref="P-1003"))
    assert result["response"].get("failed_validation") or result["response"]["kind"] == "handoff"
    assert result.get("proposal") is None


def test_rung_one_drafts_instead_of_proposing(graph):
    result = run_turn(graph, make_state(CHANGE, tenant_id="T-ZEN", participant_ref="P-2001"))
    assert result["response"]["kind"] == "draft"
    assert odl.get_elections("P-2001")["payload"]["rate"] == "0.05"


def test_rung_three_executes_without_a_human_pause_but_keeps_every_other_gate(graph):
    result = run_turn(graph, make_state(CHANGE, tenant_id="T-NOVA", participant_ref="P-3001"))
    assert result["response"]["kind"] == "receipt"
    assert result["response"]["receipt"]["notified"] is True
    assert result["proposal"].validations == ["eligibility", "plan_max_rate", "limit_room"]
    assert odl.get_elections("P-3001")["payload"]["rate"] == "0.08"


def test_a_view_only_tenant_cannot_reach_the_corridor(graph):
    result = run_turn(graph, make_state(CHANGE, tenant_id="T-LOCK", participant_ref="P-4001"))
    assert result["response"]["kind"] == "handoff"
    assert result.get("proposal") is None
    assert odl.get_elections("P-4001")["payload"]["rate"] == "0.07"


def test_a_participant_from_another_tenant_is_refused(graph):
    result = run_turn(graph, make_state(CHANGE, tenant_id="T-NOVA", participant_ref="P-1001"))
    assert result["response"]["reason"] == "tenant_mismatch"
    assert result.get("proposal") is None
