"""I1-I6, plus the failure drills from the design's §11."""
import os

import pytest
from langgraph.types import Command

from app.mocks import model_gateway, odl, sor
from app.nodes import planner
from app.registry import catalog, decision_table
from tests.conftest import config_for, make_state, run_turn

CHANGE = "change my 401k contribution to 8 percent"
READ = "what happens if I raise my contribution to 8 percent"


def test_i1_the_table_decides_the_graph_not_the_classifier(graph):
    known = {spec["name"] for spec in catalog()["intents"]} | {"any", "__out_of_scope__"}
    assert {row["intent"] for row in decision_table()["rows"]} <= known
    result = run_turn(graph, make_state(READ))
    assert result["plan"].graph_id == next(
        row["graph"] for row in decision_table()["rows"]
        if row["row"] == result["plan"].table_row_id)


def test_i2_an_advisory_row_can_never_reach_the_corridor(graph):
    for row in decision_table()["rows"]:
        if row["posture"] == "READ":
            assert row["rung_max"] == 0


def test_i3_every_number_in_an_answer_is_backed_by_the_envelope(graph):
    result = run_turn(graph, make_state(READ))
    assert result["validation"]["passed"] is True
    assert result["validation"]["reasons"] == []


def test_i4_a_closed_catalog_means_unknown_intents_are_out_of_scope():
    intent = planner.stage1_classifier("please reticulate my splines")
    known = {spec["name"] for spec in catalog()["intents"]} | {"__out_of_scope__"}
    assert intent.name in known


def test_i5_no_write_happens_without_a_matching_confirmation(graph):
    state = make_state(CHANGE)
    run_turn(graph, state)
    assert odl.get_elections("P-1001")["payload"]["rate"] == "0.06"


def test_i6_the_same_proposal_executes_at_most_once(graph):
    state = make_state(CHANGE)
    config = config_for(state)
    preview = run_turn(graph, state)["response"]
    resume = Command(resume={"proposal_id": preview["proposal"]["proposal_id"],
                             "nonce": preview["nonce"]})
    first = graph.invoke(resume, config)["response"]["receipt"]
    assert sor.lookup_by_key(first["idempotency_key"])["receipt_id"] == first["receipt_id"]
    assert len(sor.receipts()) == 1


def test_drill_an_empty_retrieval_abstains_rather_than_guesses(graph, monkeypatch):
    monkeypatch.setenv("EVIDENCE_EMPTY", "always")
    result = run_turn(graph, make_state(READ))
    assert result["response"]["citations"] == []
    assert "missing required evidence" in result["response"]["limitation"]


def test_drill_a_model_timeout_retries_once_then_falls_back(graph, monkeypatch):
    monkeypatch.setenv("MODEL_TIMEOUT_ONCE", "1")
    model_gateway.reset()
    result = run_turn(graph, make_state(READ))
    assert result["retries"]["reason"] >= 2
    assert result["response"]["kind"] == "answer"


def test_drill_an_uncited_claim_is_never_shown_to_the_user(graph, monkeypatch):
    monkeypatch.setenv("MODEL_UNCITED_CLAIM", "1")
    model_gateway.reset()
    result = run_turn(graph, make_state(READ))
    assert "12345.00" not in result["response"]["text"]


def test_drill_an_unknown_sor_outcome_reconciles_instead_of_resubmitting(graph, monkeypatch):
    monkeypatch.setenv("SOR_TIMEOUT_ONCE", "1")
    sor.reset()
    state = make_state(CHANGE)
    config = config_for(state)
    preview = run_turn(graph, state)["response"]
    final = graph.invoke(Command(resume={"proposal_id": preview["proposal"]["proposal_id"],
                                         "nonce": preview["nonce"]}), config)
    assert final["response"]["kind"] == "receipt"
    assert len(sor.receipts()) == 1
    assert odl.get_elections("P-1001")["payload"]["rate"] == "0.08"


@pytest.mark.parametrize("flag", ["MODEL_UNCITED_CLAIM", "MODEL_TIMEOUT_ONCE",
                                  "EVIDENCE_EMPTY", "SOR_TIMEOUT_ONCE"])
def test_the_fault_flags_are_off_by_default(flag):
    assert os.environ.get(flag) is None
