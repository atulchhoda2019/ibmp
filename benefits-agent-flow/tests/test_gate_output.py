"""The output gate is the last thing between a composed sentence and the participant."""
from app.nodes import gate_output
from app.state import EnvelopeItem, TurnState
from tests.conftest import make_state, run_turn


def state_with(draft: str) -> TurnState:
    state = make_state("x")
    state.envelope = [EnvelopeItem(
        item_id="ev-1", kind="fact", source="odl.elections.v5", effective_from="2024-01-01",
        observed_at="2026-06-01T00:00:00+00:00", evidence_type="elections",
        payload={"rate": "0.06", "annual_limit": "23500.00"},
    )]
    state.draft = draft
    return state


def test_uncited_sentence_fails():
    result = gate_output.validate(state_with("Your rate is 0.06 [ev-1]. You should switch plans."))
    assert not result["passed"]
    assert any("uncited" in reason for reason in result["reasons"])


def test_number_not_present_in_the_envelope_fails():
    result = gate_output.validate(state_with("Your rate is 0.09 [ev-1]."))
    assert not result["passed"]
    assert any("unsupported numbers" in reason for reason in result["reasons"])


def test_citation_to_an_item_that_is_not_in_the_envelope_fails():
    result = gate_output.validate(state_with("Your rate is 0.06 [ev-9]."))
    assert not result["passed"]
    assert any("unknown citation" in reason for reason in result["reasons"])


def test_pii_in_the_output_fails():
    result = gate_output.validate(state_with("Your rate is 0.06 [ev-1]. Sent to a@b.com [ev-1]."))
    assert not result["passed"]
    assert "pii_in_output" in result["reasons"]


def test_supported_draft_passes_and_reports_its_citations():
    result = gate_output.validate(state_with("Your rate is 0.06 [ev-1]. The limit is 23500.00 [ev-1]."))
    assert result["passed"] and result["citations"] == ["ev-1"]


def test_an_uncited_claim_is_caught_and_the_retry_answers(graph, monkeypatch):
    """Fault drill: the composer injects one uncited claim, the gate catches it, one retry."""
    monkeypatch.setenv("MODEL_UNCITED_CLAIM", "1")
    result = run_turn(graph, make_state("what is my 401k balance right now"))
    assert result["retries"]["reason"] == 2
    assert result["response"]["kind"] == "answer"
    assert result["validation"]["passed"]


def test_a_model_timeout_retries_once_then_falls_back(graph, monkeypatch):
    monkeypatch.setenv("MODEL_TIMEOUT_ONCE", "1")
    result = run_turn(graph, make_state("what is my 401k balance right now"))
    assert result["retries"]["reason"] >= 2
    assert result["response"]["kind"] == "answer"
