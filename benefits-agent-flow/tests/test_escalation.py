"""The frontier rung of the escalation ladder: advisory, envelope-only, still gated."""
import pytest

from app import escalation, graph_build
from app.state import EnvelopeItem, PlanDecision
from tests.conftest import make_state, run_turn


@pytest.fixture
def frontier(monkeypatch):
    """Stand in for deepagents: record what it was asked, return a scripted draft."""
    calls: list[dict] = []

    def fake(question: str, envelope: list[dict]) -> dict:
        calls.append({"question": question, "envelope": envelope})
        return {"text": "Your 401k balance is 412300.00 [ev-1].", "model": "fake-frontier"}

    monkeypatch.setattr(escalation, "available", lambda: True)
    monkeypatch.setattr(escalation, "compose", fake)
    return calls


def test_escalation_is_off_by_default(monkeypatch):
    monkeypatch.delenv("ESCALATION_BACKEND", raising=False)
    assert escalation.backend() == "none"
    assert not escalation.available()


def test_the_frontier_only_sees_read_only_tools_over_the_envelope():
    envelope = [{
        "item_id": "ev-1", "kind": "fact", "evidence_type": "balances",
        "effective_from": "2026-01-01", "effective_to": None, "payload": {"401k": "412300.00"},
    }]
    tools = escalation._tools(envelope)
    assert [tool.__name__ for tool in tools] == list(escalation.ADVISORY_TOOLS)
    listed = tools[0]()
    assert "412300.00" not in listed          # ids first, payload only when asked for
    assert "412300.00" in tools[1]("ev-1")
    assert "no such item" in tools[1]("ev-99")


def test_a_failed_composition_escalates_once_and_the_frontier_answer_is_revalidated(
    graph, monkeypatch, frontier
):
    """The mock composer keeps injecting an uncited claim, so both retries fail."""
    monkeypatch.setenv("MODEL_UNCITED_CLAIM", "always")
    result = run_turn(graph, make_state("what is my 401k balance right now"))
    assert result["retries"]["escalate"] == 1
    assert len(frontier) == 1
    assert result["validation"]["passed"]
    assert result["response"]["text"] == "Your 401k balance is 412300.00 [ev-1]."


def test_the_frontier_is_handed_the_envelope_and_nothing_else(graph, monkeypatch, frontier):
    monkeypatch.setenv("MODEL_UNCITED_CLAIM", "always")
    run_turn(graph, make_state("what is my 401k balance right now"))
    handed = frontier[0]["envelope"]
    assert handed and all(set(item) >= {"item_id", "payload"} for item in handed)
    assert "@example.com" not in str(handed)


def test_an_unsupported_frontier_answer_is_rejected_like_any_other_draft(
    graph, monkeypatch
):
    monkeypatch.setenv("MODEL_UNCITED_CLAIM", "always")
    monkeypatch.setattr(escalation, "available", lambda: True)
    monkeypatch.setattr(escalation, "compose", lambda q, e: {
        "text": "Your balance is 999999.00 [ev-1].", "model": "fake-frontier",
    })
    result = run_turn(graph, make_state("what is my 401k balance right now"))
    assert not result["validation"]["passed"]
    assert result["response"].get("scripted")


def test_a_frontier_error_falls_through_to_the_human(graph, monkeypatch):
    monkeypatch.setenv("MODEL_UNCITED_CLAIM", "always")
    monkeypatch.setattr(escalation, "available", lambda: True)
    monkeypatch.setattr(escalation, "compose", lambda q, e: {"error": "TimeoutError"})
    result = run_turn(graph, make_state("what is my 401k balance right now"))
    assert result["response"].get("scripted")


def test_a_write_turn_never_reaches_the_frontier(monkeypatch):
    monkeypatch.setattr(escalation, "available", lambda: True)
    state = make_state("change my 401k contribution to 8 percent")
    state.envelope = [EnvelopeItem(
        item_id="ev-1", kind="fact", source="odl.elections.v5", effective_from="2026-01-01",
        observed_at="2026-06-01T00:00:00+00:00", evidence_type="elections",
        payload={"rate": "0.06"},
    )]
    state.plan = _plan("WRITE")
    assert not graph_build.escalates(state)
    state.plan = _plan("READ")
    assert graph_build.escalates(state)


def _plan(posture: str) -> PlanDecision:
    return PlanDecision(
        graph_id="g_balance_readonly_v1", table_row_id="r-030", rung=0, posture=posture,
        requires_capability="retirement_view",
    )
