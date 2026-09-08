"""The HTTP surface: the chat UI, /turn, /confirm, /trace."""
import importlib
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "api.sqlite"))
    import app.main
    module = importlib.reload(app.main)
    return TestClient(module.app)


def body(utterance: str, conversation_id: str | None = None, **kwargs) -> dict:
    return {
        "conversationId": conversation_id or f"C-{uuid.uuid4().hex[:8]}",
        "tenantId": kwargs.get("tenant_id", "T-ACME"),
        "participantRef": kwargs.get("participant_ref", "P-1001"),
        "utterance": utterance,
        "uiContext": kwargs.get("ui_context", {}),
    }


def test_read_turn_answers_with_citations(client):
    response = client.post("/turn", json=body("what is my 401k balance right now"))
    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "answer" and payload["citations"]


def test_write_turn_previews_then_confirm_receipts(client):
    conversation = f"C-{uuid.uuid4().hex[:8]}"
    preview = client.post("/turn", json=body("change my 401k contribution to 8 percent", conversation)).json()
    assert preview["kind"] == "preview"

    receipt = client.post("/confirm", json={
        "conversationId": conversation,
        "proposalId": preview["proposal"]["proposal_id"],
        "nonce": preview["nonce"],
    })
    assert receipt.status_code == 200
    assert receipt.json()["receipt"]["verified_rate"] == "0.08"


def test_a_bad_nonce_is_rejected_by_the_api(client):
    conversation = f"C-{uuid.uuid4().hex[:8]}"
    preview = client.post("/turn", json=body("change my 401k contribution to 8 percent", conversation)).json()
    bad = client.post("/confirm", json={
        "conversationId": conversation,
        "proposalId": preview["proposal"]["proposal_id"],
        "nonce": "wrong",
    })
    assert bad.status_code == 409


def test_confirming_twice_does_not_execute_twice(client):
    conversation = f"C-{uuid.uuid4().hex[:8]}"
    preview = client.post("/turn", json=body("change my 401k contribution to 8 percent", conversation)).json()
    confirm = {"conversationId": conversation,
               "proposalId": preview["proposal"]["proposal_id"],
               "nonce": preview["nonce"]}
    assert client.post("/confirm", json=confirm).status_code == 200
    assert client.post("/confirm", json=confirm).status_code == 409


def test_a_new_utterance_never_resumes_a_pending_proposal(client):
    conversation = f"C-{uuid.uuid4().hex[:8]}"
    client.post("/turn", json=body("change my 401k contribution to 8 percent", conversation))
    blocked = client.post("/turn", json=body("what is my 401k balance right now", conversation))
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["kind"] == "pending_confirmation"


def test_trace_returns_the_audit_events_for_the_conversation(client):
    conversation = f"C-{uuid.uuid4().hex[:8]}"
    client.post("/turn", json=body("what is my 401k balance right now", conversation))
    trace = client.get(f"/trace/{conversation}").json()
    nodes = [event["node"] for event in trace["events"]]
    assert "planner" in nodes and "respond" in nodes
    assert trace["versions"]["catalog_version"] == "intents-v7"


def test_a_balance_answer_states_the_balance(client):
    answer = client.post("/turn", json=body("what is my 401k balance right now")).json()
    assert answer["kind"] == "answer"
    assert "412300.00" in answer["text"]


def test_a_deductible_answer_states_the_accumulator(client):
    answer = client.post("/turn", json=body("how much of my deductible have i met")).json()
    assert answer["kind"] == "answer"
    assert "1450.00" in answer["text"] and answer["citations"]


def test_absent_coverage_says_so_instead_of_reporting_no_values(client):
    answer = client.post("/turn", json=body(
        "how much of my deductible have i met", participant_ref="P-1002")).json()
    assert answer["kind"] == "answer"
    assert "coverage on file" in answer["text"]
    assert answer["limitation"].startswith("not applicable to this participant")


def test_a_second_turn_does_not_inherit_the_first_turns_answer(client):
    conversation = f"C-{uuid.uuid4().hex[:8]}"
    first = client.post("/turn", json=body("what is my 401k balance right now", conversation)).json()
    second = client.post("/turn", json=body("how much of my deductible have i met", conversation)).json()
    assert second["kind"] == "answer"
    assert second["text"] != first["text"]


def test_a_turn_after_a_receipt_answers_instead_of_repeating_the_receipt(client):
    conversation = f"C-{uuid.uuid4().hex[:8]}"
    preview = client.post("/turn", json=body("change my 401k contribution to 8 percent", conversation)).json()
    client.post("/confirm", json={"conversationId": conversation,
                                  "proposalId": preview["proposal"]["proposal_id"],
                                  "nonce": preview["nonce"]})
    after = client.post("/turn", json=body("what is my 401k balance right now", conversation)).json()
    assert after["kind"] == "answer"


def test_a_slot_filling_clarify_option_completes_the_route(client):
    conversation = f"C-{uuid.uuid4().hex[:8]}"
    asked = client.post("/turn", json=body("i want to change my contributions", conversation)).json()
    assert asked["kind"] == "clarify"
    resolved = client.post("/turn", json=body("8 percent", conversation, ui_context={
        "clarify_rounds": 1,
        "clarify_options": asked["options"],
        "clarify_utterance": "i want to change my contributions",
    })).json()
    assert resolved["kind"] == "preview"
    assert resolved["proposal"]["requested"]["rate"] == "0.08"


def test_chat_ui_is_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "/static/app.js" in page.text
    assert client.get("/static/app.js").status_code == 200


def test_no_participant_email_reaches_the_trace(client):
    conversation = f"C-{uuid.uuid4().hex[:8]}"
    client.post("/turn", json=body("what is my 401k balance right now", conversation))
    trace = client.get(f"/trace/{conversation}").text
    assert "@example.com" not in trace


def test_trace_reports_no_langsmith_link_when_tracing_is_off(client):
    conversation = f"C-{uuid.uuid4().hex[:8]}"
    client.post("/turn", json=body("what is my 401k balance right now", conversation))
    trace = client.get(f"/trace/{conversation}").json()
    assert trace["langsmith_run_url"] is None
