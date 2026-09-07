"""The HTTP surface: /turn, /confirm, /trace."""
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


def test_no_participant_email_reaches_the_trace(client):
    conversation = f"C-{uuid.uuid4().hex[:8]}"
    client.post("/turn", json=body("what is my 401k balance right now", conversation))
    trace = client.get(f"/trace/{conversation}").text
    assert "@example.com" not in trace
