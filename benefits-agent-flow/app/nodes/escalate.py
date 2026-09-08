"""Frontier step of the escalation ladder. Advisory only, and re-validated by gate_output."""
from app import escalation
from app.audit import node_event
from app.state import TurnState


def run(state: TurnState) -> dict:
    attempts = state.retries.get("escalate", 0)
    question = state.utterance
    envelope = [item.model_dump() for item in state.envelope]
    result = escalation.compose(question, envelope)
    retries = {**state.retries, "escalate": attempts + 1}
    if "error" in result:
        node_event(state, "escalate", failed=True, reason=result["error"])
        return {"retries": retries}
    node_event(state, "escalate", chars=len(result["text"]), model=result["model"])
    return {"draft": result["text"], "retries": retries}
