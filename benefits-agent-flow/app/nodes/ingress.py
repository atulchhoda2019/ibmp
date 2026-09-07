"""Canonicalize the request and attach trusted context.

Identity arrives as a claim (an email in the mock) and is exchanged here for an opaque
participant ref. The email never enters TurnState, a prompt, or a downstream tool.
"""
from datetime import datetime, timezone

from app.audit import node_event
from app.mocks import odl
from app.state import TurnState


def resolve_participant(claim: str) -> str:
    """Exchange a verified identity claim for a participant ref."""
    if claim in odl._load():  # already a ref
        return claim
    for ref, record in odl._load().items():
        if record["email"].lower() == claim.lower():
            return ref
    raise KeyError("unresolvable identity claim")


def fresh_turn() -> dict:
    """Clear everything the previous turn left on the thread.

    The checkpointer keys the thread on the conversation, so without this a second turn
    inherits the first turn's response, envelope and proposal.
    """
    return {
        "intent": None,
        "plan": None,
        "evidence_items": [],
        "fact_items": [],
        "calc_items": [],
        "envelope": [],
        "not_applicable": [],
        "abstain_reason": None,
        "draft": None,
        "validation": None,
        "proposal": None,
        "confirmed": False,
        "execution": None,
        "receipt": None,
        "response": None,
        "retries": {},
    }


def run(state: TurnState) -> dict:
    service_date = state.service_date or state.ui_context.get("service_date") or datetime.now(
        timezone.utc
    ).date().isoformat()
    utterance = " ".join(state.utterance.split())
    started_at = datetime.now(timezone.utc).isoformat()
    node_event(
        state, "ingress",
        service_date=service_date, utterance_len=len(utterance),
        intent=None, band=None, row=None, graph_id=None,
    )
    return {
        **fresh_turn(),
        "service_date": service_date,
        "utterance": utterance,
        "turn_started_at": started_at,
    }
