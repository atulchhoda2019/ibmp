"""The transaction corridor: propose, preview, nonce-bound confirmation, revalidate,
idempotent execute, read-after-write verify, receipt.

The nonce is checked inside `wait_confirmation`, not in the API handler, so nothing that
resumes the thread can skip it. Rung 3 and 4 keep every step except the human pause.
"""
import secrets
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from langgraph.types import interrupt

from app.audit import node_event
from app.mocks import calculator, odl, sor
from app.state import Proposal, TurnState

CONFIRMATION_TTL_MINUTES = 10


def _now() -> datetime:
    return datetime.now(timezone.utc)


def deterministic_checks(state: TurnState, rate: str) -> tuple[list[str], str | None]:
    """Returns (passed check names, failure explanation)."""
    passed: list[str] = []
    record = odl.participant(state.participant_ref)
    plan = odl.plans()[record["plan_id"]]

    eligibility = odl.get_eligibility(state.participant_ref)["payload"]
    if not eligibility["eligible"]:
        return passed, (
            f"You are not eligible to defer until {eligibility['eligibility_date']}, "
            "so I cannot submit this change yet."
        )
    passed.append("eligibility")

    if Decimal(rate) > Decimal(plan["max_deferral_rate"]):
        return passed, f"Your plan caps deferrals at {plan['max_deferral_rate']} of pay."
    passed.append("plan_max_rate")

    room = calculator.limit_room(state.participant_ref)["payload"]
    comp = Decimal(record["elections"].get("annual_compensation", "0"))
    projected_remaining = comp * Decimal(rate)
    if projected_remaining > Decimal(room["room_remaining"]):
        return passed, (
            f"At {rate} of pay you would defer {projected_remaining:.2f} more this year, but only "
            f"{room['room_remaining']} of your {room['annual_limit']} annual limit is left "
            f"after {room['ytd_deferral']} year to date."
        )
    passed.append("limit_room")
    return passed, None


def propose(state: TurnState) -> dict:
    rate = state.intent.slots.get("rate")
    if not rate:
        node_event(state, "corridor_propose", ok=False, reason="missing_rate")
        return {"response": {
            "kind": "clarify",
            "question": "What contribution rate would you like?",
            "options": [{"intent": "contribution_change", "label": label} for label in ("6 percent", "8 percent")],
        }}

    passed, failure = deterministic_checks(state, rate)
    if failure:
        node_event(state, "corridor_propose", ok=False, reason="validation_failed", checks=passed)
        citations = [item.item_id for item in state.envelope]
        return {"response": {
            "kind": "answer",
            "text": failure,
            "citations": citations,
            "failed_validation": True,
        }}

    current = odl.get_elections(state.participant_ref)["payload"]
    proposal = Proposal(
        proposal_id=f"PRP-{secrets.token_hex(6)}",
        action="ContributionChange",
        participant_ref=state.participant_ref,
        current={"rate": current["rate"]},
        requested={"rate": rate, "effective_date": state.intent.slots.get("effective_date")
                   or state.service_date},
        validations=passed,
        nonce=secrets.token_urlsafe(16),
        expires_at=(_now() + timedelta(minutes=CONFIRMATION_TTL_MINUTES)).isoformat(),
    )
    node_event(state, "corridor_propose", ok=True, proposal_id=proposal.proposal_id, checks=passed)
    return {"proposal": proposal}


def preview(state: TurnState) -> dict:
    proposal = state.proposal
    response = {
        "kind": "preview",
        "proposal": proposal.model_dump(),
        "nonce": proposal.nonce,
        "expires_at": proposal.expires_at,
        "undo_window": "You can reverse this from the same conversation until the next payroll cutoff.",
        "citations": [item.item_id for item in state.envelope],
    }
    node_event(state, "corridor_preview", proposal_id=proposal.proposal_id)
    return {"response": response}


def wait_confirmation(state: TurnState) -> dict:
    """Pauses the run. Resumes only for this proposal id with this nonce, before expiry."""
    proposal = state.proposal
    answer = interrupt({
        "kind": "await_confirmation",
        "proposal_id": proposal.proposal_id,
        "expires_at": proposal.expires_at,
    })
    answer = answer or {}
    if answer.get("proposal_id") != proposal.proposal_id:
        raise ValueError("confirmation is for a different proposal")
    if not secrets.compare_digest(str(answer.get("nonce", "")), proposal.nonce):
        raise ValueError("confirmation nonce mismatch")
    if _now() > datetime.fromisoformat(proposal.expires_at):
        raise ValueError("confirmation expired")
    node_event(state, "corridor_wait_confirmation", confirmed=True, proposal_id=proposal.proposal_id)
    return {"confirmed": True}


def revalidate(state: TurnState) -> dict:
    """The world may have moved while the proposal sat waiting: check again before writing."""
    proposal = state.proposal
    passed, failure = deterministic_checks(state, proposal.requested["rate"])
    if failure:
        node_event(state, "corridor_revalidate", ok=False)
        return {"response": {
            "kind": "answer",
            "text": f"That change no longer passes plan validation: {failure}",
            "citations": [item.item_id for item in state.envelope],
            "failed_validation": True,
        }, "confirmed": False}
    node_event(state, "corridor_revalidate", ok=True, checks=passed)
    return {}


def execute(state: TurnState) -> dict:
    """Idempotent submit keyed by proposal id. An unknown outcome is a state, not a crash."""
    proposal = state.proposal
    command = {
        "idempotency_key": proposal.proposal_id,
        "action": proposal.action,
        "participant_ref": proposal.participant_ref,
        "params": proposal.requested,
    }
    try:
        receipt = sor.submit(command)
        node_event(state, "corridor_execute", outcome="known", proposal_id=proposal.proposal_id)
        return {"execution": {"outcome": "known", "receipt": receipt}}
    except sor.SorUnknownOutcome:
        node_event(state, "corridor_execute", outcome="unknown", proposal_id=proposal.proposal_id)
        return {"execution": {"outcome": "unknown", "receipt": None}}


def verify(state: TurnState) -> dict:
    """Read after write, reconciling by idempotency key when the submit outcome was unknown."""
    proposal = state.proposal
    receipt = (state.execution or {}).get("receipt")
    if receipt is None:
        receipt = sor.lookup_by_key(proposal.proposal_id)
    if receipt is None:
        node_event(state, "corridor_verify", verified=False, reason="no_receipt")
        return {"response": {
            "kind": "answer",
            "text": "I could not confirm the change was recorded, so I have not reported it as done. "
                    "A specialist will reconcile it.",
            "citations": [],
        }}

    observed = odl.get_elections(proposal.participant_ref)["payload"]["rate"]
    verified = Decimal(observed) == Decimal(proposal.requested["rate"])
    node_event(state, "corridor_verify", verified=verified, receipt_id=receipt["receipt_id"])
    if not verified:
        return {"response": {
            "kind": "answer",
            "text": "The system of record does not yet show the new rate, so I will not claim it is done.",
            "citations": [],
        }}
    return {"receipt": {**receipt, "verified_rate": observed, "notified": state.plan.rung >= 3}}
