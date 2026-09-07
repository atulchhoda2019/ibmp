"""Presentation layer for the demo assistant.

The planner picks a graph; this module runs that graph's *stub* and turns the result into
an assistant turn. It contains no routing logic and no business logic: the questions it
asks are the ones the decision table told it to ask, and the corridor decides whether a
confirmation may fire.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from graphs.registry import GraphRegistry, Proposal, Receipt
from router.models import RequestContext, RoutingDecision

from app import evidence
from app.corridor import ConfirmationError, ConfirmationStore, PendingProposal

READ_ANSWERS = {
    "plan.deductible.status": (
        "You've met $840 of your $1,500 in-network deductible on {plan_id}. "
        "That's from claims processed through last week."
    ),
    "plan.deductible.info": "The {plan_id} plan has a $1,500 in-network individual deductible.",
}
DEDUCTIBLE_SUMMARY = (
    "Here's the safe part of that while we sort out the rest: on {plan_id} you've met $840 of "
    "your $1,500 in-network deductible. Tell me which plan or which number you meant and I'll go deeper."
)
ELECTIONS = "You're enrolled in {plan_id} with a 6% 401(k) deferral and no HSA contribution."

EXPLAIN_ROUTE_FROZEN = (
    "Contribution changes are paused right now during open enrollment, so I can't start one. "
    "Here's what will happen when the window reopens, and I can hand you to a specialist in the meantime."
)
EXPLAIN_ROUTE_DISABLED = (
    "Your plan doesn't let me change contributions from here. I can explain how the change works "
    "and connect you with someone who can make it for you."
)
EXPLAIN_ROUTE_UNENTITLED = (
    "You're not set up with permission to make that change from here — viewing and changing are "
    "granted separately. I can explain the process and get you to someone who can do it."
)
LIFE_EVENT = (
    "I'm sorry you're going through that. A divorce is a qualifying life event, so you can update "
    "your coverage within 60 days. I can show your current coverage and connect you with a specialist "
    "who can walk through the options."
)
FALLBACK = (
    "I'm not sure I understood that one. Let me get a specialist involved: I'll pass along what "
    "we've discussed so you don't have to start over."
)
NOTHING_TO_CONFIRM = (
    "There's nothing waiting for your confirmation right now. Tell me what you'd like to change "
    "and I'll build it for you to check first."
)


@dataclass
class AssistantTurn:
    text: str
    options: List[Dict[str, str]] = field(default_factory=list)
    proposal: Optional[Dict[str, Any]] = None
    receipt: Optional[Dict[str, Any]] = None
    tool_calls: List[str] = field(default_factory=list)
    conversation_state: Dict[str, Any] = field(default_factory=dict)


async def respond(
    decision: RoutingDecision,
    registry: GraphRegistry,
    context: RequestContext,
    store: ConfirmationStore,
    *,
    evidence_age_days: int = 2,
) -> AssistantTurn:
    graph = registry.instantiate(decision.graph)
    state = dict(decision.conversation_state)

    if decision.graph == "G-CLARIFY":
        question = decision.clarifying_question or {}
        return AssistantTurn(
            text=question.get("text", "Could you say a bit more?"),
            options=[{"label": label, "intent": intent} for label, intent in question.get("options", [])],
            conversation_state=state,
        )

    if decision.graph == "G-DEDUCTIBLE":
        return await _deductible(decision, graph, state, evidence_age_days)

    if decision.graph == "G-ELECTIONS-RO":
        await graph.call("GetElections")
        envelope = await _evidence(graph, decision, evidence_age_days)
        verdict = evidence.validate(envelope, decision.evidence_policy)
        text = ELECTIONS.format(plan_id=decision.slots.get("plan_id", "your medical plan"))
        return AssistantTurn(
            text=text if verdict.ok else verdict.message,
            tool_calls=_names(graph),
            conversation_state=state,
        )

    if decision.graph == "G-CONTRIB-DRAFT":
        return await _draft(decision, graph, state)

    if decision.graph == "G-CONTRIB-CHANGE":
        return await _contribution_change(decision, graph, context, store, state)

    if decision.graph == "G-CONTRIB-COMMIT":
        return await _commit(decision, graph, context, store, state)

    if decision.graph == "G-CONTRIB-AUTO":
        return await _autonomous(decision, graph, context, store, state)

    if decision.graph == "G-EXPLAIN-ROUTE":
        await graph.call("RetrieveEvidence", topic=decision.intent, age_days=evidence_age_days)
        return AssistantTurn(
            text=_explanation(decision), tool_calls=_names(graph), conversation_state=state
        )

    if decision.graph == "G-LIFE-EVENT-RO":
        await graph.call("GetCoverage")
        await graph.call("RetrieveEvidence", topic=decision.intent, age_days=evidence_age_days)
        return AssistantTurn(text=LIFE_EVENT, tool_calls=_names(graph), conversation_state=state)

    await graph.call("CsrHandoff", payload=decision.handoff_payload)
    text = NOTHING_TO_CONFIRM if "no_pending_proposal" in decision.ladder_path else FALLBACK
    return AssistantTurn(text=text, tool_calls=_names(graph), conversation_state=state)


def _explanation(decision: RoutingDecision) -> str:
    if decision.note == "writes-disabled-during-freeze":
        return EXPLAIN_ROUTE_FROZEN
    if decision.note == "permission-not-granted":
        return EXPLAIN_ROUTE_UNENTITLED
    return EXPLAIN_ROUTE_DISABLED


async def _deductible(decision, graph, state, evidence_age_days: int) -> AssistantTurn:
    await graph.call("GetCoverage", plan_id=decision.slots.get("plan_id"))
    await graph.call("GetAccumulators", plan_id=decision.slots.get("plan_id"))
    envelope = await _evidence(graph, decision, evidence_age_days)
    verdict = evidence.validate(envelope, decision.evidence_policy)
    if not verdict.ok:
        return AssistantTurn(text=verdict.message, tool_calls=_names(graph), conversation_state=state)
    plan_id = decision.slots.get("plan_id", "your plan")
    template = DEDUCTIBLE_SUMMARY if decision.entry_node == "readonly_summary" else READ_ANSWERS[decision.intent]
    return AssistantTurn(text=template.format(plan_id=plan_id), tool_calls=_names(graph), conversation_state=state)


async def _evidence(graph, decision: RoutingDecision, evidence_age_days: int) -> evidence.Envelope:
    payload = await graph.call("RetrieveEvidence", topic=decision.intent, age_days=evidence_age_days)
    return evidence.envelope_from_tool(payload)


async def _draft(decision: RoutingDecision, graph, state) -> AssistantTurn:
    """Rung 1: prepare the content, take zero direct action."""
    if "rate_pct" in decision.missing_slots:
        return AssistantTurn(text="What percentage would you like to contribute?", conversation_state=state)
    await graph.call("GetElections")
    await graph.call("GetContributionLimits")
    await graph.call("Calc402g", rate_pct=decision.slots["rate_pct"])
    await graph.call("DraftForHuman", rate_pct=decision.slots["rate_pct"])
    return AssistantTurn(
        text=(
            f"I've drafted the change to {decision.slots['rate_pct']:g}% and sent it to your benefits "
            "team to review. I can't submit it myself on your plan's settings."
        ),
        tool_calls=_names(graph),
        conversation_state=state,
    )


async def _contribution_change(
    decision: RoutingDecision, graph, context: RequestContext, store: ConfirmationStore, state
) -> AssistantTurn:
    if decision.entry_node == "step_up_auth":
        return AssistantTurn(
            text=(
                "I can set that up. First I need you to verify it's you — switch auth to "
                "\"stepped up\" and send that again."
            ),
            conversation_state=state,
        )
    if "rate_pct" in decision.missing_slots:
        return AssistantTurn(text="What percentage would you like to contribute?", conversation_state=state)

    await graph.call("GetElections")
    await graph.call("GetContributionLimits")
    await graph.call("Calc402g", rate_pct=decision.slots["rate_pct"])
    proposal = await graph.call("ProposeContributionChange", rate_pct=decision.slots["rate_pct"])
    assert isinstance(proposal, Proposal)
    pending = store.register(
        proposal, tenant_id=context.tenant_id, participant_ref=context.participant_ref
    )
    state["pending_proposal"] = pending.summary()
    return AssistantTurn(
        text=(
            f"Here's the change, ready for your confirmation: contribution to "
            f"{decision.slots['rate_pct']:g}%, effective {proposal.effective_date}. "
            "Nothing happens until you confirm."
        ),
        proposal={
            "action": proposal.action,
            "params": dict(proposal.params),
            "effective_date": proposal.effective_date,
            "proposal_id": proposal.proposal_id,
            "confirmation_nonce": proposal.confirmation_nonce,
            "expires_at": pending.expires_at.isoformat(timespec="seconds"),
        },
        tool_calls=_names(graph),
        conversation_state=state,
    )


async def _commit(
    decision: RoutingDecision, graph, context: RequestContext, store: ConfirmationStore, state
) -> AssistantTurn:
    """Revalidate -> execute -> verify, keyed by the proposal id so retries collapse."""
    held = state.get("pending_proposal") or {}
    proposal_id = held.get("proposal_id")
    pending = store.get(proposal_id) if proposal_id else None
    if pending is None:
        replay = store.replay(proposal_id) if proposal_id else None
        state.pop("pending_proposal", None)
        if replay is not None:
            return AssistantTurn(text=_receipt_text(replay), receipt=_receipt(replay), conversation_state=state)
        return AssistantTurn(text=NOTHING_TO_CONFIRM, conversation_state=state)

    try:
        claimed = store.claim(
            pending.proposal_id,
            pending.confirmation_nonce,
            tenant_id=context.tenant_id,
            participant_ref=context.participant_ref,
        )
    except ConfirmationError as exc:
        state.pop("pending_proposal", None)
        return AssistantTurn(text=str(exc), conversation_state=state)

    return await execute_pending(claimed, graph, store, state)


async def _autonomous(
    decision: RoutingDecision, graph, context: RequestContext, store: ConfirmationStore, state
) -> AssistantTurn:
    """Rung 3+: act on the reversible change without asking, then notify with the receipt."""
    if "rate_pct" in decision.missing_slots:
        return AssistantTurn(text="What percentage would you like to contribute?", conversation_state=state)
    pending = PendingProposal(
        proposal_id=str(uuid.uuid4()),
        confirmation_nonce="",
        action="ExecuteContributionChange",
        params={"rate_pct": decision.slots["rate_pct"]},
        effective_date="next-payroll-period",
        tenant_id=context.tenant_id,
        participant_ref=context.participant_ref,
        expires_at=datetime.now(timezone.utc),
    )
    return await execute_pending(pending, graph, store, state)


async def execute_pending(pending, graph, store: ConfirmationStore, state) -> AssistantTurn:
    await graph.call("RevalidateProposal", proposal_id=pending.proposal_id)
    receipt = await graph.call(
        "ExecuteContributionChange",
        command_key=pending.proposal_id,
        effective_date=pending.effective_date,
        **pending.params,
    )
    assert isinstance(receipt, Receipt)
    await graph.call("VerifyContributionChange", command_key=receipt.command_key)
    if "EmitReceipt" in graph.spec.manifest:
        await graph.call("EmitReceipt", command_key=receipt.command_key)
    else:
        await graph.call("NotifyParticipant", command_key=receipt.command_key)
    store.record(pending, receipt)
    state.pop("pending_proposal", None)
    return AssistantTurn(
        text=_receipt_text(receipt),
        receipt=_receipt(receipt),
        tool_calls=_names(graph),
        conversation_state=state,
    )


def _receipt_text(receipt: Receipt) -> str:
    prefix = "That one was already done" if receipt.duplicate else "Done"
    rate = receipt.params.get("rate_pct")
    amount = f" to {rate:g}%" if isinstance(rate, (int, float)) else ""
    return f"{prefix}: your contribution{amount} takes effect {receipt.effective_date}. {receipt.reversal}"


def _receipt(receipt: Receipt) -> Dict[str, Any]:
    return {
        "command_key": receipt.command_key,
        "action": receipt.action,
        "params": dict(receipt.params),
        "effective_date": receipt.effective_date,
        "executed_at": receipt.executed_at,
        "reversal": receipt.reversal,
        "duplicate": receipt.duplicate,
    }


def _names(graph) -> List[str]:
    return [tool for tool, _ in graph.calls]
