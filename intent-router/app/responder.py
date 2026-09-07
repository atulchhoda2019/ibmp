"""Presentation layer for the demo assistant.

The planner picks a graph; this module runs that graph's *stub* and turns the result into
an assistant turn. It contains no routing logic and no business logic: the questions it
asks are the ones the decision table told it to ask.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graphs.registry import GraphRegistry, Proposal
from router.models import RoutingDecision

READ_ANSWERS = {
    "plan.deductible.status": (
        "You've met $840 of your $1,500 in-network deductible on {plan_id}. "
        "That's from claims processed through last week."
    ),
    "plan.deductible.info": "The {plan_id} plan has a $1,500 in-network individual deductible.",
}

EXPLAIN_ROUTE_FROZEN = (
    "Contribution changes are paused right now during open enrollment, so I can't start one. "
    "Here's what will happen when the window reopens, and I can hand you to a specialist in the meantime."
)
EXPLAIN_ROUTE_DISABLED = (
    "Your plan doesn't let me change contributions from here. I can explain how the change works "
    "and connect you with someone who can make it for you."
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


@dataclass
class AssistantTurn:
    text: str
    options: List[Dict[str, str]] = field(default_factory=list)
    proposal: Optional[Dict[str, Any]] = None
    tool_calls: List[str] = field(default_factory=list)


async def respond(decision: RoutingDecision, registry: GraphRegistry) -> AssistantTurn:
    graph = registry.instantiate(decision.graph)

    if decision.graph == "G-CLARIFY":
        question = decision.clarifying_question or {}
        return AssistantTurn(
            text=question.get("text", "Could you say a bit more?"),
            options=[{"label": label, "intent": intent} for label, intent in question.get("options", [])],
        )

    if decision.graph == "G-DEDUCTIBLE":
        await graph.call("GetCoverage", plan_id=decision.slots.get("plan_id"))
        await graph.call("GetAccumulators", plan_id=decision.slots.get("plan_id"))
        template = READ_ANSWERS[decision.intent]
        text = template.format(plan_id=decision.slots.get("plan_id", "your plan"))
        return AssistantTurn(text=text, tool_calls=_names(graph))

    if decision.graph == "G-CONTRIB-CHANGE":
        return await _contribution_change(decision, graph)

    if decision.graph == "G-EXPLAIN-ROUTE":
        await graph.call("RetrieveEvidence", topic=decision.intent)
        frozen = decision.note == "writes-disabled-during-freeze"
        return AssistantTurn(
            text=EXPLAIN_ROUTE_FROZEN if frozen else EXPLAIN_ROUTE_DISABLED,
            tool_calls=_names(graph),
        )

    if decision.graph == "G-LIFE-EVENT-RO":
        await graph.call("GetCoverage")
        await graph.call("RetrieveEvidence", topic=decision.intent)
        return AssistantTurn(text=LIFE_EVENT, tool_calls=_names(graph))

    await graph.call("CsrHandoff", payload=decision.handoff_payload)
    return AssistantTurn(text=FALLBACK, tool_calls=_names(graph))


async def _contribution_change(decision: RoutingDecision, graph) -> AssistantTurn:
    if decision.entry_node == "step_up_auth":
        return AssistantTurn(
            text=(
                "I can set that up. First I need you to verify it's you — switch auth to "
                "\"stepped up\" and send that again."
            )
        )
    if "rate_pct" in decision.missing_slots:
        return AssistantTurn(text="What percentage would you like to contribute?")

    await graph.call("GetElections")
    await graph.call("GetContributionLimits")
    await graph.call("Calc402g", rate_pct=decision.slots["rate_pct"])
    proposal = await graph.call("ProposeContributionChange", rate_pct=decision.slots["rate_pct"])
    assert isinstance(proposal, Proposal)
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
        },
        tool_calls=_names(graph),
    )


def _names(graph) -> List[str]:
    return [tool for tool, _ in graph.calls]
