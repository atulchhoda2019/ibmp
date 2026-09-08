"""One turn is one graph invocation; thread_id is the conversation id.

Nodes are synchronous on purpose: on Python 3.10 LangGraph cannot carry the runnable
config into async nodes, and `interrupt()` needs it.
"""
import pathlib
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app import escalation
from app.nodes import (assemble, corridor, escalate, gate_output, gate_plan, ingress, planner,
                       read_calc, read_evidence, read_facts, reason, respond)
from app.state import TurnState


def route_after_planner(state: TurnState) -> str:
    if state.response is not None:      # LOW handoff or a clarifying question
        return "respond"                # a clarify turn reads nothing at all
    return "gate_plan"


READS = ["read_evidence", "read_facts", "read_calc"]


def route_after_gate_plan(state: TurnState) -> str | list[str]:
    return "respond" if state.response is not None else READS


def route_after_gate_output(state: TurnState) -> str:
    if not (state.validation or {}).get("passed"):
        if state.retries.get("reason", 0) < 2:
            return "reason"             # one bounded retry
        if escalates(state):
            return "escalate"           # then the frontier, once
        return "respond"                # scripted fallback, i.e. the human
    if state.plan.posture == "WRITE":
        return "corridor_propose"
    return "respond"


def escalates(state: TurnState) -> bool:
    """Advisory reads only: a WRITE turn goes to the human rather than to a frontier model."""
    return (
        state.retries.get("escalate", 0) == 0
        and state.plan.posture == "READ"
        and bool(state.envelope)
        and escalation.available()
    )


def route_after_propose(state: TurnState) -> str:
    if state.proposal is None:          # validation failed or a slot is missing
        return "respond"
    return "corridor_preview"


def route_after_preview(state: TurnState) -> str:
    # Rung 2 pauses for a human nonce; rung 3 and 4 keep every gate except the pause.
    return "corridor_wait_confirmation" if state.plan.rung <= 2 else "corridor_revalidate"


def route_after_revalidate(state: TurnState) -> str:
    return "corridor_execute" if state.confirmed or state.plan.rung >= 3 else "respond"


def build_graph(checkpoint_path: str | None = "checkpoints.sqlite"):
    g = StateGraph(TurnState)
    g.add_node("ingress", ingress.run)
    g.add_node("planner", planner.run)
    g.add_node("gate_plan", gate_plan.run)
    g.add_node("read_evidence", read_evidence.run)
    g.add_node("read_facts", read_facts.run)
    g.add_node("read_calc", read_calc.run)
    g.add_node("assemble", assemble.run)
    g.add_node("reason", reason.run)
    g.add_node("gate_output", gate_output.run)
    g.add_node("escalate", escalate.run)
    g.add_node("corridor_propose", corridor.propose)
    g.add_node("corridor_preview", corridor.preview)
    g.add_node("corridor_wait_confirmation", corridor.wait_confirmation)
    g.add_node("corridor_revalidate", corridor.revalidate)
    g.add_node("corridor_execute", corridor.execute)
    g.add_node("corridor_verify", corridor.verify)
    g.add_node("respond", respond.run)

    g.add_edge(START, "ingress")
    g.add_edge("ingress", "planner")
    g.add_conditional_edges("planner", route_after_planner,
                            {"gate_plan": "gate_plan", "respond": "respond"})
    g.add_conditional_edges("gate_plan", route_after_gate_plan, [*READS, "respond"])
    g.add_edge("read_evidence", "assemble")
    g.add_edge("read_facts", "assemble")
    g.add_edge("read_calc", "assemble")
    g.add_edge("assemble", "reason")
    g.add_edge("reason", "gate_output")
    g.add_conditional_edges("gate_output", route_after_gate_output,
                            {"reason": "reason",
                             "escalate": "escalate",
                             "corridor_propose": "corridor_propose",
                             "respond": "respond"})
    g.add_edge("escalate", "gate_output")
    g.add_conditional_edges("corridor_propose", route_after_propose,
                            {"corridor_preview": "corridor_preview", "respond": "respond"})
    g.add_conditional_edges("corridor_preview", route_after_preview,
                            {"corridor_wait_confirmation": "corridor_wait_confirmation",
                             "corridor_revalidate": "corridor_revalidate"})
    g.add_edge("corridor_wait_confirmation", "corridor_revalidate")
    g.add_conditional_edges("corridor_revalidate", route_after_revalidate,
                            {"corridor_execute": "corridor_execute", "respond": "respond"})
    g.add_edge("corridor_execute", "corridor_verify")
    g.add_edge("corridor_verify", "respond")
    g.add_edge("respond", END)

    return g.compile(checkpointer=checkpointer(checkpoint_path))


def checkpointer(checkpoint_path: str | None):
    """SqliteSaver survives a process restart, which is what the kill/resume drill needs."""
    if checkpoint_path is None:
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    path = pathlib.Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
