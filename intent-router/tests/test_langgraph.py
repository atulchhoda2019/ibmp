"""LangGraph execution of the approved graphs, including the confirmation interrupt."""
from __future__ import annotations

import asyncio
import uuid

import pytest

from app.corridor import ConfirmationStore
from errors import ConfigError
from graphs.langgraph_backend import (
    CORRIDOR_NODES,
    LangGraphGraph,
    build_corridor,
    build_tool_graph,
    pending_interrupt,
    resume_confirmation,
)
from graphs.registry import Proposal, Receipt


@pytest.fixture
def corridor(registry):
    return build_corridor(registry.get("G-CONTRIB-CHANGE"), registry.get("G-CONTRIB-COMMIT"))


def thread():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def test_the_compiled_node_set_is_exactly_the_manifest(registry):
    spec = registry.get("G-DEDUCTIBLE")
    compiled = build_tool_graph(spec)
    assert set(compiled.get_graph().nodes) - {"__start__", "__end__"} == set(spec.manifest)


def test_a_manifest_tool_runs_as_a_langgraph_node(registry):
    graph = LangGraphGraph(spec=registry.get("G-DEDUCTIBLE"), budgets={"steps": 4})
    result = asyncio.run(graph.call("GetAccumulators", plan_id="PPO-High"))
    assert result == {"tool": "GetAccumulators", "params": {"plan_id": "PPO-High"}}
    assert graph.calls == [("GetAccumulators", {"plan_id": "PPO-High"})]


def test_an_off_manifest_tool_has_no_node_to_run_in(registry):
    graph = LangGraphGraph(spec=registry.get("G-DEDUCTIBLE"))
    with pytest.raises(ConfigError, match="not in the manifest"):
        asyncio.run(graph.call("ExecuteContributionChange", command_key="x"))


def test_the_backend_switch_keeps_the_stub_interface(registry, monkeypatch):
    monkeypatch.setenv("GRAPH_BACKEND", "langgraph")
    graph = registry.instantiate("G-ELECTIONS-RO", budgets={"steps": 3})
    assert isinstance(graph, LangGraphGraph)
    asyncio.run(graph.call("GetElections"))
    monkeypatch.setenv("GRAPH_BACKEND", "stub")
    assert not isinstance(registry.instantiate("G-ELECTIONS-RO"), LangGraphGraph)


def test_the_corridor_nodes_are_the_boards_node_names(corridor):
    compiled = set(corridor.get_graph().nodes) - {"__start__", "__end__"}
    assert compiled == set(CORRIDOR_NODES)


def test_wait_confirmation_interrupts_before_anything_executes(corridor):
    config = thread()
    state = corridor.invoke({"rate_pct": 8.0}, config=config)
    interrupted = pending_interrupt(corridor, config["configurable"]["thread_id"])
    assert interrupted is not None and interrupted["action"] == "ProposeContributionChange"
    assert "ExecuteContributionChange" not in (state.get("calls") or [])


def test_the_corridor_nonce_resumes_the_same_run(corridor):
    config = thread()
    corridor.invoke({"rate_pct": 8.0}, config=config)
    thread_id = config["configurable"]["thread_id"]
    proposal = corridor.get_state(config).values["proposal"]
    assert isinstance(proposal, Proposal)

    store = ConfirmationStore()
    pending = store.register(proposal, tenant_id="tenant-acme", participant_ref="participant-1")
    final = resume_confirmation(
        corridor,
        thread_id,
        proposal_id=pending.proposal_id,
        confirmation_nonce=pending.confirmation_nonce,
    )
    receipt = final["receipt"]
    assert isinstance(receipt, Receipt) and receipt.command_key == proposal.proposal_id
    assert final["calls"][-3:] == [
        "ExecuteContributionChange", "VerifyContributionChange", "EmitReceipt"
    ]


def test_a_wrong_nonce_never_resumes_the_run(corridor):
    config = thread()
    corridor.invoke({"rate_pct": 8.0}, config=config)
    proposal = corridor.get_state(config).values["proposal"]
    with pytest.raises(ConfigError, match="nonce does not match"):
        resume_confirmation(
            corridor,
            config["configurable"]["thread_id"],
            proposal_id=proposal.proposal_id,
            confirmation_nonce="not-the-nonce",
        )


def test_a_confirmation_for_another_proposal_never_resumes_this_one(corridor):
    config = thread()
    corridor.invoke({"rate_pct": 8.0}, config=config)
    proposal = corridor.get_state(config).values["proposal"]
    with pytest.raises(ConfigError, match="different proposal"):
        resume_confirmation(
            corridor,
            config["configurable"]["thread_id"],
            proposal_id=str(uuid.uuid4()),
            confirmation_nonce=proposal.confirmation_nonce,
        )
