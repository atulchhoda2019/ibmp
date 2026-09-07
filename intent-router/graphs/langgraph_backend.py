"""LangGraph execution of the approved graphs.

The graphs are compiled *from* the registry manifest, never assembled at request time: the set of
nodes a request can reach is the set of tools the graph was approved for, so an off-manifest tool
has no node to run in. `wait_confirmation` is a real LangGraph interrupt with a checkpointer, so
the corridor's nonce resumes the same run instead of opening a second transaction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from errors import ConfigError

from .registry import GraphSpec, Proposal, Receipt, run_tool

#: board §9 journey, transaction branch
CORRIDOR_NODES = (
    "extract_action",
    "policy_check",
    "validate_action",
    "create_preview",
    "wait_confirmation",
    "revalidate",
    "execute",
    "verify",
    "receipt",
)
DEFAULT_RECURSION_LIMIT = 25


class ToolState(TypedDict, total=False):
    tool: str
    params: Dict[str, Any]
    result: Any


class CorridorState(TypedDict, total=False):
    rate_pct: float
    tenant_id: str
    participant_ref: str
    proposal: Proposal
    confirmation: Dict[str, Any]
    receipt: Receipt
    calls: List[str]


def _tool_node(spec: GraphSpec, tool: str):
    def node(state: ToolState) -> ToolState:
        return {"result": run_tool(spec, tool, **state.get("params", {}))}

    return node


def build_tool_graph(spec: GraphSpec):
    """One node per manifest tool; the manifest *is* the reachable node set."""
    if not spec.manifest:
        return None
    builder = StateGraph(ToolState)
    for tool in spec.manifest:
        builder.add_node(tool, _tool_node(spec, tool))
        builder.add_edge(tool, END)
    builder.add_conditional_edges(
        START, lambda state: state["tool"], {tool: tool for tool in spec.manifest}
    )
    return builder.compile()


@dataclass
class LangGraphGraph:
    """Same surface as `StubGraph`, but every tool call runs as a compiled LangGraph node."""

    spec: GraphSpec
    budgets: Mapping[str, int] = field(default_factory=dict)
    calls: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self._compiled = build_tool_graph(self.spec)

    async def call(self, tool: str, **params: Any) -> Any:
        if self._compiled is None or tool not in self.spec.manifest:
            raise ConfigError(f"{tool} is not in the manifest of {self.spec.name}")
        state = await self._compiled.ainvoke(
            {"tool": tool, "params": dict(params)},
            config={"recursion_limit": self._recursion_limit()},
        )
        self.calls.append((tool, params))
        return state["result"]

    def _recursion_limit(self) -> int:
        """A budget is a runtime limit, not a comment: hops become the recursion limit."""
        return int(self.budgets.get("hops", DEFAULT_RECURSION_LIMIT)) or DEFAULT_RECURSION_LIMIT


def build_corridor(propose: GraphSpec, commit: GraphSpec):
    """Compile the transaction corridor: preview, interrupt for confirmation, then execute."""
    for spec, tools in ((propose, propose.manifest), (commit, commit.manifest)):
        for tool in tools:
            if tool not in spec.manifest:  # pragma: no cover - defensive
                raise ConfigError(f"{tool} is not in the manifest of {spec.name}")

    def extract_action(state: CorridorState) -> CorridorState:
        run_tool(propose, "GetElections")
        return {"calls": ["GetElections"]}

    def policy_check(state: CorridorState) -> CorridorState:
        run_tool(propose, "GetContributionLimits")
        return {"calls": (state.get("calls") or []) + ["GetContributionLimits"]}

    def validate_action(state: CorridorState) -> CorridorState:
        run_tool(propose, "Calc402g", rate_pct=state["rate_pct"])
        return {"calls": (state.get("calls") or []) + ["Calc402g"]}

    def create_preview(state: CorridorState) -> CorridorState:
        proposal = run_tool(
            propose, "ProposeContributionChange", rate_pct=state["rate_pct"]
        )
        return {
            "proposal": proposal,
            "calls": (state.get("calls") or []) + ["ProposeContributionChange"],
        }

    def wait_confirmation(state: CorridorState) -> CorridorState:
        proposal = state["proposal"]
        answer = interrupt(
            {
                "proposal_id": proposal.proposal_id,
                "action": proposal.action,
                "params": dict(proposal.params),
                "effective_date": proposal.effective_date,
            }
        )
        if not isinstance(answer, Mapping):
            raise ConfigError("a resumption must carry the proposal id and its nonce")
        if answer.get("proposal_id") != proposal.proposal_id:
            raise ConfigError("resumption is for a different proposal")
        if answer.get("confirmation_nonce") != proposal.confirmation_nonce:
            raise ConfigError("resumption nonce does not match the proposal")
        return {"confirmation": dict(answer)}

    def revalidate(state: CorridorState) -> CorridorState:
        run_tool(commit, "RevalidateProposal", proposal_id=state["proposal"].proposal_id)
        return {"calls": (state.get("calls") or []) + ["RevalidateProposal"]}

    def execute(state: CorridorState) -> CorridorState:
        proposal = state["proposal"]
        receipt = run_tool(
            commit,
            "ExecuteContributionChange",
            command_key=proposal.proposal_id,
            effective_date=proposal.effective_date,
            **proposal.params,
        )
        return {
            "receipt": receipt,
            "calls": (state.get("calls") or []) + ["ExecuteContributionChange"],
        }

    def verify(state: CorridorState) -> CorridorState:
        run_tool(
            commit, "VerifyContributionChange", command_key=state["receipt"].command_key
        )
        return {"calls": (state.get("calls") or []) + ["VerifyContributionChange"]}

    def receipt(state: CorridorState) -> CorridorState:
        tool = "EmitReceipt" if "EmitReceipt" in commit.manifest else "NotifyParticipant"
        run_tool(commit, tool, command_key=state["receipt"].command_key)
        return {"calls": (state.get("calls") or []) + [tool]}

    implementations = {
        "extract_action": extract_action,
        "policy_check": policy_check,
        "validate_action": validate_action,
        "create_preview": create_preview,
        "wait_confirmation": wait_confirmation,
        "revalidate": revalidate,
        "execute": execute,
        "verify": verify,
        "receipt": receipt,
    }
    builder = StateGraph(CorridorState)
    for name in CORRIDOR_NODES:
        builder.add_node(name, implementations[name])
    builder.add_edge(START, CORRIDOR_NODES[0])
    for previous, following in zip(CORRIDOR_NODES, CORRIDOR_NODES[1:]):
        builder.add_edge(previous, following)
    builder.add_edge(CORRIDOR_NODES[-1], END)
    return builder.compile(checkpointer=MemorySaver())


def resume_confirmation(
    corridor, thread_id: str, *, proposal_id: str, confirmation_nonce: str
) -> Dict[str, Any]:
    """Resume the interrupted run this proposal belongs to — not a fresh one."""
    return corridor.invoke(
        Command(resume={"proposal_id": proposal_id, "confirmation_nonce": confirmation_nonce}),
        config={"configurable": {"thread_id": thread_id}},
    )


def pending_interrupt(corridor, thread_id: str) -> Optional[Mapping[str, Any]]:
    state = corridor.get_state({"configurable": {"thread_id": thread_id}})
    if not state.interrupts:
        return None
    return state.interrupts[0].value


def board_nodes(spec: GraphSpec) -> Sequence[str]:
    return tuple(spec.nodes)
