"""Graph registry: pre-approved graphs with declared tool manifests.

The graphs are stubs, but the interfaces are honest: every tool is async and typed.
An `Execute*` tool is illegal unless the graph declares it in `executes` together with its
governance — minimum autonomy rung, reversibility, and whether a nonce-bound confirmation
must already exist. Conversational graphs therefore still end at a typed proposal.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from errors import ConfigError

READ_ONLY_TOOLS = (
    "GetCoverage", "GetAccumulators", "RetrieveEvidence", "GetElections",
    "GetContributionLimits", "Calc402g", "CsrHandoff", "DraftForHuman",
    "RevalidateProposal", "VerifyContributionChange", "EmitReceipt", "NotifyParticipant",
)
WRITE_TOOLS = ("ProposeContributionChange", "ExecuteContributionChange")
POSTURES = ("read", "draft", "propose", "execute")


@dataclass(frozen=True)
class Proposal:
    """Typed proposal: requires a nonce-bound confirmation before anything executes."""

    action: str
    params: Mapping[str, Any]
    effective_date: str
    proposal_id: str
    confirmation_nonce: str


@dataclass(frozen=True)
class Receipt:
    """What the participant gets after an execute node: what happened, and how to reverse it."""

    command_key: str
    action: str
    params: Mapping[str, Any]
    effective_date: str
    executed_at: str
    reversal: str
    duplicate: bool = False


@dataclass(frozen=True)
class Governance:
    min_rung: int
    reversible: bool
    requires_confirmation: bool


@dataclass(frozen=True)
class GraphSpec:
    name: str
    manifest: Sequence[str]
    writes: Sequence[str]
    nodes: Sequence[str] = ()
    executes: Sequence[str] = ()
    posture: str = "read"
    governance: Optional[Governance] = None

    @property
    def is_read_only(self) -> bool:
        return not self.writes


@dataclass
class StubGraph:
    """Records tool calls so tests can assert `no_data_reads` rows perform zero reads."""

    spec: GraphSpec
    calls: list = field(default_factory=list)

    async def call(self, tool: str, **params: Any) -> Any:
        if tool not in self.spec.manifest:
            raise ConfigError(f"{tool} is not in the manifest of {self.spec.name}")
        self.calls.append((tool, params))
        if tool == "ProposeContributionChange":
            proposal_id = str(uuid.uuid4())
            return Proposal(
                action=tool,
                params=dict(params),
                effective_date="next-payroll-period",
                proposal_id=proposal_id,
                confirmation_nonce=str(uuid.uuid4()),
            )
        if tool == "RetrieveEvidence":
            # Every retrieval comes back as an envelope: facts are worthless without the
            # citation and the date the citation was effective.
            age_days = int(params.get("age_days", 2))
            effective = date.today() - timedelta(days=age_days)
            return {
                "tool": tool,
                "params": dict(params),
                "facts": {},
                "citations": [
                    {"source": f"summary-plan-description:{params.get('topic', 'general')}",
                     "effective_date": effective.isoformat()}
                ],
            }
        if tool == "ExecuteContributionChange":
            # The command key is the proposal id: a retried command collapses onto one write.
            command_key = params.get("command_key")
            if not command_key:
                raise ConfigError(f"{tool} requires a command_key so retries stay idempotent")
            return Receipt(
                command_key=str(command_key),
                action=tool,
                params={k: v for k, v in params.items() if k not in ("command_key", "effective_date")},
                effective_date=str(params.get("effective_date", "next-payroll-period")),
                executed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                reversal="Reply \u201cundo my contribution change\u201d before the next payroll run.",
            )
        return {"tool": tool, "params": dict(params)}


@dataclass(frozen=True)
class GraphRegistry:
    graphs: Mapping[str, GraphSpec]

    def get(self, name: str) -> GraphSpec:
        try:
            return self.graphs[name]
        except KeyError:
            raise ConfigError(f"unknown graph {name!r}") from None

    def instantiate(self, name: str) -> StubGraph:
        return StubGraph(spec=self.get(name))

    def assert_manifest_legal(
        self,
        name: str,
        risk: str,
        capability_enabled: bool,
        *,
        rung: int = 0,
        confirmed: bool = False,
    ) -> None:
        """A selected graph's manifest must be legal for the row's risk tier and autonomy rung."""
        spec = self.get(name)
        for tool in spec.manifest:
            if tool.startswith("Execute") and tool not in spec.executes:
                raise ConfigError(f"{name}: {tool} is not declared in the graph's executes")
            if tool not in READ_ONLY_TOOLS + WRITE_TOOLS:
                raise ConfigError(f"{name}: unknown tool {tool!r}")
        if spec.writes and not (risk == "TRANSACT" and capability_enabled):
            raise ConfigError(f"{name}: write-capable graph selected for risk={risk}")
        if not spec.executes:
            return
        governance = spec.governance
        if governance is None:
            raise ConfigError(f"{name}: an executing graph must declare its governance")
        if not governance.reversible:
            raise ConfigError(f"{name}: only reversible actions may execute from a conversation")
        if rung < governance.min_rung:
            raise ConfigError(f"{name}: rung {rung} is below the graph's minimum {governance.min_rung}")
        if governance.requires_confirmation and not confirmed:
            raise ConfigError(f"{name}: execution requires a live nonce-bound confirmation")


def _governance(name: str, raw: Optional[Mapping[str, Any]]) -> Optional[Governance]:
    if raw is None:
        return None
    missing = {"min_rung", "reversible", "requires_confirmation"} - set(raw)
    if missing:
        raise ConfigError(f"{name}: governance missing {sorted(missing)}")
    return Governance(
        min_rung=int(raw["min_rung"]),
        reversible=bool(raw["reversible"]),
        requires_confirmation=bool(raw["requires_confirmation"]),
    )


def load_registry(path: Path) -> GraphRegistry:
    raw = yaml.safe_load(Path(path).read_text())
    graphs: Dict[str, GraphSpec] = {}
    for name, entry in raw["graphs"].items():
        writes = tuple(entry.get("writes", ()))
        manifest = tuple(entry.get("manifest", ()))
        executes = tuple(entry.get("executes", ()))
        posture = entry.get("posture", "read")
        if posture not in POSTURES:
            raise ConfigError(f"{name}: unknown posture {posture!r}")
        missing = set(writes) - set(manifest)
        if missing:
            raise ConfigError(f"{name}: writes {sorted(missing)} absent from the manifest")
        stray = set(executes) - set(writes)
        if stray:
            raise ConfigError(f"{name}: executes {sorted(stray)} must also be declared as writes")
        graphs[name] = GraphSpec(
            name=name,
            manifest=manifest,
            writes=writes,
            nodes=tuple(entry.get("nodes", ())),
            executes=executes,
            posture=posture,
            governance=_governance(name, entry.get("governance")),
        )
    return GraphRegistry(graphs=graphs)
