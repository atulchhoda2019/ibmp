"""Graph registry: pre-approved graphs with declared tool manifests.

The graphs are stubs, but the interfaces are honest: every tool is async and typed, and
no manifest contains an `Execute*` tool. Transactional graphs end at a typed proposal.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml

from errors import ConfigError

READ_ONLY_TOOLS = (
    "GetCoverage", "GetAccumulators", "RetrieveEvidence", "GetElections",
    "GetContributionLimits", "Calc402g", "CsrHandoff",
)
WRITE_TOOLS = ("ProposeContributionChange",)


@dataclass(frozen=True)
class Proposal:
    """Typed proposal: requires a nonce-bound confirmation before anything executes."""

    action: str
    params: Mapping[str, Any]
    effective_date: str
    proposal_id: str
    confirmation_nonce: str


@dataclass(frozen=True)
class GraphSpec:
    name: str
    manifest: Sequence[str]
    writes: Sequence[str]

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
        if tool in WRITE_TOOLS:
            return Proposal(
                action=tool,
                params=dict(params),
                effective_date="next-payroll-period",
                proposal_id=str(uuid.uuid4()),
                confirmation_nonce=str(uuid.uuid4()),
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

    def assert_manifest_legal(self, name: str, risk: str, capability_enabled: bool) -> None:
        """A selected graph's manifest must be legal for the row's risk tier."""
        spec = self.get(name)
        for tool in spec.manifest:
            if tool.startswith("Execute"):
                raise ConfigError(f"{name}: execute tools are never legal in a conversational graph")
            if tool not in READ_ONLY_TOOLS + WRITE_TOOLS:
                raise ConfigError(f"{name}: unknown tool {tool!r}")
        if spec.writes and not (risk == "TRANSACT" and capability_enabled):
            raise ConfigError(f"{name}: write-capable graph selected for risk={risk}")


def load_registry(path: Path) -> GraphRegistry:
    raw = yaml.safe_load(Path(path).read_text())
    graphs: Dict[str, GraphSpec] = {}
    for name, entry in raw["graphs"].items():
        writes = tuple(entry.get("writes", ()))
        manifest = tuple(entry.get("manifest", ()))
        missing = set(writes) - set(manifest)
        if missing:
            raise ConfigError(f"{name}: writes {sorted(missing)} absent from the manifest")
        graphs[name] = GraphSpec(name=name, manifest=manifest, writes=writes)
    return GraphRegistry(graphs=graphs)
