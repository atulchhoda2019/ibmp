"""Model routing table: which served config answers this request.

Ordered rows, first match wins, `any` is a wildcard — the same shape as the intent decision table,
for the same reason: the model does not get to pick the model. A tenant-scoped row is only legal
when its config's eval shows a measured gap over the broader scope, so the adapter ladder cannot
grow a per-tenant adapter that nobody proved was needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import yaml

from errors import ConfigError

from .registry import ModelRegistry, ServedConfig

WILDCARD = "any"


@dataclass(frozen=True)
class ModelRow:
    index: int
    task: str
    domain: str
    posture: str
    tenant: str
    config: str

    def matches(self, *, task: str, domain: str, posture: str, tenant: str) -> bool:
        return all(
            cell in (WILDCARD, value)
            for cell, value in (
                (self.task, task),
                (self.domain, domain),
                (self.posture, posture),
                (self.tenant, tenant),
            )
        )


@dataclass(frozen=True)
class ModelRoutingTable:
    version: str
    model_registry: str
    rows: Sequence[ModelRow]

    def select(
        self, *, task: str, domain: str, posture: str, tenant: str
    ) -> ModelRow:
        for row in self.rows:
            if row.matches(task=task, domain=domain, posture=posture, tenant=tenant):
                return row
        raise ConfigError(
            f"no model row for task={task} domain={domain} posture={posture} tenant={tenant}"
        )


@dataclass(frozen=True)
class ModelChoice:
    config: ServedConfig
    row: ModelRow

    @property
    def rung_of_ladder(self) -> str:
        return self.config.scope


def load_model_routing(path: Path) -> ModelRoutingTable:
    raw = yaml.safe_load(Path(path).read_text())
    rows = tuple(
        ModelRow(
            index=index,
            task=str(entry.get("task", WILDCARD)),
            domain=str(entry.get("domain", WILDCARD)),
            posture=str(entry.get("posture", WILDCARD)),
            tenant=str(entry.get("tenant", WILDCARD)),
            config=str(entry["config"]),
        )
        for index, entry in enumerate(raw["rows"])
    )
    return ModelRoutingTable(
        version=str(raw["version"]), model_registry=str(raw["model_registry"]), rows=rows
    )


def choose(
    registry: ModelRegistry,
    table: ModelRoutingTable,
    *,
    task: str,
    domain: str,
    posture: str,
    tenant: str,
) -> ModelChoice:
    row = table.select(task=task, domain=domain, posture=posture, tenant=tenant)
    config = registry.get(row.config)
    if config.task != task:
        raise ConfigError(f"model row {row.index} routes {task} to a {config.task} config")
    if row.tenant != WILDCARD and config.eval.measured_gap <= 0:
        raise ConfigError(
            f"model row {row.index} pins {config.name} to {row.tenant} without a measured gap"
        )
    return ModelChoice(config=config, row=row)


def assert_versions(registry: ModelRegistry, table: ModelRoutingTable) -> None:
    if table.model_registry != registry.version:
        raise ConfigError(
            f"routing {table.version} declares registry {table.model_registry}, got {registry.version}"
        )


def unreachable_configs(
    registry: ModelRegistry, table: ModelRoutingTable
) -> Tuple[str, ...]:
    """Configs nothing can route to are dead weight in a plane that promises tested serving."""
    reachable = {row.config for row in table.rows}
    reachable.add(registry.frontier)
    reachable.update(pointer.config for pointer in registry.serving.values())
    reachable.update(pointer.rollback_to for pointer in registry.serving.values())
    return tuple(sorted(set(registry.served) - reachable))


def routing_for(
    registry: ModelRegistry, table: ModelRoutingTable, task: str
) -> Optional[Mapping[str, str]]:
    pointer = registry.serving.get(task)
    if pointer is None:
        return None
    return {"config": pointer.config, "rollback_to": pointer.rollback_to}
