#!/usr/bin/env python3
"""Config validation gate (design spec §7). Non-zero exit fails the build."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphs.registry import GraphRegistry, load_registry  # noqa: E402
from modelplane.registry import ModelRegistry, load_model_registry  # noqa: E402
from modelplane.routing import (  # noqa: E402
    ModelRoutingTable,
    assert_versions,
    choose,
    load_model_routing,
    unreachable_configs,
)
from router.catalog import Catalog, load_catalog  # noqa: E402
from router.models import ConfigError  # noqa: E402
from router.table import DecisionTable, capability_rung, load_table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = (ROOT / "catalog" / "v1.yaml", ROOT / "table" / "v1.yaml", ROOT / "graphs" / "registry.yaml")
MODEL_DEFAULTS = (ROOT / "modelplane" / "registry.yaml", ROOT / "modelplane" / "routing.yaml")
#: every task the plane must be able to serve before a build is allowed out
REQUIRED_MODEL_ROUTES = (
    ("classify", "benefits", "read"),
    ("classify", "retirement", "read"),
    ("compose", "benefits", "read"),
    ("compose", "retirement", "read"),
    ("compose", "retirement", "propose"),
    ("compose", "retirement", "execute"),
)

WRITE_SAFE_BANDS = ("RULE_OR_HIGH", "HIGH", "RULE")
GUARD_CAPABILITIES = ("FROZEN", "unentitled")
BAND_NAMES = ("HIGH", "MEDIUM", "LOW")


def validate(catalog: Catalog, table: DecisionTable, registry: GraphRegistry) -> List[str]:
    errors: List[str] = []

    if table.catalog_version != catalog.version:
        errors.append(f"table declares catalog {table.catalog_version}, loaded {catalog.version}")

    for row in table.rows:
        where = f"row {row.index}"
        # 1. intents exist in the declared catalog version
        if row.intent != "any" and row.intent not in catalog:
            errors.append(f"{where}: unknown intent {row.intent!r} for {catalog.version}")
        # 2. graphs exist in the registry
        if row.graph not in registry.graphs:
            errors.append(f"{where}: unknown graph {row.graph!r}")
            continue
        # 3. write-capable graphs are only reachable from safe rows
        if registry.get(row.graph).writes:
            if row.risk != "TRANSACT":
                errors.append(f"{where}: write graph {row.graph} on risk {row.risk}")
            if row.band not in WRITE_SAFE_BANDS:
                errors.append(f"{where}: write graph {row.graph} on band {row.band}")
            if row.capability in ("disabled", "FROZEN", "any"):
                errors.append(f"{where}: write graph {row.graph} with capability {row.capability}")
        # 6. budgets present and positive
        for key in ("steps", "tokens"):
            value = row.budgets.get(key)
            if not isinstance(value, int) or value <= 0:
                errors.append(f"{where}: budget {key}={value!r} must be a positive int")
        # 7. a row may only enter a graph at one of its declared nodes
        spec = registry.get(row.graph)
        if row.entry_node is not None and spec.nodes and row.entry_node not in spec.nodes:
            errors.append(f"{where}: {row.graph} has no node {row.entry_node!r}")
        # 8. executing graphs are rung-gated by the row that can reach them
        if spec.executes:
            errors.extend(_validate_execute_row(row, spec))
        # 9. a read that answers from evidence declares how stale that evidence may be
        if "RetrieveEvidence" in spec.manifest and row.risk == "READ" and row.evidence is None:
            errors.append(f"{where}: {row.graph} reads evidence but the row declares no evidence policy")

    errors.extend(_validate_manifest_nodes(registry))
    errors.extend(_validate_guards(table))
    errors.extend(_validate_slot_coverage(catalog, table))
    errors.extend(_validate_band_edges(catalog, table))
    return errors


def validate_model_plane(models: ModelRegistry, routing: ModelRoutingTable) -> List[str]:
    """§5 gate: an untested config, an unreachable config or an uncovered route fails the build."""
    errors: List[str] = []
    try:
        assert_versions(models, routing)
    except ConfigError as exc:
        errors.append(str(exc))

    for config in models.served.values():
        if not config.eval.passing:
            errors.append(f"{config.name}: served without a passing eval")
        if config.ring != "prod" and config.name in {
            pointer.config for pointer in models.serving.values()
        }:
            errors.append(f"{config.name}: serving pointer targets a {config.ring} ring config")

    if models.get(models.frontier).adapter is not None:
        errors.append(f"{models.frontier}: the frontier escalation must not carry an adapter")

    for row in routing.rows:
        if row.config not in models.served:
            errors.append(f"model row {row.index}: unknown served config {row.config!r}")

    for task, domain, posture in REQUIRED_MODEL_ROUTES:
        try:
            choose(models, routing, task=task, domain=domain, posture=posture, tenant="any-tenant")
        except ConfigError as exc:
            errors.append(f"model routing: {exc}")

    for name in unreachable_configs(models, routing):
        errors.append(f"{name}: served config no row can reach")
    return errors


def _validate_manifest_nodes(registry: GraphRegistry) -> List[str]:
    """A LangGraph node set is compiled from the manifest, so the manifest must be sane."""
    errors = []
    for spec in registry.graphs.values():
        if len(set(spec.manifest)) != len(spec.manifest):
            errors.append(f"{spec.name}: duplicate tool in the manifest")
        if not spec.nodes:
            errors.append(f"{spec.name}: no declared nodes to enter at")
    return errors


def _validate_execute_row(row, spec) -> List[str]:
    """An execute node is only reachable from a row that bought the rung for it."""
    where = f"row {row.index}"
    errors = []
    governance = spec.governance
    if governance is None:
        return [f"{where}: {spec.name} executes without declared governance"]
    if not governance.reversible:
        errors.append(f"{where}: {spec.name} executes an irreversible action")
    rung = capability_rung(row.capability)
    if rung < governance.min_rung:
        errors.append(
            f"{where}: capability {row.capability!r} cannot reach {spec.name} (needs rung>={governance.min_rung})"
        )
    if row.band not in WRITE_SAFE_BANDS:
        errors.append(f"{where}: execute graph {spec.name} on band {row.band}")
    return errors


def _validate_band_edges(catalog: Catalog, table: DecisionTable) -> List[str]:
    """Global and per-intent edges must partition [0, 1] with no gap and no overlap."""
    errors = _check_edges("table", table.bands)
    for intent in catalog.intents.values():
        if intent.bands is not None:
            errors.extend(_check_edges(intent.id, intent.bands))
    return errors


def _check_edges(where: str, bands) -> List[str]:
    errors = []
    for name in BAND_NAMES:
        low, high = bands[name]
        if not 0.0 <= low < high <= 1.0:
            errors.append(f"{where}: band {name} edges {low}-{high} are not an ascending range in [0, 1]")
    if bands["LOW"][1] != bands["MEDIUM"][0] or bands["MEDIUM"][1] != bands["HIGH"][0]:
        errors.append(f"{where}: band edges leave a gap or overlap")
    if bands["LOW"][0] != 0.0 or bands["HIGH"][1] != 1.0:
        errors.append(f"{where}: band edges must cover 0.0 to 1.0")
    return errors


def _validate_guards(table: DecisionTable) -> List[str]:
    """4. LOW, FROZEN and unentitled guards are present and precede every normal row."""
    guards = table.rows[:3]
    errors = []
    if not any(row.band == "LOW" and row.intent == "any" for row in guards):
        errors.append("missing LOW guard row among the leading guard rows")
    for capability in GUARD_CAPABILITIES:
        if not any(row.capability == capability for row in guards):
            errors.append(f"missing {capability} guard row among the leading guard rows")
    for row in table.rows[3:]:
        if row.band == "LOW" and row.intent == "any":
            errors.append(f"row {row.index}: LOW guard must precede normal rows")
        if row.capability in GUARD_CAPABILITIES:
            errors.append(f"row {row.index}: {row.capability} guard must precede normal rows")
    return errors


def _validate_slot_coverage(catalog: Catalog, table: DecisionTable) -> List[str]:
    """5. Required slots are extractable, or the intent has a clarify path."""
    errors = []
    for intent in catalog.intents.values():
        required = [s for s in intent.slots.values() if s.required]
        if not required:
            continue
        rows = [r for r in table.rows if r.intent == intent.id]
        has_clarify = any(r.band == "MEDIUM" or r.graph == "G-CLARIFY" for r in rows)
        for spec in required:
            extractable = bool(intent.rule_patterns) or set(spec.source) & {"utterance", "ui_context"}
            if not extractable and not has_clarify:
                errors.append(f"{intent.id}: required slot {spec.name} is neither extractable nor clarifiable")
    return errors


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULTS[0])
    parser.add_argument("--table", type=Path, default=DEFAULTS[1])
    parser.add_argument("--registry", type=Path, default=DEFAULTS[2])
    parser.add_argument("--models", type=Path, default=MODEL_DEFAULTS[0])
    parser.add_argument("--model-routing", type=Path, default=MODEL_DEFAULTS[1])
    args = parser.parse_args(argv)

    try:
        errors = validate(load_catalog(args.catalog), load_table(args.table), load_registry(args.registry))
        errors += validate_model_plane(
            load_model_registry(args.models), load_model_routing(args.model_routing)
        )
    except ConfigError as exc:
        errors = [str(exc)]

    for error in errors:
        print(f"config error: {error}", file=sys.stderr)
    if errors:
        return 1
    print("config ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
