#!/usr/bin/env python3
"""Config validation gate (design spec §7). Non-zero exit fails the build."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphs.registry import GraphRegistry, load_registry  # noqa: E402
from router.catalog import Catalog, load_catalog  # noqa: E402
from router.models import ConfigError  # noqa: E402
from router.table import DecisionTable, load_table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = (ROOT / "catalog" / "v1.yaml", ROOT / "table" / "v1.yaml", ROOT / "graphs" / "registry.yaml")

WRITE_SAFE_BANDS = ("RULE_OR_HIGH", "HIGH", "RULE")


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

    errors.extend(_validate_guards(table))
    errors.extend(_validate_slot_coverage(catalog, table))
    return errors


def _validate_guards(table: DecisionTable) -> List[str]:
    """4. LOW and FROZEN guards are present and precede every normal row."""
    if len(table.rows) < 2:
        return ["table must start with the LOW and FROZEN guard rows"]
    guards = table.rows[:2]
    errors = []
    if not any(row.band == "LOW" and row.intent == "any" for row in guards):
        errors.append("missing LOW guard row among the first two rows")
    if not any(row.capability == "FROZEN" for row in guards):
        errors.append("missing FROZEN guard row among the first two rows")
    for row in table.rows[2:]:
        if row.band == "LOW" and row.intent == "any":
            errors.append(f"row {row.index}: LOW guard must precede normal rows")
        if row.capability == "FROZEN":
            errors.append(f"row {row.index}: FROZEN guard must precede normal rows")
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
    args = parser.parse_args(argv)

    try:
        errors = validate(load_catalog(args.catalog), load_table(args.table), load_registry(args.registry))
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
