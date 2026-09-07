"""Versioned decision table: ordered rows, first match wins, routes but never adjudicates."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import yaml

from .catalog import IntentSpec
from .models import Band, ConfigError, RequestContext

CAPABILITY_EXPR = re.compile(r"^enabled&rung>=(?P<rung>\d+)$")

#: capability states the engine derives from context; never from the utterance
FROZEN = "FROZEN"
DISABLED = "disabled"
ENABLED = "enabled"
UNENTITLED = "unentitled"
UNSCOPED = "any"


@dataclass(frozen=True)
class Row:
    index: int
    intent: str
    band: str
    capability: str
    risk: str
    auth: str
    graph: str
    budgets: Mapping[str, int]
    entry_node: Optional[str] = None
    note: Optional[str] = None
    log_for_catalog_review: bool = False
    no_data_reads: bool = False
    evidence: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class DecisionTable:
    version: str
    catalog_version: str
    bands: Mapping[str, Sequence[float]]
    rows: Sequence[Row] = field(default_factory=tuple)

    def band_for(self, confidence: float, edges: Optional[Mapping[str, Sequence[float]]] = None) -> Band:
        """Band edges are versioned with routing, and per-intent edges override the global ones."""
        table = edges or self.bands
        for name in ("HIGH", "MEDIUM", "LOW"):
            low, high = table[name]
            upper_inclusive = name == "HIGH"
            if low <= confidence and (confidence <= high if upper_inclusive else confidence < high):
                return name  # type: ignore[return-value]
        raise ConfigError(f"confidence {confidence} falls outside the declared bands")

    def match(self, *, intent: str, band: Band, capability: str, risk: str, auth: str) -> Row:
        for row in self.rows:
            if _matches(row, intent=intent, band=band, capability=capability, risk=risk, auth=auth):
                return row
        raise ConfigError(
            f"no row matched (intent={intent}, band={band}, capability={capability}, risk={risk}, auth={auth})"
        )


def _matches(row: Row, *, intent: str, band: Band, capability: str, risk: str, auth: str) -> bool:
    return (
        _col(row.intent, intent)
        and _band_matches(row.band, band)
        and _capability_matches(row.capability, capability)
        and _col(row.risk, risk)
        and _col(row.auth, auth)
    )


def _col(pattern: str, value: str) -> bool:
    return pattern == "any" or pattern == value


def _band_matches(pattern: str, band: Band) -> bool:
    if pattern == "any":
        return True
    if pattern == "RULE_OR_HIGH":
        return band in ("RULE", "HIGH")
    return pattern == band


def _capability_matches(pattern: str, capability: str) -> bool:
    """`capability` is the state derived from tenant config: FROZEN, disabled, or enabled&rung>=N."""
    if pattern == "any":
        return True
    if pattern in (FROZEN, DISABLED, ENABLED, UNENTITLED):
        return capability == pattern
    expr = CAPABILITY_EXPR.match(pattern)
    if expr is None:
        raise ConfigError(f"unsupported capability expression {pattern!r}")
    actual = CAPABILITY_EXPR.match(capability)
    if actual is None:
        return False
    return int(actual.group("rung")) >= int(expr.group("rung"))


def capability_state(intent: IntentSpec, context: RequestContext) -> str:
    """Freeze outranks everything for write-capable tiers; reads are unaffected.

    Entitlement is checked here, before any graph runs and therefore before any data fetch:
    VIEW and CHANGE are separate grants and neither is inferred from the other.
    """
    if context.tenant_frozen and intent.risk_tier == "TRANSACT":
        return FROZEN
    if not context.entitlements.get(intent.permission, False):
        return UNENTITLED
    capability = context.tenant_capabilities.get(intent.capability_id)
    if capability is None:
        return UNSCOPED
    if not capability.get("enabled", False):
        return DISABLED
    rung = capability.get("rung")
    return ENABLED if rung is None else f"enabled&rung>={rung}"


def capability_rung(capability: str) -> int:
    """Autonomy rung behind a capability state; 0 when the tenant bought no autonomy."""
    match = CAPABILITY_EXPR.match(capability)
    return int(match.group("rung")) if match else 0


def load_table(path: Path) -> DecisionTable:
    raw = yaml.safe_load(Path(path).read_text())
    rows = []
    for index, entry in enumerate(raw["rows"]):
        unknown = set(entry) - {
            "intent", "band", "capability", "risk", "auth", "graph", "budgets",
            "entry_node", "note", "log_for_catalog_review", "no_data_reads", "evidence",
        }
        if unknown:
            raise ConfigError(f"row {index}: unknown keys {sorted(unknown)}")
        rows.append(
            Row(
                index=index,
                intent=entry["intent"],
                band=entry["band"],
                capability=entry["capability"],
                risk=entry["risk"],
                auth=entry["auth"],
                graph=entry["graph"],
                budgets=dict(entry["budgets"]),
                entry_node=entry.get("entry_node"),
                note=entry.get("note"),
                log_for_catalog_review=bool(entry.get("log_for_catalog_review", False)),
                no_data_reads=bool(entry.get("no_data_reads", False)),
                evidence=dict(entry["evidence"]) if entry.get("evidence") else None,
            )
        )
    return DecisionTable(
        version=raw["version"],
        catalog_version=raw["catalog_version"],
        bands={k: tuple(v) for k, v in raw["bands"].items()},
        rows=tuple(rows),
    )
