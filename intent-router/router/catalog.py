"""Versioned intent catalog: closed set of intents, slot schemas, stage 0 rules."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from .models import ConfigError, RiskTier

RISK_TIERS = ("READ", "TRANSACT", "SENSITIVE")


@dataclass(frozen=True)
class SlotSpec:
    name: str
    type: str
    required: bool = False
    source: Sequence[str] = ("utterance",)
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass(frozen=True)
class IntentSpec:
    id: str
    risk_tier: RiskTier
    label: str
    personal: bool = True
    scope: Optional[str] = None
    slots: Mapping[str, SlotSpec] = field(default_factory=dict)
    rule_patterns: Sequence[re.Pattern] = ()


@dataclass(frozen=True)
class Catalog:
    version: str
    intents: Mapping[str, IntentSpec]

    def __contains__(self, intent_id: str) -> bool:
        return intent_id in self.intents

    def get(self, intent_id: str) -> IntentSpec:
        try:
            return self.intents[intent_id]
        except KeyError:
            raise ConfigError(f"unknown intent {intent_id!r} for catalog {self.version}") from None

    def label(self, intent_id: str) -> str:
        return self.get(intent_id).label

    def with_version(self, version: str) -> "Catalog":
        return replace(self, version=version)


def _slot(name: str, raw: Mapping[str, Any]) -> SlotSpec:
    return SlotSpec(
        name=name,
        type=raw["type"],
        required=bool(raw.get("required", False)),
        source=tuple(raw.get("source", ("utterance",))),
        min=raw.get("min"),
        max=raw.get("max"),
    )


def load_catalog(path: Path) -> Catalog:
    raw = yaml.safe_load(Path(path).read_text())
    intents: dict[str, IntentSpec] = {}
    for entry in raw["intents"]:
        intent_id = entry["id"]
        if entry["risk_tier"] not in RISK_TIERS:
            raise ConfigError(f"{intent_id}: unknown risk tier {entry['risk_tier']!r}")
        if intent_id in intents:
            raise ConfigError(f"duplicate intent {intent_id!r}")
        intents[intent_id] = IntentSpec(
            id=intent_id,
            risk_tier=entry["risk_tier"],
            label=entry.get("label", intent_id),
            personal=bool(entry.get("personal", True)),
            scope=entry.get("scope"),
            slots={n: _slot(n, s) for n, s in (entry.get("slots") or {}).items()},
            rule_patterns=tuple(re.compile(p, re.IGNORECASE) for p in entry.get("rule_patterns", ())),
        )
    return Catalog(version=raw["version"], intents=intents)
