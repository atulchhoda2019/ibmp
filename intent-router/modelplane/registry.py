"""Served-config registry: which model, which adapter, and the eval that earned it a request.

A served config is the whole thing a request runs on — base model, at most one adapter, decoding
parameters and cost class — not just a model name. Loading refuses a config that has not passed a
frozen golden set, so an untested config cannot be served by editing the routing table alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from errors import ConfigError

RINGS = ("canary", "cohort", "prod")
SCOPES = ("domain", "task", "tenant")
#: an eval judged by the family it is judging grades its own homework
FAMILY = {"openai": "openai", "anthropic": "anthropic", "fake": "fake"}


@dataclass(frozen=True)
class EvalRecord:
    eval_id: str
    judge_family: str
    score: float
    threshold: float
    measured_gap: float

    @property
    def passing(self) -> bool:
        return self.score >= self.threshold


@dataclass(frozen=True)
class ServedConfig:
    name: str
    provider: str
    base: str
    adapter: Optional[str]
    scope: str
    domain: str
    task: str
    decoding: Mapping[str, Any]
    cost_per_1k_tokens_usd: float
    ring: str
    eval: EvalRecord

    def cost(self, tokens: int) -> float:
        return round(self.cost_per_1k_tokens_usd * tokens / 1000, 6)


@dataclass(frozen=True)
class Pointer:
    """A promotion is a pointer flip, so a rollback is the same operation in reverse."""

    config: str
    rollback_to: str


@dataclass(frozen=True)
class ModelRegistry:
    version: str
    served: Mapping[str, ServedConfig]
    serving: Mapping[str, Pointer]
    frontier: str

    def get(self, name: str) -> ServedConfig:
        try:
            return self.served[name]
        except KeyError:
            raise ConfigError(f"unknown served config {name!r}") from None

    def frontier_config(self) -> ServedConfig:
        return self.get(self.frontier)


def _eval(name: str, raw: Optional[Mapping[str, Any]]) -> EvalRecord:
    if raw is None:
        raise ConfigError(f"{name}: a served config must carry its eval record")
    missing = {"eval_id", "judge_family", "score", "threshold", "measured_gap"} - set(raw)
    if missing:
        raise ConfigError(f"{name}: eval record missing {sorted(missing)}")
    return EvalRecord(
        eval_id=str(raw["eval_id"]),
        judge_family=str(raw["judge_family"]),
        score=float(raw["score"]),
        threshold=float(raw["threshold"]),
        measured_gap=float(raw["measured_gap"]),
    )


def _validate(config: ServedConfig) -> None:
    if config.ring not in RINGS:
        raise ConfigError(f"{config.name}: unknown ring {config.ring!r}")
    if config.scope not in SCOPES:
        raise ConfigError(f"{config.name}: unknown scope {config.scope!r}")
    if not config.eval.passing:
        raise ConfigError(
            f"{config.name}: eval {config.eval.eval_id} scored {config.eval.score} "
            f"below its threshold {config.eval.threshold}"
        )
    judged_by = FAMILY.get(config.eval.judge_family, config.eval.judge_family)
    if judged_by == FAMILY.get(config.provider, config.provider):
        raise ConfigError(f"{config.name}: judged by its own family {judged_by!r}")
    if config.scope == "tenant" and config.eval.measured_gap <= 0:
        raise ConfigError(f"{config.name}: a tenant adapter needs a measured gap to justify it")


def load_model_registry(path: Path) -> ModelRegistry:
    raw = yaml.safe_load(Path(path).read_text())
    served: Dict[str, ServedConfig] = {}
    for name, entry in raw["served"].items():
        config = ServedConfig(
            name=name,
            provider=str(entry["provider"]),
            base=str(entry["base"]),
            adapter=entry.get("adapter"),
            scope=str(entry.get("scope", "domain")),
            domain=str(entry.get("domain", "any")),
            task=str(entry.get("task", "compose")),
            decoding=dict(entry.get("decoding", {})),
            cost_per_1k_tokens_usd=float(entry.get("cost_per_1k_tokens_usd", 0.0)),
            ring=str(entry.get("ring", "prod")),
            eval=_eval(name, entry.get("eval")),
        )
        _validate(config)
        served[name] = config

    serving = {
        task: Pointer(config=str(entry["config"]), rollback_to=str(entry["rollback_to"]))
        for task, entry in raw.get("serving", {}).items()
    }
    frontier = str(raw["frontier"])
    for task, pointer in serving.items():
        for target in (pointer.config, pointer.rollback_to):
            if target not in served:
                raise ConfigError(f"serving.{task} points at unknown config {target!r}")
    if frontier not in served:
        raise ConfigError(f"frontier points at unknown config {frontier!r}")
    return ModelRegistry(
        version=str(raw["version"]), served=served, serving=serving, frontier=frontier
    )
