"""Model gateway: the only place a model is called, and the only place one can be escalated.

Two invariants live here. The composer *composes*: it may only restate facts the graph already
retrieved, so a number the tools never produced is a validator catch, not an answer. And escalation
is a ladder, not a loop: failed check -> one retry -> frontier -> human.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from errors import ConfigError

from .registry import ModelRegistry, ServedConfig, load_model_registry
from .routing import ModelChoice, ModelRoutingTable, assert_versions, choose, load_model_routing

CONFIG_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_REGISTRY = CONFIG_ROOT / "registry.yaml"
DEFAULT_MODEL_ROUTING = CONFIG_ROOT / "routing.yaml"

NUMBER = re.compile(r"\d+(?:\.\d+)?")
COMPOSER_SYSTEM = (
    "You are a benefits assistant composing the final message. Restate only the facts given to "
    "you. Never introduce a number, date, amount or plan name that is not in the facts. If the "
    "facts do not answer the question, say so."
)


class EscalatedToHuman(Exception):
    """The ladder ran out: nothing a model produced passed the validator."""


@dataclass(frozen=True)
class Attempt:
    config: str
    ok: bool
    reason: Optional[str]
    tokens: int


@dataclass(frozen=True)
class Composition:
    text: str
    config: str
    escalated: bool
    attempts: Sequence[Attempt]
    tokens: int
    cost_usd: float


class ModelBackend(Protocol):
    """Honest interface: a served config in, text plus a token count out."""

    def generate(self, config: ServedConfig, system: str, user: str) -> tuple[str, int]:
        ...


@dataclass
class FakeBackend:
    """Deterministic stand-in so the whole plane runs in CI with no key and no network."""

    #: text to return instead of the composed facts, per config name, for validator tests
    canned: Mapping[str, str] = field(default_factory=dict)

    def generate(self, config: ServedConfig, system: str, user: str) -> tuple[str, int]:
        if config.name in self.canned:
            text = self.canned[config.name]
        else:
            # Restating the facts verbatim is the most conservative composition there is.
            facts = user.split("FACTS:", 1)[-1].strip().splitlines()
            text = " ".join(line.split(": ", 1)[-1].strip() for line in facts if line.strip())
        return text, len(system.split()) + len(user.split())


@dataclass
class OpenAIBackend:
    """Real calls. The adapter is passed through as the model name when one is deployed."""

    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ConfigError("OpenAIBackend needs OPENAI_API_KEY")
        from openai import OpenAI

        self._client = OpenAI(api_key=key)

    def generate(self, config: ServedConfig, system: str, user: str) -> tuple[str, int]:
        response = self._client.responses.create(
            model=config.adapter or config.base,
            instructions=system,
            input=user,
            temperature=float(config.decoding.get("temperature", 0.0)),
            max_output_tokens=int(config.decoding.get("max_output_tokens", 512)),
        )
        usage = response.usage
        tokens = (usage.input_tokens + usage.output_tokens) if usage else 0
        return response.output_text, tokens


def _numbers(text: str) -> List[str]:
    """`$1,500` and `1500` are the same fact written two ways."""
    return NUMBER.findall(text.replace(",", ""))


def validate_composition(text: str, facts: Mapping[str, Any]) -> Optional[str]:
    """Reject any number the tools did not produce: the model composes, it creates no facts."""
    known = {number for value in facts.values() for number in _numbers(str(value))}
    for number in _numbers(text):
        if number not in known:
            return f"unsupported number {number}"
    return None


@dataclass
class ModelGateway:
    registry: ModelRegistry
    table: ModelRoutingTable
    backend: ModelBackend
    frontier_backend: Optional[ModelBackend] = None

    def __post_init__(self) -> None:
        assert_versions(self.registry, self.table)

    def select(self, *, task: str, domain: str, posture: str, tenant: str) -> ModelChoice:
        return choose(
            self.registry, self.table, task=task, domain=domain, posture=posture, tenant=tenant
        )

    def compose(
        self,
        *,
        question: str,
        facts: Mapping[str, Any],
        domain: str,
        posture: str,
        tenant: str,
    ) -> Composition:
        """failed check -> one retry -> frontier -> human, and every step is on the record."""
        choice = self.select(task="compose", domain=domain, posture=posture, tenant=tenant)
        prompt = _prompt(question, facts)
        attempts: List[Attempt] = []
        total_tokens = 0
        cost = 0.0

        ladder = [
            (choice.config, self.backend),  # first call
            (choice.config, self.backend),  # exactly one retry
            (self.registry.frontier_config(), self.frontier_backend or self.backend),
        ]
        for config, backend in ladder:
            text, tokens = backend.generate(config, COMPOSER_SYSTEM, prompt)
            total_tokens += tokens
            cost += config.cost(tokens)
            reason = validate_composition(text, facts)
            attempts.append(
                Attempt(config=config.name, ok=reason is None, reason=reason, tokens=tokens)
            )
            if reason is None:
                return Composition(
                    text=text,
                    config=config.name,
                    escalated=config.name != choice.config.name,
                    attempts=tuple(attempts),
                    tokens=total_tokens,
                    cost_usd=round(cost, 6),
                )
        raise EscalatedToHuman(f"composition failed on {len(attempts)} attempts")


def _prompt(question: str, facts: Mapping[str, Any]) -> str:
    lines = "\n".join(f"- {key}: {value}" for key, value in facts.items())
    return f"QUESTION: {question}\nFACTS:\n{lines}"


def build_gateway(
    *,
    backend: Optional[ModelBackend] = None,
    registry_path: Path = DEFAULT_MODEL_REGISTRY,
    routing_path: Path = DEFAULT_MODEL_ROUTING,
) -> ModelGateway:
    return ModelGateway(
        registry=load_model_registry(registry_path),
        table=load_model_routing(routing_path),
        backend=backend or FakeBackend(),
    )


def backend_from_env() -> ModelBackend:
    """`MODEL_BACKEND=openai` opts into real calls; everything else stays deterministic."""
    if os.environ.get("MODEL_BACKEND", "fake").lower() == "openai":
        return OpenAIBackend()
    return FakeBackend()


@dataclass
class Scorecard:
    """What a served config is judged on once it is live, counted per served turn."""

    answers: int = 0
    validator_catch: int = 0
    abstention: int = 0
    wrong_graph: int = 0
    escalation: int = 0
    handoff: int = 0
    cost_usd: float = 0.0
    confidence_sum: float = 0.0
    correct: int = 0
    cache_hit: int = 0
    cache_regret: int = 0

    def record(self, composition: Composition) -> None:
        self.answers += 1
        self.validator_catch += sum(1 for attempt in composition.attempts if not attempt.ok)
        self.escalation += int(composition.escalated)
        self.cost_usd = round(self.cost_usd + composition.cost_usd, 6)

    def snapshot(self) -> Dict[str, float]:
        served = max(self.answers, 1)
        return {
            "validator_catch": self.validator_catch,
            "abstention": self.abstention,
            "wrong_graph": self.wrong_graph,
            "escalation": self.escalation,
            "handoff": self.handoff,
            "containment": round(1 - self.handoff / served, 4),
            "calibration_error": round(
                abs(self.confidence_sum / served - self.correct / served), 4
            ),
            "cost_per_successful_answer_usd": round(self.cost_usd / served, 6),
            "cache_regret": self.cache_regret,
        }
