"""Deterministic, table-driven classifier stub. No embeddings, no model calls."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

from router.catalog import Catalog
from router.models import OUT_OF_SCOPE, Interpretation, RequestContext

Row = Tuple[str, Mapping[str, Any], float, Tuple[Tuple[str, float], ...]]

#: normalized utterance -> (intent, slots, confidence, top_alternatives)
STUB_UTTERANCES: Dict[str, Row] = {
    "how much of my deductible have i met?": (
        "plan.deductible.status", {}, 0.93, (("plan.deductible.status", 0.93),),
    ),
    "what is the ppo-high deductible?": (
        "plan.deductible.info", {"plan_id": "PPO-High"}, 0.93, (("plan.deductible.info", 0.93),),
    ),
    "increase my 401(k) to 8%": (
        "ret.contribution.change", {"rate_pct": "8"}, 0.95,
        (("ret.contribution.change", 0.95),),
    ),
    "i want to change my contributions": (
        "ret.contribution.change", {}, 0.72,
        (("ret.contribution.change", 0.72), ("plan.elections.view", 0.60)),
    ),
    "i want to change my contributions change the amount": (
        "ret.contribution.change", {}, 0.91, (("ret.contribution.change", 0.91),),
    ),
    "i want to change my contributions i dunno, the money thing": (
        "ret.contribution.change", {}, 0.58,
        (("ret.contribution.change", 0.58), ("plan.elections.view", 0.55)),
    ),
    "i'm going through a divorce, what happens to my coverage?": (
        "life.event.divorce", {}, 0.91, (("life.event.divorce", 0.91),),
    ),
    "my money stuff is wrong": (
        OUT_OF_SCOPE, {}, 0.41,
        ((OUT_OF_SCOPE, 0.41), ("ret.contribution.change", 0.33), ("plan.elections.view", 0.26)),
    ),
}


@dataclass
class StubClassifier:
    """Records calls so tests can assert stage 0 rule matches never reach the classifier."""

    catalog: Catalog
    utterances: Mapping[str, Row] = field(default_factory=lambda: dict(STUB_UTTERANCES))
    calls: List[str] = field(default_factory=list)

    def classify(self, utterance: str, context: RequestContext) -> Interpretation:
        self.calls.append(utterance)
        intent, slots, confidence, alternatives = self.utterances.get(
            _normalize(utterance), (OUT_OF_SCOPE, {}, 0.40, ((OUT_OF_SCOPE, 0.40),))
        )
        self.catalog.get(intent)  # closed catalog: output must be a known id
        return Interpretation(
            intent=intent,
            slots=dict(slots),
            confidence=confidence,
            top_alternatives=tuple(alternatives),
            source="model",
        )


def _normalize(utterance: str) -> str:
    return " ".join(utterance.lower().split())
