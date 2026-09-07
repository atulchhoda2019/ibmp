"""Model-backed classifier: the interpretation step, routed through the model plane.

It proposes `(intent, slots, confidence)` and nothing else. An intent outside the closed catalog
is not a route, it is out of scope, and a malformed answer degrades to out-of-scope at zero
confidence so the decision table still fires the fallback row rather than the engine crashing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

from modelplane.gateway import ModelGateway
from router.catalog import Catalog
from router.models import OUT_OF_SCOPE, Interpretation, RequestContext

SYSTEM = (
    "Classify the user's benefits request into exactly one intent id from the catalog below. "
    "Return JSON only: {\"intent\": str, \"slots\": object, \"confidence\": number, "
    "\"alternatives\": [[intent, confidence], ...]}. Use the exact intent ids given. If nothing "
    f"fits, use {OUT_OF_SCOPE!r}. Never infer tenant, participant or authentication from the text."
)


@dataclass
class ModelClassifier:
    """Wraps a gateway so the classifier honours the same served-config governance as compose."""

    gateway: ModelGateway
    catalog: Catalog
    domain: str = "benefits"

    def classify(self, utterance: str, context: RequestContext) -> Interpretation:
        choice = self.gateway.select(
            task="classify", domain=self.domain, posture="read", tenant=context.tenant_id
        )
        text, _ = self.gateway.backend.generate(choice.config, SYSTEM, self._prompt(utterance))
        return self._parse(text)

    def _prompt(self, utterance: str) -> str:
        catalog = "\n".join(f"- {intent}" for intent in sorted(self.catalog.intents))
        return f"CATALOG:\n{catalog}\n\nUTTERANCE: {utterance}"

    def _parse(self, text: str) -> Interpretation:
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return Interpretation(intent=OUT_OF_SCOPE, confidence=0.0, source="model")
        if not isinstance(payload, dict):
            return Interpretation(intent=OUT_OF_SCOPE, confidence=0.0, source="model")

        intent = str(payload.get("intent", OUT_OF_SCOPE))
        if intent != OUT_OF_SCOPE and intent not in self.catalog:
            # A model naming an intent nobody approved is a miss, not a new capability.
            return Interpretation(intent=OUT_OF_SCOPE, confidence=0.0, source="model")

        raw_slots = payload.get("slots") or {}
        slots: Dict[str, Any] = dict(raw_slots) if isinstance(raw_slots, Mapping) else {}
        alternatives: List[Tuple[str, float]] = []
        for entry in payload.get("alternatives") or ():
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                candidate, score = entry
                if candidate == OUT_OF_SCOPE or candidate in self.catalog:
                    alternatives.append((str(candidate), float(score)))
        return Interpretation(
            intent=intent,
            slots=slots,
            confidence=float(payload.get("confidence", 0.0)),
            source="model",
            top_alternatives=tuple(alternatives),
        )
