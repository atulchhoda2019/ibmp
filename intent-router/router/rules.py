"""Stage 0: ordered phrase/regex rules from the catalog. A match bypasses the classifier."""
from __future__ import annotations

from typing import Optional

from .catalog import Catalog
from .models import Interpretation


def match_rules(utterance: str, catalog: Catalog) -> Optional[Interpretation]:
    for intent in catalog.intents.values():
        for pattern in intent.rule_patterns:
            found = pattern.search(utterance)
            if found is None:
                continue
            return Interpretation(
                intent=intent.id,
                slots={k: v for k, v in found.groupdict().items() if v is not None},
                confidence=1.0,
                top_alternatives=(),
                source="rule",
            )
    return None
