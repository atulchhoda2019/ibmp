"""Classifier protocol: a real model can be swapped in without touching the engine."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from router.models import Interpretation, RequestContext


@runtime_checkable
class Classifier(Protocol):
    def classify(self, utterance: str, context: RequestContext) -> Interpretation:
        """Return one intent from the closed catalog with calibrated confidence.

        Implementations must not read tenant, participant or auth from the utterance.
        """
