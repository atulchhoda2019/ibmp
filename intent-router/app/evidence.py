"""Evidence envelopes and the validator that stands between a graph and the client voice.

The row says how stale evidence may be; this module decides whether an answer is allowed to
be spoken at all. Stale or uncited evidence abstains — the assistant says it cannot answer
rather than answering from memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Mapping, Optional, Sequence

ABSTENTION = (
    "I can see the record, but the numbers I have are older than I'm allowed to quote for this, "
    "so I'd rather not give you a figure that might be wrong. A specialist can read you the live value."
)
UNSUPPORTED = (
    "I don't have a citable source for that yet, so I won't guess. Let me get you to someone who can check."
)


@dataclass(frozen=True)
class Citation:
    source: str
    effective_date: date


@dataclass(frozen=True)
class Envelope:
    """Facts plus the citations that support them, each with its own effective date."""

    facts: Mapping[str, Any] = field(default_factory=dict)
    citations: Sequence[Citation] = ()

    @property
    def oldest_effective_date(self) -> Optional[date]:
        return min((c.effective_date for c in self.citations), default=None)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: Optional[str] = None
    message: Optional[str] = None


def validate(envelope: Envelope, policy: Optional[Mapping[str, Any]], *, today: Optional[date] = None) -> Verdict:
    """Schema, citations and staleness, in that order. No policy means no evidence claim."""
    if policy is None:
        return Verdict(ok=True)
    if not envelope.citations:
        return Verdict(ok=False, reason="uncited", message=UNSUPPORTED)
    max_age_days = policy.get("max_age_days")
    if max_age_days is None:
        return Verdict(ok=True)
    oldest = envelope.oldest_effective_date
    assert oldest is not None
    age = (today or date.today()) - oldest
    if age > timedelta(days=int(max_age_days)) and policy.get("abstain_if_stale", True):
        return Verdict(ok=False, reason="stale", message=ABSTENTION)
    return Verdict(ok=True)


def envelope_from_tool(payload: Mapping[str, Any]) -> Envelope:
    """Adapt the stub RetrieveEvidence result into a typed envelope."""
    citations = tuple(
        Citation(source=item["source"], effective_date=date.fromisoformat(item["effective_date"]))
        for item in payload.get("citations", ())
    )
    return Envelope(facts=dict(payload.get("facts", {})), citations=citations)
