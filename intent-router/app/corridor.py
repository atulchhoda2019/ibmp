"""The transaction corridor: proposal -> nonce-bound confirmation -> idempotent command.

Rules the corridor enforces, independent of whatever the model said:

* a confirmation only fires for a proposal this participant actually holds;
* the nonce must be the one issued with that exact proposal;
* a proposal expires, and a superseded proposal is dead the moment a newer one is issued;
* the command key is the proposal id, so a retried or double-clicked confirmation collapses
  onto the first receipt instead of moving money twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional

from graphs.registry import Proposal, Receipt

DEFAULT_TTL = timedelta(minutes=15)


class ConfirmationError(Exception):
    """A confirmation that must not fire. The message is safe to show a participant."""


@dataclass(frozen=True)
class PendingProposal:
    proposal_id: str
    confirmation_nonce: str
    action: str
    params: Mapping[str, Any]
    effective_date: str
    tenant_id: str
    participant_ref: str
    expires_at: datetime

    def summary(self) -> Dict[str, Any]:
        """The slice the client is allowed to hold in conversation state."""
        return {
            "proposal_id": self.proposal_id,
            "action": self.action,
            "params": dict(self.params),
            "effective_date": self.effective_date,
            "expires_at": self.expires_at.isoformat(timespec="seconds"),
        }


@dataclass
class ConfirmationStore:
    ttl: timedelta = DEFAULT_TTL
    pending: Dict[str, PendingProposal] = field(default_factory=dict)
    #: latest live proposal per participant; issuing a new one supersedes the old
    latest: Dict[str, str] = field(default_factory=dict)
    receipts: Dict[str, Receipt] = field(default_factory=dict)

    def register(
        self, proposal: Proposal, *, tenant_id: str, participant_ref: str, now: Optional[datetime] = None
    ) -> PendingProposal:
        moment = now or datetime.now(timezone.utc)
        pending = PendingProposal(
            proposal_id=proposal.proposal_id,
            confirmation_nonce=proposal.confirmation_nonce,
            action=proposal.action,
            params=dict(proposal.params),
            effective_date=proposal.effective_date,
            tenant_id=tenant_id,
            participant_ref=participant_ref,
            expires_at=moment + self.ttl,
        )
        owner = self._owner(tenant_id, participant_ref)
        superseded = self.latest.get(owner)
        if superseded is not None:
            self.pending.pop(superseded, None)
        self.pending[proposal.proposal_id] = pending
        self.latest[owner] = proposal.proposal_id
        return pending

    def get(self, proposal_id: str) -> Optional[PendingProposal]:
        return self.pending.get(proposal_id)

    def claim(
        self,
        proposal_id: str,
        confirmation_nonce: str,
        *,
        tenant_id: str,
        participant_ref: str,
        now: Optional[datetime] = None,
    ) -> PendingProposal:
        """Validate a confirmation and hand back the proposal it is bound to."""
        moment = now or datetime.now(timezone.utc)
        pending = self.pending.get(proposal_id)
        if pending is None:
            raise ConfirmationError(
                "That confirmation is no longer valid — the change it referred to has expired or "
                "was replaced. Ask me to set it up again."
            )
        if pending.tenant_id != tenant_id or pending.participant_ref != participant_ref:
            raise ConfirmationError("That confirmation does not belong to this session.")
        if pending.confirmation_nonce != confirmation_nonce:
            raise ConfirmationError("That confirmation code doesn't match the change I proposed.")
        if moment >= pending.expires_at:
            self._forget(pending)
            raise ConfirmationError(
                "That proposal expired before it was confirmed, so nothing happened. "
                "Ask me to set it up again and I'll build a fresh one."
            )
        return pending

    def record(self, pending: PendingProposal, receipt: Receipt) -> Receipt:
        """Commit the receipt under the command key and retire the proposal."""
        self.receipts[receipt.command_key] = receipt
        self._forget(pending)
        return receipt

    def replay(self, proposal_id: str) -> Optional[Receipt]:
        """A duplicate command collapses onto the receipt the first one produced."""
        receipt = self.receipts.get(proposal_id)
        if receipt is None:
            return None
        return Receipt(
            command_key=receipt.command_key,
            action=receipt.action,
            params=dict(receipt.params),
            effective_date=receipt.effective_date,
            executed_at=receipt.executed_at,
            reversal=receipt.reversal,
            duplicate=True,
        )

    def _forget(self, pending: PendingProposal) -> None:
        self.pending.pop(pending.proposal_id, None)
        owner = self._owner(pending.tenant_id, pending.participant_ref)
        if self.latest.get(owner) == pending.proposal_id:
            self.latest.pop(owner, None)

    @staticmethod
    def _owner(tenant_id: str, participant_ref: str) -> str:
        return f"{tenant_id}:{participant_ref}"
