"""TurnState: the single typed state every node reads and writes.

Parallel reads write to three separate keys (evidence_items / fact_items / calc_items)
that `assemble` joins into `envelope`: a pydantic state object cannot carry an append
reducer, so concurrent writes to one list would be rejected by LangGraph.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Band = Literal["HIGH", "MEDIUM", "LOW"]
Posture = Literal["READ", "WRITE", "DRAFT"]


class Intent(BaseModel):
    name: str
    confidence: float
    band: Band
    slots: dict[str, str] = {}
    alternates: list[str] = []
    source: Literal["rule", "classifier"] = "classifier"


class PlanDecision(BaseModel):
    graph_id: str
    table_row_id: str
    rung: int
    budgets: dict[str, int] = {}
    required_evidence: list[str] = []
    posture: Posture
    requires_capability: Optional[str] = None
    clarify: bool = False


class EnvelopeItem(BaseModel):
    item_id: str
    kind: Literal["policy", "fact", "calc"]
    source: str
    effective_from: str
    effective_to: Optional[str] = None
    observed_at: str
    freshness_policy: str = "fp_live"
    evidence_type: str = ""
    payload: dict = {}


class Proposal(BaseModel):
    proposal_id: str
    action: Literal["ContributionChange"]
    participant_ref: str
    current: dict
    requested: dict
    validations: list[str] = []
    nonce: str
    expires_at: str


class TurnState(BaseModel):
    conversation_id: str
    turn_id: str
    tenant_id: str
    participant_ref: str
    utterance: str
    ui_context: dict = {}
    service_date: str = ""
    turn_started_at: str = ""
    clarify_rounds: int = 0

    intent: Optional[Intent] = None
    plan: Optional[PlanDecision] = None

    evidence_items: list[EnvelopeItem] = []
    fact_items: list[EnvelopeItem] = []
    calc_items: list[EnvelopeItem] = []
    envelope: list[EnvelopeItem] = []
    abstain_reason: Optional[str] = None

    draft: Optional[str] = None
    validation: Optional[dict] = None
    proposal: Optional[Proposal] = None
    confirmed: bool = False
    execution: Optional[dict] = None
    receipt: Optional[dict] = None
    response: Optional[dict] = None
    retries: dict[str, int] = {}
    audit: list[dict] = Field(default_factory=list)
