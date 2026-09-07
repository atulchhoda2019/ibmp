# design.md · Benefits Conversation Runtime, Agent-Call Flow with Mocked Data Plane

Reference implementation spec for Devin. Build the flow INSIDE the agent calls: the
LangGraph state machine for one conversational turn, with every fact source mocked
(Operational Data Layer, policy evidence, calculators, model gateway, system of record),
LangGraph checkpointing for resume, a human-in-the-loop transaction corridor, and
LangSmith traces on every node and tool call.

This file is self-contained. The companion file `design_scenarios_v1.md` holds the 12
acceptance scenarios (S1..S12); the tests in section 12 map to them.

---

## 0. Non-negotiable invariants

These are enforced by structure, and the test suite proves each one.

- I1 · The model never supplies an intent, a decision, or a fact. The classifier selects
  an intent from a closed catalog; a versioned decision table selects the graph and
  budgets; facts come only from typed mock tools.
- I2 · No `Execute*` tool exists in any *advisory* graph; an advisory row carries
  `rung_max: 0` and its only write-shaped tool is `propose_contribution_change`, which
  returns a typed proposal object. Execution lives solely in the transaction corridor.
- I3 · A write reaches the (mock) system of record only through the corridor: proposal,
  deterministic validation, preview, nonce-bound confirmation, idempotent command,
  read-after-write verification, receipt.
- I4 · Every sentence in a composed answer must cite an evidence envelope item id. The
  validator rejects uncited claims; one bounded retry, then scripted fallback.
- I5 · The graph is resumable: kill the process between any two nodes, restart, and the
  turn completes without re-running side effects (checkpointer + idempotency keys).
- I6 · `userEmail` style identifiers are used to identify the user only and are never
  forwarded to unrelated services or into model prompts.

---

## 1. Stack and repo layout

Python 3.11. Dependencies: `langgraph`, `langchain-core`, `langsmith`, `pydantic>=2`,
`fastapi`, `uvicorn`, `pyyaml`, `pytest`. No real LLM call anywhere; the model gateway is
a deterministic mock (section 8) so the whole system runs offline and tests are stable.

```
benefits-agent-flow/
  app/
    main.py                 # FastAPI: POST /turn, POST /confirm, GET /trace/{conversation_id}
    state.py                # TurnState schema (single source of truth)
    graph_build.py          # LangGraph StateGraph wiring + checkpointer + interrupts
    nodes/
      ingress.py            # canonicalize request, attach trusted context
      planner.py            # stage 0 rules, stage 1 classifier, stage 2 table, stage 3 ladder
      gate_plan.py          # governance gate 1: policy plane, capability bundle check
      read_evidence.py      # mock policy evidence, filter first then rerank
      read_facts.py         # mock ODL typed reads
      read_calc.py          # mock deterministic calculators
      assemble.py           # evidence envelope join + freshness enforcement
      reason.py             # mock model gateway composes draft from envelope only
      gate_output.py        # governance gate 2: validator (schema, citations, PII)
      respond.py            # final answer or scripted fallback or handoff
      corridor.py           # propose, preview, confirm, execute, verify, receipt
    registry/
      intent_catalog.yaml   # closed catalog (12 intents for the mock)
      decision_table.yaml   # intent x band x capability x risk -> graph + budgets + rung
      bundles.yaml          # per-tenant capability bundles (rungs per action)
      freshness.yaml        # per-evidence-type freshness policies
    mocks/
      odl.py                # participants, elections, balances, accumulators
      evidence.py           # versioned policy passages with effective dates
      calculator.py         # limit room, match projection, PT cost estimate
      model_gateway.py      # deterministic composer + fault injection
      sor.py                # mock system of record: execute-once, receipts, fault injection
    audit.py                # JSONL audit events, one line per node transition
    tracing.py              # LangSmith setup helpers
  fixtures/
    participants.json
    plans.json
    passages.json
    tenants.json
  tests/
    test_planner.py  test_ladder.py  test_envelope.py  test_corridor.py
    test_resume.py   test_scenarios.py   test_invariants.py
  .env.example              # LANGSMITH keys, fault-injection flags
```

---

## 2. TurnState · the single typed state

Everything the graph knows lives here. LangGraph checkpoints this object at every node
boundary. Pydantic model serialized to dict for the graph.

```python
# app/state.py
from typing import Literal, Optional
from pydantic import BaseModel, Field

class Intent(BaseModel):
    name: str                       # from the closed catalog only
    confidence: float               # calibrated 0..1
    band: Literal["HIGH", "MEDIUM", "LOW"]
    slots: dict[str, str] = {}      # canonicalized: "six percent" -> "0.06"
    alternates: list[str] = []      # top competing labels, for MEDIUM options

class PlanDecision(BaseModel):
    graph_id: str                   # e.g. "g_contribution_change_v3"
    table_row_id: str               # audit pointer into decision_table.yaml
    rung: int                       # 1 draft, 2 propose+confirm, 3 execute+notify, 4 auto
    budgets: dict[str, int]         # {"max_steps": 8, "max_tools": 6, "max_tokens": 1200, "deadline_ms": 6000}
    required_evidence: list[str]    # evidence types + freshness policy ids
    posture: Literal["READ", "WRITE"]

class EnvelopeItem(BaseModel):
    item_id: str                    # cite handle, e.g. "ev-7"
    kind: Literal["policy", "fact", "calc"]
    source: str                     # tool + version
    effective_from: str
    effective_to: Optional[str]
    observed_at: str
    payload: dict

class Proposal(BaseModel):
    proposal_id: str
    action: Literal["ContributionChange"]
    participant_ref: str            # opaque ref, never an email
    current: dict                   # {"rate": "0.06"}
    requested: dict                 # {"rate": "0.08", "effective_date": "2026-10-01"}
    validations: list[str]          # deterministic checks that passed
    nonce: str
    expires_at: str

class TurnState(BaseModel):
    conversation_id: str
    turn_id: str
    tenant_id: str
    participant_ref: str
    utterance: str
    ui_context: dict = {}
    intent: Optional[Intent] = None
    plan: Optional[PlanDecision] = None
    envelope: list[EnvelopeItem] = []
    draft: Optional[str] = None
    validation: Optional[dict] = None
    proposal: Optional[Proposal] = None
    receipt: Optional[dict] = None
    response: Optional[dict] = None     # {"kind": "answer|clarify|handoff|preview|receipt", ...}
    retries: dict[str, int] = {}
    audit: list[dict] = Field(default_factory=list)
```

Rule: nodes only read and write TurnState. No globals, no hidden caches. That is what
makes checkpoint resume trivially correct.

---

## 3. The LangGraph flow

One turn is one graph invocation. `thread_id = conversation_id`, so the checkpointer
carries multi-turn context and pending transactions across invocations.

```
ingress
  -> planner                      (stage 0 rules, stage 1 classifier, stage 2 table, stage 3 ladder)
  -> gate_plan                    (governance gate 1: bundle allows graph? consent? rung?)
  -> [fan-out]  read_evidence  |  read_facts  |  read_calc      (parallel, READ ONLY)
  -> assemble                     (join + freshness; missing required -> retry once or abstain)
  -> reason                       (mock model composes draft, cites item_ids)
  -> gate_output                  (validator; fail -> one retry of reason, then scripted)
  -> branch:
       posture READ   -> respond (END)
       posture WRITE  -> corridor_propose -> corridor_preview -> [INTERRUPT]
                         corridor_execute -> corridor_verify -> respond (END)
```

The interrupt before `corridor_execute` is the confirmation gate. LangGraph pauses the
graph, the checkpointer persists the pending state, and the graph resumes only when the
client posts the nonce back. Confirmation is structural, not a prompt convention.

```python
# app/graph_build.py
import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from app.state import TurnState
from app.nodes import (ingress, planner, gate_plan, read_evidence, read_facts,
                       read_calc, assemble, reason, gate_output, respond, corridor)

def route_after_planner(state: TurnState) -> str:
    band = state.intent.band
    if band == "LOW":
        return "respond"                      # scripted or warm handoff, never a guess
    return "gate_plan"

def route_after_gate_output(state: TurnState) -> str:
    if not state.validation["passed"]:
        if state.retries.get("reason", 0) < 1:
            return "reason"                   # one bounded retry
        return "respond"                      # scripted fallback
    if state.plan.posture == "WRITE":
        return "corridor_propose"
    return "respond"

def build_graph(checkpoint_path: str = "checkpoints.sqlite"):
    g = StateGraph(TurnState)
    g.add_node("ingress", ingress.run)
    g.add_node("planner", planner.run)
    g.add_node("gate_plan", gate_plan.run)
    g.add_node("read_evidence", read_evidence.run)
    g.add_node("read_facts", read_facts.run)
    g.add_node("read_calc", read_calc.run)
    g.add_node("assemble", assemble.run)
    g.add_node("reason", reason.run)
    g.add_node("gate_output", gate_output.run)
    g.add_node("corridor_propose", corridor.propose)
    g.add_node("corridor_preview", corridor.preview)
    g.add_node("corridor_execute", corridor.execute)
    g.add_node("corridor_verify", corridor.verify)
    g.add_node("respond", respond.run)

    g.add_edge(START, "ingress")
    g.add_edge("ingress", "planner")
    g.add_conditional_edges("planner", route_after_planner,
                            {"gate_plan": "gate_plan", "respond": "respond"})
    # fan out to the three parallel reads, fan in at assemble
    g.add_edge("gate_plan", "read_evidence")
    g.add_edge("gate_plan", "read_facts")
    g.add_edge("gate_plan", "read_calc")
    g.add_edge("read_evidence", "assemble")
    g.add_edge("read_facts", "assemble")
    g.add_edge("read_calc", "assemble")
    g.add_edge("assemble", "reason")
    g.add_edge("reason", "gate_output")
    g.add_conditional_edges("gate_output", route_after_gate_output,
                            {"reason": "reason",
                             "corridor_propose": "corridor_propose",
                             "respond": "respond"})
    g.add_edge("corridor_propose", "corridor_preview")
    g.add_edge("corridor_preview", "corridor_execute")   # interrupted, see compile()
    g.add_edge("corridor_execute", "corridor_verify")
    g.add_edge("corridor_verify", "respond")
    g.add_edge("respond", END)

    saver = SqliteSaver.from_conn_string(checkpoint_path)
    return g.compile(checkpointer=saver,
                     interrupt_before=["corridor_execute"])   # the confirmation gate
```

Parallel-read note: with plain edges as above LangGraph runs the three read nodes in the
same superstep and merges their state writes; each read node must only append to its own
key (`envelope` uses an append reducer, declare with `Annotated[list, operator.add]` in
the graph schema if you use TypedDict instead of pydantic). If merge conflicts appear,
switch `envelope` to three separate keys and join them in `assemble`.

---

## 4. API surface (FastAPI)

```
POST /turn
  body: {conversationId, tenantId, participantRef, utterance, uiContext}
  -> runs the graph with config {"configurable": {"thread_id": conversationId}}
  -> 200 {kind: "answer", text, citations[]}                (READ path)
  -> 200 {kind: "clarify", question, options[]}             (MEDIUM ladder)
  -> 200 {kind: "handoff", summary, top_intents[]}          (LOW ladder)
  -> 200 {kind: "preview", proposal, nonce, expires_at}     (WRITE path, interrupted)

POST /confirm
  body: {conversationId, proposalId, nonce}
  -> validates nonce + expiry against checkpointed pending proposal
  -> resumes the interrupted graph:  graph.invoke(None, config)  after injecting
     {"confirmed": true} via graph.update_state(config, {...})
  -> 200 {kind: "receipt", receipt}
  -> 409 if nonce mismatch, expired, or revalidation now fails (limits changed)

GET /trace/{conversationId}
  -> returns the JSONL audit events for the thread (and the LangSmith run URL if enabled)
```

Wrong or replayed nonce NEVER executes: `/confirm` checks nonce equality, expiry, and
re-runs deterministic validation before resuming. Expired proposals require a fresh turn.

---

## 5. Planner internals (nodes/planner.py)

Four stages, all deterministic except the mock classifier's lookup.

Stage 0, rules: exact-match utterance patterns jump straight to a catalog intent with
confidence 1.0 (examples: "change my contribution to N percent", "what is my balance").
A rule hit never touches the classifier.

Stage 1, classifier mock: keyword-scored lookup over `intent_catalog.yaml`. Returns
top intent, calibrated confidence, alternates, and typed slots. The canonicalizer maps
"six percent", "6%", "0.06" to slot `rate = "0.06"` so cache keys and proposals are stable.

Stage 2, decision table: load `decision_table.yaml`, match on
(intent, band, tenant capability, risk tier). Exactly one row wins; the row id goes into
the audit record. The row supplies graph_id, posture, rung, budgets, required evidence.

```yaml
# registry/decision_table.yaml (excerpt)
- row: r-017
  intent: contribution_change
  band: HIGH
  requires_capability: retirement_write
  risk: transactional
  graph: g_contribution_change_v3
  posture: WRITE
  rung_max: 2                 # propose + confirm is the ceiling for this action
  budgets: {max_steps: 10, max_tools: 6, max_tokens: 1500, deadline_ms: 8000}
  evidence: [plan_rules:fp_current, elections:fp_live, limit_room:fp_session]
- row: r-021
  intent: contribution_change
  band: MEDIUM
  graph: g_retirement_readonly_v2   # table-chosen read-only variant
  posture: READ
  budgets: {max_steps: 6, max_tools: 4, max_tokens: 900, deadline_ms: 6000}
```

Stage 3, ambiguity ladder: HIGH runs the selected graph. MEDIUM either runs the
read-only variant the table names or emits ONE clarifying question with named options
(from `intent.alternates`); the answer re-enters the planner once, and a second MEDIUM
falls to LOW. LOW returns a scripted answer or a handoff payload carrying the transcript,
slots, and top guesses. LOW events append to `fixtures/fallback_queue.jsonl` (the weekly
catalog review feed).

---

## 6. Mock data plane

All mocks are deterministic, versioned, effective-dated, and fault-injectable through
env flags. Fixtures ship in `fixtures/`.

Participants (`mocks/odl.py`, reading `participants.json`):

| ref     | persona                  | elections            | facts for scenarios                          |
|---------|--------------------------|----------------------|----------------------------------------------|
| P-1001  | mid-career, clean path   | 401k rate 0.06       | limit room comfortably positive              |
| P-1002  | high earner, near limit  | 401k rate 0.10       | requested raise would exceed annual limit    |
| P-1003  | new hire, waiting period | not yet eligible     | eligibility_date in the future               |
| P-1004  | PT cost asker            | medical PPO plan     | deductible met 40 percent, accumulator fresh |

ODL tool contract (every read returns provenance):

```python
def get_elections(participant_ref: str) -> dict:
    # {"payload": {...}, "source": "odl.elections.v5", "observed_at": iso8601,
    #  "effective_from": ..., "effective_to": None}
```

Policy evidence (`mocks/evidence.py`): passages in `passages.json`, each with tenant,
plan, population, effective dates, version, and a citation handle. Retrieval FILTERS
FIRST on tenant + plan + service date with hard predicates, then applies a trivial
keyword rerank inside the authorized set. A cross-tenant passage in the fixtures exists
specifically so `test_invariants.py` can prove it is unreachable.

Calculators (`mocks/calculator.py`): pure functions with receipts.
`limit_room(participant)`, `match_projection(rate)`, `pt_cost_estimate(plan, accumulator)`.
The model mock never does arithmetic; it can only quote calc payloads.

Freshness (`registry/freshness.yaml`): per evidence type, max age and behavior on
violation. `fp_session` means must be fetched this turn; `fp_live` allows 5 minutes;
`fp_current` means the version effective on the service date. `assemble` enforces these:
stale optional evidence is dropped and disclosed, stale required evidence triggers one
refetch then abstention.

System of record (`mocks/sor.py`): `submit(command)` is execute-once keyed by
`idempotency_key` (equal to proposal_id). Repeat submits return the original receipt.
Fault flags: `SOR_TIMEOUT_ONCE=1` makes the first submit raise after committing, which
forces the reconcile-by-key path in `corridor_verify`.

Model gateway (`mocks/model_gateway.py`): composes the draft from the envelope with
templates per graph, appending `[ev-N]` citations to every sentence. Fault flags:
`MODEL_UNCITED_CLAIM=1` injects one fabricated sentence (validator must catch it),
`MODEL_TIMEOUT_ONCE=1` forces the retry path.

---

## 7. The corridor (nodes/corridor.py)

`propose`: builds the typed `Proposal` from canonical slots + live facts, runs
deterministic validations (eligibility, limit room, plan rules), generates
`nonce = secrets.token_urlsafe(16)` and `expires_at = now + 10 minutes`. On validation
failure the turn returns an answer explaining the exact failing check, with citations.

`preview`: writes the response `{kind: "preview", ...}` including current value,
requested value, effective date, and the undo window. The graph then hits the
`interrupt_before=["corridor_execute"]` breakpoint and pauses.

`execute` (runs only after `/confirm` resumes the graph): submits the idempotent command
to the mock SoR. Never retries blindly; on unknown outcome it proceeds to verify.

`verify`: reads back the election from ODL (the mock SoR updates the ODL fixture) and
compares to the requested change; queries the SoR by idempotency key if the submit
outcome was unknown. Only then writes `receipt` into state.

`respond`: for the WRITE path, reports completion ONLY from the verified receipt.

Rung enforcement: `gate_plan` compares the table row's rung ceiling with the tenant
bundle (`bundles.yaml`). A tenant at rung 1 for `contribution_change` gets a drafted
change form instead of the corridor; rung 2 is this full flow. Rungs 3 and 4 run the same
corridor without the human pause (rung 3 also notifies) and keep every other gate:
authorization, revalidation, idempotency, read-after-write verification and the receipt.

---

## 8. LangSmith tracing (app/tracing.py)

Every node, tool call, and the classifier are visible as a LangSmith run tree, one root
run per turn, so a reviewer can answer "why did it say that" from the trace alone.

```python
# .env.example
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...        # leave empty to run fully offline
LANGSMITH_PROJECT=benefits-agent-flow
```

LangGraph auto-traces the graph when the env vars are set. Add explicit spans on the
mock tools and stamp versions as metadata:

```python
# app/tracing.py
from langsmith import traceable

def tool_span(name: str, version: str):
    def deco(fn):
        return traceable(name=name, run_type="tool",
                         metadata={"tool_version": version})(fn)
    return deco

# mocks/odl.py
@tool_span("odl.get_elections", "v5")
def get_elections(participant_ref: str) -> dict: ...
```

Per-turn root metadata, set when invoking the graph:

```python
config = {
    "configurable": {"thread_id": conversation_id},
    "run_name": f"turn:{intent_guess or 'unknown'}",
    "tags": [tenant_id, "mocked-data-plane"],
    "metadata": {
        "catalog_version": "intents-v7",
        "table_version": "table-v12",
        "bundle_version": bundle["version"],
        "graph_id": None,   # planner overwrites via trace metadata event
    },
}
result = graph.invoke(initial_state, config)
```

Redaction rule: participant payloads enter traces as refs and field names only, never
raw values; enforce with a `process_inputs` hook on `traceable` that masks
`payload` keys. `test_invariants.py` asserts no fixture SSN-like or email-like string
appears in the emitted trace/audit output.

The local `audit.py` JSONL log is the offline mirror of the trace: one event per node
with state diff summary, row ids, versions, and timings. `GET /trace/{id}` serves it, and
includes the LangSmith run URL when tracing is on.

---

## 9. Worked example the demo must reproduce

Turn 1, advisory: P-1001 asks "what happens if I raise my contribution from 6 to 8
percent". Rules miss, classifier returns `retirement_projection` HIGH. Table row r-009
selects `g_retirement_readonly_v2`, posture READ. Parallel reads: plan rules passage,
live elections, `match_projection(0.08)` and `limit_room`. Envelope has 4 items. Draft
cites all of them. Validator passes. Response kind `answer`.

Turn 2, action: "ok, change it to 8 percent". Stage 0 rule fires,
`contribution_change` at confidence 1.0, slots `{rate: "0.08"}`. Row r-017, posture
WRITE, rung 2 confirmed against the T-ACME bundle. Reads and envelope again (fresh, per
`fp_session`). Proposal built and validated, preview returned, graph interrupted.
`/confirm` with the right nonce resumes: execute once, verify reads back 0.08, receipt
returned. LangSmith shows one run tree per turn; turn 2 shows the interrupt and resume.

Negative twins: P-1002 same turn 2 fails validation with limit-room math in the
explanation; P-1003 fails eligibility with the waiting-period passage cited; T-ZEN
tenant (rung 1) gets a drafted form, never a preview.

---

## 10. Failure and resume drills (must be demonstrable)

- `MODEL_UNCITED_CLAIM=1`: validator rejects, one retry without the flag consumed,
  clean answer on retry. Trace shows both attempts.
- `MODEL_TIMEOUT_ONCE=1`: reason node retries once within budget, then scripted fallback.
- `EVIDENCE_EMPTY=1` for a required type: assemble refetches once, then abstains with a
  safe limitation message. No generation happens.
- `SOR_TIMEOUT_ONCE=1`: execute raises after commit; verify reconciles by idempotency
  key and still produces exactly one receipt. Submitting `/confirm` twice also produces
  exactly one receipt.
- Kill test: run turn 2 to the interrupt, kill the process, restart, `/confirm`. The
  SqliteSaver checkpoint restores the pending proposal and the turn completes.

---

## 11. Milestones for Devin

- M1: repo skeleton, TurnState, fixtures, audit log, graph compiles with no-op nodes.
- M2: planner complete (rules, classifier mock, table, ladder) with `test_planner.py`
  and `test_ladder.py` green.
- M3: READ path end to end (worked example turn 1) with envelope, freshness, validator.
- M4: corridor with interrupt + `/confirm` resume (worked example turn 2).
- M5: failure drills and kill test green (`test_resume.py`, `test_corridor.py`).
- M6: full `test_scenarios.py` mapped to S1..S12 from `design_scenarios_v1.md`, plus
  `test_invariants.py` (no Execute tool importable from any node module, cross-tenant
  passage unreachable, uncited claim blocked, no raw PII in traces).

Definition of done: `pytest -q` green offline; with LANGSMITH_API_KEY set, one browsable
run tree per turn showing planner stages, three parallel reads, both gates, and the
interrupt/resume pair.

---

## 12. Test map to acceptance scenarios

| Test                              | Scenario | Proves invariant |
|-----------------------------------|----------|------------------|
| test_planner::test_rule_bypass    | S1       | I1               |
| test_planner::test_closed_catalog | S2       | I1               |
| test_ladder::test_medium_one_question | S3   | ladder no-loop   |
| test_ladder::test_low_handoff_payload | S4   | context carried  |
| test_envelope::test_filter_first  | S5       | tenant fence     |
| test_envelope::test_freshness     | S6       | stale = abstain  |
| test_scenarios::test_read_journey | S7       | I4               |
| test_corridor::test_preview_nonce | S8       | I3               |
| test_corridor::test_idempotent    | S9       | I3, I5           |
| test_corridor::test_rung_ceiling  | S10      | rung gates action|
| test_resume::test_kill_resume     | S11      | I5               |
| test_invariants::test_i2_advisory_row | S12  | I2               |
