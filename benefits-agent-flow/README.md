# benefits-agent-flow

The runtime *inside* the agent call, built to `design.md`: one LangGraph state machine per
conversational turn, over a fully mocked data plane. It is standalone — its own catalog,
decision table, capability bundles and fixtures — and shares nothing with `intent-router/`,
which remains the routing reference implementation.

Models interpret, tables decide. The classifier only proposes an intent; `decision_table.yaml`
(`table-v13`) picks the graph, the posture, the rung ceiling, the budgets and the evidence the
turn is allowed to read. Nothing the model emits can create a route, a tool call, a fact or a write.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q                  # 78 tests
python scripts/validate_config.py    # served-config build gate
uvicorn app.main:app --port 8200
```

The chat UI is at `http://localhost:8200/`: the tenant and participant come from the
context panel rather than the message, clarifying questions render as the exact options
the planner offered, a corridor preview renders as a Confirm button carrying the proposal
id and nonce, and the right panel replays `/trace` for the turn that just ran.

```bash
curl -s localhost:8200/turn -H 'content-type: application/json' -d '{
  "conversationId":"C-1","tenantId":"T-ACME","participantRef":"P-1001",
  "utterance":"change my 401k contribution to 8 percent"}'
# -> {"kind":"preview", "proposal":{...}, "nonce":"..."}

curl -s localhost:8200/confirm -H 'content-type: application/json' -d '{
  "conversationId":"C-1","proposalId":"PRP-...","nonce":"..."}'
# -> {"kind":"receipt", "receipt":{"verified_rate":"0.08", ...}}

curl -s localhost:8200/trace/C-1     # redacted per-node audit trail
```

## The graph

```
ingress -> planner -> gate_plan -> [read_evidence | read_facts | read_calc]
        -> assemble -> reason -> gate_output
        -> READ:  respond            (gate_output may send one failed read to `escalate` first)
        -> WRITE: corridor_propose -> corridor_preview -> corridor_wait_confirmation
                  -> corridor_revalidate -> corridor_execute -> corridor_verify -> respond
```

`corridor_wait_confirmation` is a LangGraph `interrupt()`, so the nonce is checked *inside* the
graph and `/confirm` resumes the same run from the SQLite checkpoint rather than starting a
second one. Killing the process between preview and confirm loses nothing (`tests/test_resume.py`).

## Governance

| Rung | Behaviour |
| --- | --- |
| 0 | handoff; the change is not available in this workspace |
| 1 | draft only, the participant submits it themselves |
| 2 | propose -> preview -> nonce -> revalidate -> execute -> verify -> receipt |
| 3 | same corridor without the human pause, plus notify |
| 4 | autonomous, same corridor |

The effective rung is `min(tenant bundle rung, row rung_max)`, so a table row can never grant a
tenant more autonomy than its bundle bought. A participant whose record belongs to another tenant
is refused at `gate_plan`, and evidence retrieval filters on tenant, plan and the version
effective on the service date before it ranks anything.

`gate_output` rejects a draft with an uncited sentence, a citation that is not in the envelope, a
number no envelope payload supports, or PII in the text; one retry, then a scripted fallback.
Missing required evidence abstains instead of guessing.

## Frontier escalation (deepagents)

The ladder is failed check -> one composer retry -> frontier -> human, and `escalate` is the
frontier rung: a deepagents research loop that gets one attempt at the same answer. It is
advisory by construction.

- Only a READ turn escalates. A WRITE turn goes to the human; the corridor is never model-driven.
- Its only tools are `list_evidence` and `read_evidence_item`, closures over the envelope the
  graph already assembled and authorized. No ODL, no SOR, no corridor tool, no network.
- deepagents' own scratchpad filesystem is mounted deny-write, and the loop is bounded by
  `ESCALATION_MAX_STEPS` (default 12).
- The table still picked the graph. The frontier answer re-enters `gate_output` and is rejected
  on the same citation/number/PII rules; a rejection or a provider error falls through to the
  human handoff.

Off unless `ESCALATION_BACKEND=deepagents`, and it needs the extra install (Python 3.11+):

```bash
pip install -r requirements-escalation.txt
ESCALATION_BACKEND=deepagents ESCALATION_MODEL=openai:gpt-4.1 OPENAI_API_KEY=sk-... \
  uvicorn app.main:app --port 8200
```

`tests/test_escalation.py` covers the ladder with a stubbed frontier;
`tests/test_escalation_deepagents.py` drives the real harness with a scripted model, so neither
needs a provider.

## Fault drills

`MODEL_UNCITED_CLAIM=1|always`, `MODEL_TIMEOUT_ONCE=1`, `EVIDENCE_EMPTY=1|always`, `SOR_TIMEOUT_ONCE=1`
(commits then loses the response, so the corridor must reconcile by idempotency key rather than
resubmit). All four are exercised in `tests/test_invariants.py`.

## Tracing

Tracing is off unless both `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set (optionally
`LANGSMITH_PROJECT`); without them every span is a no-op and nothing leaves the box:

```bash
LANGSMITH_TRACING=true LANGSMITH_API_KEY=lsv2_... LANGSMITH_PROJECT=benefits-agent-flow \
  uvicorn app.main:app --port 8200
```

One root run per turn (`turn:<conversation_id>`, tagged with the tenant, config versions as
metadata), with every node and tool call as child spans; tool spans carry their tool version and
payloads go through the same redaction as the audit log. `/trace/{conversation_id}` returns the
root run URL of the last turn as `langsmith_run_url`, and the chat UI links it above the node
list.
