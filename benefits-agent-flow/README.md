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
python -m pytest -q                  # 77 tests
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
        -> READ:  respond
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

## Fault drills

`MODEL_UNCITED_CLAIM=1`, `MODEL_TIMEOUT_ONCE=1`, `EVIDENCE_EMPTY=1|always`, `SOR_TIMEOUT_ONCE=1`
(commits then loses the response, so the corridor must reconcile by idempotency key rather than
resubmit). All four are exercised in `tests/test_invariants.py`.

## Tracing

Set `LANGSMITH_API_KEY` (and optionally `LANGSMITH_PROJECT`) to send runs to LangSmith; tool spans
carry their tool version and the payloads go through the same redaction as the audit log.
