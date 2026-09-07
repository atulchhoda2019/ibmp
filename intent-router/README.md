# intent-router

Reference implementation of the deterministic planner described in [`../design.md`](../design.md),
sketched on `../Intent_Routing_Whiteboard_1.png` and revised by `../Benefits Platform 1.png`: an
utterance plus request context in, exactly one pre-approved graph selection with budgets out.
Models interpret, tables decide.

## Layout

```
app/               FastAPI assistant: /api/chat over route(), the transaction corridor, the
                   evidence validator, and the browser UI in app/static
router/            engine: stage 0 rules, bands, ladder, slots, cache key, trace
catalog/v1.yaml    versioned closed intent catalog (slot schemas + stage 0 rule patterns)
table/v1.yaml      versioned decision table: ordered rows, first match wins
graphs/            registry (tool manifests) + stub graphs that end at a typed proposal
classifier/        Classifier protocol + deterministic stub used by the tests
tests/             S1-S12 scenarios and I1-I4 invariants
scripts/           validate_config.py, the CI gate from design §7
```

## Run

```bash
pip install -r requirements.txt
python scripts/validate_config.py   # fails the build on an illegal table
python -m pytest -q
```

## Assistant UI

```bash
uvicorn app.server:app --reload --port 8000   # then open http://localhost:8000
```

The browser assistant asks the clarifying questions the table produces: `G-CLARIFY` renders as
clickable named options, picking one resolves the conversation on a single reclassification, and a
transactional turn ends at a typed proposal with a Confirm button bound to its nonce. The right-hand
panel shows the fired row, band (and whose edges fired), posture, capability, budgets, slots, ladder
path, cache key and config versions for the last turn; the context controls (capability, autonomy
rung, auth level, freeze, VIEW/CHANGE grants, evidence age, viewing plan) stand in for verified
request context, which is never parsed from the message. Downstream systems are stubbed — the UI is a
consumer of the planner, not a second one.

Things worth trying in it: drop the rung to 1 and watch the same sentence draft instead of propose;
raise it to 3 and watch it execute and hand back a receipt; confirm the same proposal twice and watch
the duplicate collapse onto the first receipt; clear CHANGE and watch the write become an explanation
*before* any data is fetched; set evidence age to 90 and watch the deductible answer abstain.

## Use

```python
from classifier import StubClassifier
from router import RequestContext, build_router, load_catalog

catalog = load_catalog("catalog/v1.yaml")
router = build_router(classifier=StubClassifier(catalog=catalog))

decision = router.route(
    "Increase my 401(k) to 8%",
    RequestContext(
        tenant_id="tenant-acme",
        participant_ref="participant-1",
        auth_level="standard",
        tenant_capabilities={"ret.contribution.change": {"enabled": True, "rung": 2}},
        tenant_frozen=False,
        ui_context={},
        conversation_state={},
    ),
)
# decision.graph == "G-CONTRIB-CHANGE", decision.entry_node == "step_up_auth"
```

`route_async` is a thin wrapper over the same pure core.

## Autonomy rungs

The rung the tenant bought — not the model's confidence — decides how far a transactional turn goes.
Rows are ordered by descending rung, so the highest one the tenant is entitled to wins:

| rung | graph | what happens |
| --- | --- | --- |
| 1 | `G-CONTRIB-DRAFT` | drafts the change for a human; zero direct action |
| 2 | `G-CONTRIB-CHANGE` → `G-CONTRIB-COMMIT` | typed proposal, then a nonce-bound confirmation executes it |
| 3 | `G-CONTRIB-AUTO` | executes the reversible change and notifies, stepped-up auth required |
| 4 | `G-CONTRIB-AUTO` | autonomous and audited |

## Doctrine enforced in code

- Guard rows first: LOW confidence, open-enrollment FROZEN, and a missing entitlement outrank every
  normal row. VIEW and CHANGE are separate grants, resolved before a graph runs and therefore before
  any data is fetched.
- Band edges are versioned with routing and may be measured per intent; the trace records which edge
  set fired.
- Confidence never buys autonomy: a `RULE`/HIGH band picks a graph faster, it never skips a gate. An
  `Execute*` tool is legal only in a graph that declares it in `executes` with governance — minimum
  rung, reversible, and whether a live confirmation is required — and the config gate refuses any row
  that could reach such a graph below its rung.
- The corridor owns confirmations, not the client: a confirmation must carry the nonce issued with
  that exact proposal, a superseded or expired proposal is dead, and the command key is the proposal
  id, so a retried confirmation collapses onto the first receipt instead of writing twice.
- Answers that rest on retrieved evidence carry an evidence policy on the row; stale or uncited
  evidence abstains rather than quoting a number.
- One clarifying question per conversation: a second unresolved pass falls down to `G-FALLBACK`.
- Out-of-schema slot values clarify; they are never clamped or guessed. Absent required slots are
  reported in `missing_slots` for the selected graph to collect.
- Cache keys are derived from `(intent, canonical slots, versions, scope)` — never similarity-keyed,
  and never emitted for participant-personal answers.
- Identity (tenant, participant, auth) comes only from `RequestContext`; no code path reads it from
  the utterance, and a test lints for it.
- Any unhandled condition raises `RoutingError` rather than improvising a route.

## Model plane

Models interpret and phrase; they never choose their own route or their own weights. A request
carries `(task, domain, posture, tenant)` from the *deterministic* routing decision, and
`modelplane/routing.yaml` — an ordered, first-match table — maps that to exactly one served config
in `modelplane/registry.yaml`:

```
deterministic router → model routing table → served config → gateway → validator
```

A served config is a base model plus at most one adapter, its decoding parameters, cost class,
rollout ring and its eval record. The adapter ladder is `tenant → task → domain`, and a tenant-pinned
row is only legal when that config's eval shows a positive measured gap. A config whose eval is
missing, failing, or judged by its own model family never loads; `scripts/validate_config.py` also
refuses a serving pointer aimed at a canary-ring config, an unroutable required `(task, domain,
posture)`, and a served config no row can reach. Rollback is a pointer flip in `serving:`.

Composition is validated, not trusted: `validate_composition` rejects any number the tools did not
produce, so the model can only restate facts the deterministic path already computed. Failure is a
fixed ladder — **one** retry on the same config, then the frontier config, then `EscalatedToHuman` —
and every attempt is recorded on the turn's `model` trace and counted by the scorecard
(`/api/models`): validator catches, abstentions, escalations, handoffs, containment and cost per
successful answer.

The classifier can be model-backed (`classifier/model.py`), but it only ever proposes
`(intent, slots, confidence, alternatives)` against the closed catalog — an intent outside the
catalog degrades to `OUT_OF_SCOPE` rather than becoming a new capability.

```bash
MODEL_BACKEND=openai OPENAI_API_KEY=... uvicorn app.server:app   # real calls
uvicorn app.server:app                                          # deterministic fake backend
```

## LangGraph execution

`GRAPH_BACKEND=langgraph` runs the approved graphs as compiled `StateGraph`s instead of the stub.
The graphs are compiled *from* the registry manifest, so the reachable node set is exactly the tool
set the graph was approved for and an off-manifest tool has no node to run in.

`build_corridor` compiles the board's transaction branch — `extract_action → policy_check →
validate_action → create_preview → wait_confirmation → revalidate → execute → verify → receipt` —
where `wait_confirmation` is a real LangGraph `interrupt()` over a checkpointer. The confirmation
resumes *that* run, keyed on the proposal id and the nonce issued with it; a wrong nonce or another
proposal's confirmation cannot resume it, and cannot open a second transaction. Routing budgets map
to the runtime recursion limit.
