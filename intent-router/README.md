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
