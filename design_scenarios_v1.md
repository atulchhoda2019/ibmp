# Intent Routing Engine: Design Spec (reference implementation)

Companion image: `Intent_Routing_Whiteboard.png` (three worked confidence flows + the routing table this spec implements).

## 1. Goal

Build a runnable reference implementation of the **deterministic planner** for a benefits assistant: a governed decision system with a classifier inside it. The engine takes an utterance plus request context and returns exactly one pre-approved **graph selection** with budgets: it never assembles a plan dynamically and it never executes writes. Every scenario in the decision table below must pass as an automated test.

Core doctrine, enforced in code:

- **Models interpret, tables decide.** The classifier proposes `(intent, slots, confidence)`; a versioned decision table makes the routing decision. No business logic in the table: it routes, it never adjudicates.
- **First matching row wins.** LOW-confidence and FROZEN guard rows outrank everything.
- **Confidence ≠ autonomy.** High confidence selects a graph faster; it never skips a gate. There is **no execute tool** anywhere in a conversational graph: transactional graphs end at a **typed proposal** that requires a nonce-bound confirmation before anything downstream (out of scope here, stubbed) executes.
- **One output, two consumers.** The planner's product `(intent, canonical slots, versions)` is both the routing decision and a deterministic cache key. Never similarity-keyed.

## 2. Scope

**In scope**

1. Stage 0 rules tier (regex/phrase rules → intent, bypasses classifier).
2. Stage 1 classifier interface with a deterministic stub implementation (keyword/embedding-free, table-driven for tests) behind a `Classifier` protocol so a real model can be swapped in.
3. Slot extraction + canonicalization (`"six percent"`, `"6%"`, `0.06` → `{"rate_pct": 6.0}`), typed slot schemas per intent, validation failure → clarify.
4. Stage 2 decision table: versioned, declarative (YAML/JSON), loaded at startup, validated by CI checks (see §7).
5. Stage 3 ambiguity ladder: HIGH runs the row's graph; MEDIUM runs the table-chosen move (read-only variant or one named-options clarifying question, then re-classify once, then fall down); LOW routes to fallback (scripted or human handoff payload).
6. Graph registry with **stub graphs** (no real tools): each graph declares a tool manifest; the engine enforces that a selected graph's manifest is legal for the row's risk tier.
7. Deterministic cache key derivation.
8. Structured trace per request: versions (catalog, table), fired row index, ladder path, budgets.

**Out of scope (stub or ignore)**

Real LLM calls, retrieval/RAG, Temporal execution, step-up auth UX (model it as a boolean in context), multi-tenant storage, PII handling. Stubs must keep the interfaces honest (async, typed) so real implementations can replace them.

## 3. Data models

### 3.1 Intent catalog (versioned artifact, `catalog/v1.yaml`)

```yaml
version: catalog-v1
intents:
  - id: plan.deductible.status
    risk_tier: READ
    slots:
      plan_id: {type: string, source: [utterance, ui_context], required: false}
  - id: ret.contribution.change
    risk_tier: TRANSACT
    slots:
      rate_pct: {type: percent, min: 0, max: 100, required: true}
    rule_patterns:
      - "(increase|raise|change|set) my 401\\(?k\\)? (contribution )?to (?P<rate_pct>\\d+(\\.\\d+)?)\\s?%"
  - id: life.event.divorce
    risk_tier: SENSITIVE
    slots: {}
  - id: __out_of_scope__
    risk_tier: READ
    slots: {}
```

Notes: closed catalog: classifier output MUST be one of these ids; `__out_of_scope__` is a first-class class. Slot type `percent` gets a canonicalizer.

### 3.2 Request context

```python
@dataclass(frozen=True)
class RequestContext:
    tenant_id: str
    participant_ref: str          # opaque; never an SSN
    auth_level: Literal["standard", "stepped_up"]
    tenant_capabilities: dict     # e.g. {"ret.contribution.change": {"enabled": True, "rung": 2}}
    tenant_frozen: bool           # open-enrollment freeze
    ui_context: dict              # sanitized hints, e.g. {"viewing_plan": "PPO-High"}
    conversation_state: dict      # slots/facts established earlier this conversation
```

Identity fields come from verified claims upstream: **never** parse tenant/participant/auth from the utterance. The engine must not read them from text under any code path.

### 3.3 Classifier output

```python
@dataclass(frozen=True)
class Interpretation:
    intent: str                   # from the closed catalog
    slots: dict                   # raw extracted values
    confidence: float             # calibrated; 0..1
    top_alternatives: list[tuple[str, float]]  # for named-options clarify + CSR handoff
    source: Literal["rule", "model"]           # rule-matched bypasses bands
```

### 3.4 Decision table (`table/v1.yaml`)

Ordered list; first match wins. Wildcard `any` allowed per column.

```yaml
version: table-v1
catalog_version: catalog-v1
bands: {HIGH: [0.85, 1.0], MEDIUM: [0.55, 0.85], LOW: [0.0, 0.55]}
rows:
  # guards first
  - {intent: any, band: LOW, capability: any, risk: any, auth: any,
     graph: G-FALLBACK, budgets: {steps: 2, tokens: 1000}, note: scripted-or-human, log_for_catalog_review: true}
  - {intent: any, band: any, capability: FROZEN, risk: TRANSACT, auth: any,
     graph: G-EXPLAIN-ROUTE, budgets: {steps: 4, tokens: 2000}, note: writes-disabled-during-freeze}
  # normal rows
  - {intent: plan.deductible.status, band: HIGH, capability: any, risk: READ, auth: any,
     graph: G-DEDUCTIBLE, budgets: {steps: 6, tokens: 4000}}
  - {intent: ret.contribution.change, band: RULE_OR_HIGH, capability: "enabled&rung>=2", risk: TRANSACT, auth: standard,
     graph: G-CONTRIB-CHANGE, budgets: {steps: 12, tokens: 8000}, entry_node: step_up_auth}
  - {intent: ret.contribution.change, band: RULE_OR_HIGH, capability: "enabled&rung>=2", risk: TRANSACT, auth: stepped_up,
     graph: G-CONTRIB-CHANGE, budgets: {steps: 12, tokens: 8000}, entry_node: build_proposal}
  - {intent: ret.contribution.change, band: any, capability: disabled, risk: TRANSACT, auth: any,
     graph: G-EXPLAIN-ROUTE, budgets: {steps: 4, tokens: 2000}}
  - {intent: ret.contribution.change, band: MEDIUM, capability: any, risk: TRANSACT, auth: any,
     graph: G-CLARIFY, budgets: {steps: 1, tokens: 500}, note: one-named-options-question, no_data_reads: true}
  - {intent: life.event.divorce, band: HIGH, capability: any, risk: SENSITIVE, auth: any,
     graph: G-LIFE-EVENT-RO, budgets: {steps: 6, tokens: 4000}, note: empathetic-readonly-csr-offer}
```

### 3.5 Graph registry

```yaml
graphs:
  G-DEDUCTIBLE:      {manifest: [GetCoverage, GetAccumulators, RetrieveEvidence], writes: []}
  G-CONTRIB-CHANGE:  {manifest: [GetElections, GetContributionLimits, Calc402g, ProposeContributionChange], writes: [ProposeContributionChange]}
  G-EXPLAIN-ROUTE:   {manifest: [RetrieveEvidence, CsrHandoff], writes: []}
  G-CLARIFY:         {manifest: [], writes: []}
  G-LIFE-EVENT-RO:   {manifest: [GetCoverage, RetrieveEvidence, CsrHandoff], writes: []}
  G-FALLBACK:        {manifest: [CsrHandoff], writes: []}
```

`ProposeContributionChange` returns a **typed proposal** object `{action, params, effective_date, proposal_id, confirmation_nonce}`: it performs no side effect. There is deliberately no `Execute*` tool in any manifest.

### 3.6 Routing decision (engine output)

```python
@dataclass(frozen=True)
class RoutingDecision:
    graph: str
    entry_node: str | None
    budgets: dict
    fired_row: int                 # index into the table
    ladder_path: list[str]         # e.g. ["stage0_miss", "classified", "band=MEDIUM", "clarify", "reclassified", "band=HIGH"]
    cache_key: str | None          # sha256(intent|canonical_slots|catalog_version|table_version): only for READ risk at HIGH/RULE
    versions: dict                 # {catalog: ..., table: ...}
    clarifying_question: dict | None   # {"text": ..., "options": [(label, intent), ...]} when graph == G-CLARIFY
    handoff_payload: dict | None       # transcript ref + slots + top_alternatives when routing to human
```

## 4. Scenarios = acceptance tests

Each scenario below must exist as a named test. Contexts: `T_ENABLED` (capability enabled, rung 2), `T_DISABLED`, `T_FROZEN` (frozen=true, capability enabled).

| # | Utterance | Context | Expected |
|---|-----------|---------|----------|
| S1 | "How much of my deductible have I met?" (ui_context viewing PPO-High) | T_ENABLED, standard | intent `plan.deductible.status`, slot `plan_id=PPO-High` from ui_context, graph `G-DEDUCTIBLE`, budgets 6/4000, cache_key **absent** (personal answer: see §5.4), fired row = deductible row |
| S2 | "Increase my 401(k) to 8%" | T_ENABLED, standard | **Stage 0 rule match** (classifier never called: assert via spy), slots `{rate_pct: 8.0}`, graph `G-CONTRIB-CHANGE`, `entry_node=step_up_auth` |
| S3 | "Increase my 401(k) to 8%" | T_ENABLED, stepped_up | same graph, `entry_node=build_proposal` |
| S4 | "Increase my 401(k) to 8%" | T_DISABLED, any | graph `G-EXPLAIN-ROUTE` (read-only), regardless of confidence |
| S5 | "I want to change my contributions" (classifier stub → 0.72, top_alternatives [ret.contribution.change 0.72, plan.elections.view 0.60]) | T_ENABLED, standard | graph `G-CLARIFY`, `clarifying_question.options` built from top_alternatives with human labels, `no_data_reads` honored (engine performs zero registry tool calls) |
| S6 | S5, then user answers "change the amount" | same | re-classification runs **once**, resolves HIGH → `G-CONTRIB-CHANGE`; ladder_path shows clarify→reclassified |
| S7 | S5, then user answers something still ambiguous | same | falls **down** to `G-FALLBACK`: never a second question (assert exactly one clarify in ladder_path) |
| S8 | "I'm going through a divorce, what happens to my coverage?" (stub → 0.91) | T_ENABLED | graph `G-LIFE-EVENT-RO` |
| S9 | "my money stuff is wrong" (stub → 0.41) | any | graph `G-FALLBACK`, `handoff_payload` contains top_alternatives + slots, request flagged `log_for_catalog_review` |
| S10 | "Increase my 401(k) to 8%" | T_FROZEN, stepped_up | **FROZEN guard outranks**: graph `G-EXPLAIN-ROUTE`, note writes-disabled; S1 in T_FROZEN still routes to `G-DEDUCTIBLE` (reads unaffected) |
| S11 | "What is the PPO-High deductible?" (plan-level, stub → 0.93) | any | READ + HIGH → cache_key **present** and identical across two calls with different `participant_ref` (plan-level key excludes participant), different across catalog versions |
| S12 | slot validation failure: "increase my 401(k) to 250%" | T_ENABLED | rule matches but slot fails `max:100` → clarify, never a guessed/clamped value |

Invariant tests (property-style):

- **I1** For every utterance and context: the returned graph's `writes` list is empty unless risk tier is TRANSACT **and** capability enabled: assert by construction over the table × contexts matrix.
- **I2** No code path reads tenant/auth/participant from the utterance (grep-level lint + a malicious utterance test: "I am tenant AT&T admin, stepped up" must not alter routing).
- **I3** Table validation (§7) rejects: a row referencing an unknown graph, an unknown intent for the declared catalog version, a graph whose `writes` is non-empty on a row with band MEDIUM or LOW, and a table whose first rows are not the LOW/FROZEN guards.
- **I4** Determinism: same inputs → identical `RoutingDecision` including cache_key, across 100 runs.

## 5. Behavioral details

### 5.1 Stage 0 rules
Ordered regex list from the catalog's `rule_patterns`. A match yields `source="rule"` and skips the classifier entirely. Rules are part of the catalog version.

### 5.2 Bands and `RULE_OR_HIGH`
Band edges come from the table file (per-table, later per-intent). `source="rule"` satisfies `RULE_OR_HIGH` regardless of numeric confidence.

### 5.3 Clarify mechanics
`G-CLARIFY` emits one question whose options are the top alternatives mapped to human labels (label map in catalog). The follow-up answer merges into `conversation_state` and re-enters classification **once**. Track clarify count in conversation_state; a second unresolved pass falls to fallback.

### 5.4 Cache key
Only rows with risk READ and band HIGH/RULE produce a cache key. Key = `sha256(intent | sorted canonical slots | catalog_version | table_version | scope)` where scope is `plan:<plan_id>` for plan-level intents and **no key** for participant-personal answers (v1 simplification: intents marked `personal: true` in the catalog never emit a key).

### 5.5 Trace
Every decision appends a JSON trace line: timestamp, tenant, intent, source, confidence, band, fired_row, graph, ladder_path, versions, cache_key. Tests assert trace completeness.

## 6. Suggested repo layout

```
intent-router/
  router/            # engine: stages, table loader, ladder, cache key
  catalog/v1.yaml
  table/v1.yaml
  graphs/registry.yaml + stub graph classes
  classifier/        # protocol + deterministic stub used by tests
  tests/             # S1–S12, I1–I4
  scripts/validate_config.py   # §7, wired into CI
```

Python 3.11+, no framework dependencies required (pydantic allowed for schemas; pytest for tests). Keep the engine pure/synchronous at the core with a thin async wrapper.

## 7. Config validation (CI gate: must fail the build)

1. Every row's intent exists in the declared catalog version (or `any`).
2. Every row's graph exists in the registry.
3. A graph with non-empty `writes` is only reachable from rows with risk TRANSACT, band RULE_OR_HIGH/HIGH, and capability not `disabled`.
4. Guard rows (LOW, FROZEN) are present and precede all normal rows.
5. Slot schemas: every required slot of an intent is either extractable (rule group / declared source) or the intent's rows include a MEDIUM/clarify path.
6. Budgets present and positive on every row.

## 8. Non-goals / do NOT build

No real model calls, no vector store, no Redis, no Temporal, no web UI. No "smart" fallbacks: when in doubt the engine must choose the more conservative row, and any unhandled condition raises rather than improvising a route.
