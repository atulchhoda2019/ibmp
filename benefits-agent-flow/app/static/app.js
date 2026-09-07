const transcript = document.getElementById("transcript");
const composer = document.getElementById("composer");
const input = document.getElementById("utterance");
const tracePanel = document.getElementById("trace");
const suggestions = document.getElementById("suggestions");
const tenantSelect = document.getElementById("tenant");
const participantSelect = document.getElementById("participant");
const conversationField = document.getElementById("conversation");

const PARTICIPANTS = {
  "T-ACME": [
    ["P-1001", "P-1001 · mid-career, clean path"],
    ["P-1002", "P-1002 · high earner near the deferral cap"],
    ["P-1003", "P-1003 · new hire, not yet eligible"],
    ["P-1004", "P-1004 · PPO, physical therapy"],
  ],
  "T-ZEN": [["P-2001", "P-2001 · draft-only tenant"]],
  "T-NOVA": [["P-3001", "P-3001 · execute-and-notify tenant"]],
  "T-LOCK": [["P-4001", "P-4001 · view-only tenant"]],
};

const SUGGESTIONS = [
  "How much of my deductible have I met?",
  "Change my 401(k) contribution to 8 percent",
  "What is my 401(k) balance?",
  "I want to change my contributions",
  "Am I eligible for the 401(k) yet?",
  "What would physical therapy cost me?",
  "Change my 401(k) contribution to 90 percent",
  "my money stuff is wrong",
];

// Carried into the next turn only, so one clarify pass resolves by exact option match.
let clarify = null;
let conversationId = "";

const TRACE_EMPTY = "Send a message to see the nodes that ran.";

function newConversation() {
  conversationId = `C-${Math.random().toString(36).slice(2, 10)}`;
  conversationField.value = conversationId;
  clarify = null;
  transcript.innerHTML = "";
  tracePanel.textContent = TRACE_EMPTY;
}

function fillParticipants() {
  participantSelect.innerHTML = "";
  for (const [ref, label] of PARTICIPANTS[tenantSelect.value]) {
    const option = document.createElement("option");
    option.value = ref;
    option.textContent = label;
    participantSelect.appendChild(option);
  }
}

function bubble(role, text) {
  const node = document.createElement("div");
  node.className = `msg ${role}`;
  node.textContent = text;
  transcript.appendChild(node);
  transcript.scrollTop = transcript.scrollHeight;
  return node;
}

function renderCitations(citations) {
  if (!citations || !citations.length) return;
  const row = document.createElement("div");
  row.className = "citations";
  row.innerHTML = citations.map((id) => `<span>${id}</span>`).join("");
  transcript.appendChild(row);
}

function renderOptions(options) {
  if (!options || !options.length) return;
  const row = document.createElement("div");
  row.className = "options";
  for (const option of options) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = option.label;
    button.addEventListener("click", () => send(option.label));
    row.appendChild(button);
  }
  transcript.appendChild(row);
  transcript.scrollTop = transcript.scrollHeight;
}

function renderPreview(body) {
  const proposal = body.proposal;
  const card = document.createElement("div");
  card.className = "proposal";
  card.innerHTML = `
    <h3>Typed proposal · awaiting confirmation</h3>
    <div>${proposal.action} for ${proposal.participant_ref} · ${JSON.stringify(proposal.current)} → ${JSON.stringify(proposal.requested)}</div>
    <div>checks passed: ${proposal.validations.join(", ")}</div>
    <div>${body.undo_window || ""}</div>
    <code>nonce ${proposal.nonce} · expires ${proposal.expires_at}</code>
  `;
  const confirm = document.createElement("button");
  confirm.type = "button";
  confirm.textContent = "Confirm change";
  confirm.addEventListener("click", async () => {
    confirm.disabled = true;
    const response = await fetch("/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversationId,
        proposalId: proposal.proposal_id,
        nonce: proposal.nonce,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      bubble("system", `Confirmation refused: ${JSON.stringify(payload.detail)}`);
      confirm.disabled = false;
      return;
    }
    renderReceipt(payload.receipt);
    await refreshTrace();
  });
  card.appendChild(confirm);
  transcript.appendChild(card);
  transcript.scrollTop = transcript.scrollHeight;
}

function renderReceipt(receipt) {
  if (!receipt) return;
  const card = document.createElement("div");
  card.className = "receipt";
  card.innerHTML = `
    <h3>Receipt${receipt.replayed ? " \u00b7 duplicate collapsed" : ""}</h3>
    <div>${receipt.action} \u00b7 ${JSON.stringify(receipt.applied)}</div>
    <div>verified in the system of record: rate ${receipt.verified_rate}</div>
    <code>idempotency key ${receipt.idempotency_key} \u00b7 ${receipt.committed_at}</code>
  `;
  transcript.appendChild(card);
  transcript.scrollTop = transcript.scrollHeight;
}

function renderDraft(body) {
  const card = document.createElement("div");
  card.className = "proposal";
  card.innerHTML = `
    <h3>Draft for a human to submit</h3>
    <div>${body.form.action} \u00b7 ${JSON.stringify(body.form.requested)}</div>
    <div>${body.form.instructions}</div>
  `;
  transcript.appendChild(card);
}

function renderResponse(body, utterance) {
  switch (body.kind) {
    case "clarify":
      bubble("assistant", body.question);
      renderOptions(body.options);
      clarify = { options: body.options, utterance };
      return;
    case "handoff":
      bubble("assistant", body.summary);
      return;
    case "preview":
      bubble("assistant", "Here is exactly what I would change. Nothing is submitted until you confirm.");
      renderCitations(body.citations);
      renderPreview(body);
      return;
    case "receipt":
      renderReceipt(body.receipt);
      return;
    case "draft":
      renderDraft(body);
      return;
    default:
      bubble("assistant", body.text || "");
      renderCitations(body.citations);
      if (body.disclosures) body.disclosures.forEach((line) => bubble("system", line));
  }
}

const TRACE_FIELDS = [
  "source", "confidence", "band", "row", "graph_id", "posture", "allowed", "reason",
  "kind", "items", "count", "passed", "reasons", "citations", "stale", "refetched",
  "rung", "executed", "replayed", "verified", "model", "attempt", "timeout",
];

function renderTrace(events) {
  if (!events.length) {
    tracePanel.textContent = TRACE_EMPTY;
    return;
  }
  const latest = events[events.length - 1].turn_id;
  tracePanel.innerHTML = events
    .filter((event) => event.turn_id === latest)
    .map((event) => {
      const fields = TRACE_FIELDS
        .filter((key) => event[key] !== undefined && event[key] !== null)
        .map((key) => `${key}=${JSON.stringify(event[key])}`)
        .join(" ");
      return `<div><span class="node">${event.node}</span> <span class="fields">${fields}</span></div>`;
    })
    .join("");
}

async function refreshTrace() {
  const response = await fetch(`/trace/${conversationId}`);
  if (!response.ok) return;
  const body = await response.json();
  renderTrace(body.events);
}

async function send(utterance) {
  if (!utterance.trim()) return;
  bubble("user", utterance);
  input.value = "";

  const uiContext = clarify
    ? { clarify_rounds: 1, clarify_options: clarify.options, clarify_utterance: clarify.utterance }
    : {};
  const carried = clarify;
  clarify = null;

  const response = await fetch("/turn", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversationId,
      tenantId: tenantSelect.value,
      participantRef: participantSelect.value,
      utterance,
      uiContext,
    }),
  });
  const body = await response.json();
  if (!response.ok) {
    const detail = body.detail || {};
    bubble("system", detail.message || JSON.stringify(detail));
    clarify = carried;
    return;
  }
  renderResponse(body, utterance);
  await refreshTrace();
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  send(input.value);
});

tenantSelect.addEventListener("change", () => {
  fillParticipants();
  newConversation();
  bubble("system", `Switched to ${tenantSelect.value} · ${participantSelect.value}. New conversation.`);
});

// The thread is bound to one participant: switching identity must not inherit a live proposal.
participantSelect.addEventListener("change", () => {
  newConversation();
  bubble("system", `Switched to ${participantSelect.value}. New conversation.`);
});

document.getElementById("reset").addEventListener("click", () => {
  newConversation();
  bubble("system", "New conversation.");
});

for (const suggestion of SUGGESTIONS) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = suggestion;
  button.addEventListener("click", () => send(suggestion));
  suggestions.appendChild(button);
}

fillParticipants();
newConversation();
bubble("assistant", "Hi! I can help with your benefits. Ask about your deductible, your 401(k) balance, or changing your contribution rate.");
