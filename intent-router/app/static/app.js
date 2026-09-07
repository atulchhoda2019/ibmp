const transcript = document.getElementById("transcript");
const composer = document.getElementById("composer");
const input = document.getElementById("utterance");
const decisionPanel = document.getElementById("decision");
const suggestions = document.getElementById("suggestions");

const SUGGESTIONS = [
  "How much of my deductible have I met?",
  "Increase my 401(k) to 8%",
  "I want to change my contributions",
  "What is the PPO-High deductible?",
  "I'm going through a divorce, what happens to my coverage?",
  "increase my 401(k) to 250%",
  "my money stuff is wrong",
];

let conversationState = {};

function requestContext() {
  return {
    capability: document.getElementById("capability").value,
    auth_level: document.getElementById("auth").value,
    tenant_frozen: document.getElementById("frozen").checked,
    viewing_plan: document.getElementById("plan").value,
    conversation_state: conversationState,
  };
}

function bubble(role, text) {
  const node = document.createElement("div");
  node.className = `msg ${role}`;
  node.textContent = text;
  transcript.appendChild(node);
  transcript.scrollTop = transcript.scrollHeight;
  return node;
}

function renderOptions(options) {
  if (!options.length) return;
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

function renderProposal(proposal) {
  if (!proposal) return;
  const card = document.createElement("div");
  card.className = "proposal";
  card.innerHTML = `
    <h3>Typed proposal · awaiting confirmation</h3>
    <div>${proposal.action} · ${JSON.stringify(proposal.params)} · effective ${proposal.effective_date}</div>
    <code>nonce ${proposal.confirmation_nonce}</code>
  `;
  const confirm = document.createElement("button");
  confirm.type = "button";
  confirm.textContent = "Confirm change";
  confirm.addEventListener("click", async () => {
    confirm.disabled = true;
    const response = await fetch("/api/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        proposal_id: proposal.proposal_id,
        confirmation_nonce: proposal.confirmation_nonce,
      }),
    });
    const body = await response.json();
    bubble("system", body.detail);
  });
  card.appendChild(confirm);
  transcript.appendChild(card);
  transcript.scrollTop = transcript.scrollHeight;
}

function renderDecision(decision, toolCalls) {
  decisionPanel.classList.remove("empty");
  const entries = [
    ["intent", decision.intent],
    ["band", `${decision.band} (${decision.source}, ${decision.confidence.toFixed(2)})`],
    ["fired row", decision.fired_row],
    ["entry node", decision.entry_node || "—"],
    ["budgets", `${decision.budgets.steps} steps · ${decision.budgets.tokens} tokens`],
    ["slots", JSON.stringify(decision.slots)],
    ["tools run", toolCalls.length ? toolCalls.join(", ") : "none"],
    ["cache key", decision.cache_key ? `${decision.cache_key.slice(0, 16)}…` : "not cacheable"],
    ["versions", `${decision.versions.catalog} · ${decision.versions.table}`],
  ];
  if (decision.note) entries.push(["note", decision.note]);
  if (decision.log_for_catalog_review) entries.push(["flagged", "catalog review"]);

  decisionPanel.innerHTML = `<div class="graph">${decision.graph}</div><dl>${entries
    .map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`)
    .join("")}</dl><div class="ladder">${decision.ladder_path
    .map((step) => `<span>${step}</span>`)
    .join("")}</div>`;
}

async function send(utterance) {
  if (!utterance.trim()) return;
  bubble("user", utterance);
  input.value = "";

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ utterance, context: requestContext() }),
  });
  if (!response.ok) {
    const error = await response.json();
    bubble("system", `Routing refused: ${error.detail}`);
    return;
  }
  const body = await response.json();
  conversationState = body.conversation_state;
  bubble("assistant", body.assistant);
  renderOptions(body.options);
  renderProposal(body.proposal);
  renderDecision(body.decision, body.tool_calls);
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  send(input.value);
});

document.getElementById("reset").addEventListener("click", () => {
  conversationState = {};
  transcript.innerHTML = "";
  decisionPanel.className = "decision empty";
  decisionPanel.textContent = "Send a message to see the fired row.";
  bubble("system", "Conversation reset.");
});

for (const suggestion of SUGGESTIONS) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = suggestion;
  button.addEventListener("click", () => send(suggestion));
  suggestions.appendChild(button);
}

bubble("assistant", "Hi! I can help with your benefits. Ask me about your deductible, your 401(k) contribution, or a life event.");
