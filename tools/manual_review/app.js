const state = { rows: [], progress: {}, labels: [], batch: 2, pendingOnly: true };

const byId = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
const labelList = (value) => String(value || "").split(",").map((x) => x.trim()).filter(Boolean);

async function loadReview() {
  const response = await fetch("/api/review", { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Could not load review data");
  state.rows = payload.rows;
  state.progress = payload.progress;
  state.labels = payload.allowed_labels;
  render();
}

function groupedExamples() {
  const grouped = new Map();
  for (const row of state.rows) {
    const id = Number(row.example_id);
    if (!grouped.has(id)) grouped.set(id, []);
    grouped.get(id).push(row);
  }
  return grouped;
}

function render() {
  const validated = state.rows.filter((row) => row.human_status === "validated").length;
  byId("progress-count").textContent = `${validated} / ${state.rows.length} comparisons`;
  byId("progress-detail").textContent = `${Math.round(validated / state.rows.length * 100)}% human validated`;
  renderTabs();
  renderExamples();
}

function renderTabs() {
  const validated = new Set(state.progress.human_validated_batches || []);
  byId("batch-tabs").innerHTML = [1, 2, 3, 4, 5].map((batch) => {
    const start = (batch - 1) * 10 + 1;
    const end = batch * 10;
    const classes = [batch === state.batch ? "active" : "", validated.has(batch) ? "complete" : ""].filter(Boolean).join(" ");
    return `<button class="${classes}" data-batch="${batch}">Batch ${batch} · ${start}–${end}</button>`;
  }).join("");
  document.querySelectorAll("[data-batch]").forEach((button) => {
    button.addEventListener("click", () => { state.batch = Number(button.dataset.batch); render(); });
  });
}

function renderExamples() {
  const start = (state.batch - 1) * 10 + 1;
  const end = state.batch * 10;
  const examples = [...groupedExamples().entries()].filter(([id, rows]) =>
    id >= start && id <= end && (!state.pendingOnly || rows.some((row) => row.human_status !== "validated"))
  );
  const target = byId("review-list");
  if (!examples.length) {
    target.replaceChildren(byId("empty-template").content.cloneNode(true));
    return;
  }
  target.innerHTML = examples.map(([id, rows]) => renderExample(id, rows)).join("");
  bindActions();
}

function renderExample(id, rows) {
  const first = rows[0];
  const canAcceptAll = rows.every((row) => row.suggested_labels && row.human_status !== "validated");
  return `<article class="example-card" data-example="${id}">
    <header class="example-header">
      <div><h2>Example ${id}</h2><div class="clip-id">${escapeHtml(first.clip_id)}</div></div>
      <button class="accept-all" data-accept-example="${id}" ${canAcceptAll ? "" : "disabled"}>Accept all five</button>
    </header>
    <p class="reference"><strong>Reference:</strong> ${escapeHtml(first.reference)}</p>
    <table class="model-table"><thead><tr><th>Model</th><th>Prediction</th><th>Suggestion</th><th>Your decision</th></tr></thead>
    <tbody>${rows.map(renderModelRow).join("")}</tbody></table>
  </article>`;
}

function renderModelRow(row) {
  const labels = labelList(row.suggested_labels);
  const long = row.prediction.length > 360;
  const prediction = long
    ? `<details><summary>Long prediction (${row.prediction.length} characters)</summary>${escapeHtml(row.prediction)}</details>`
    : escapeHtml(row.prediction);
  const chips = labels.length ? labels.map((label) => `<span class="chip">${label}</span>`).join("") : "<em>Not prepared</em>";
  const selected = row.human_status === "validated" ? labelList(row.human_labels) : labels;
  const options = state.labels.map((label) => `<label><input type="checkbox" value="${label}" ${selected.includes(label) ? "checked" : ""}> ${label}</label>`).join("");
  return `<tr class="model-row" data-row="${row.example_id}-${row.model_code}">
    <td class="model-cell"><span class="model-code">${row.model_code}</span><span class="model-name">${escapeHtml(row.model_name)}</span></td>
    <td class="prediction">${prediction}</td>
    <td class="suggestion"><div class="chips">${chips}</div><div class="confidence">${escapeHtml(row.confidence)}</div><div class="evidence">${escapeHtml(row.evidence)}</div></td>
    <td class="actions">
      <span class="status ${row.human_status}">${row.human_status}</span>
      <div class="button-row">
        <button class="primary" data-accept="${row.example_id}-${row.model_code}" ${row.suggested_labels ? "" : "disabled"}>Correct</button>
        <button data-edit="${row.example_id}-${row.model_code}">Edit labels</button>
        ${row.human_status === "validated" ? `<button data-reset="${row.example_id}-${row.model_code}">Reset</button>` : ""}
      </div>
      <div class="edit-panel hidden" data-panel="${row.example_id}-${row.model_code}">
        <div class="label-options">${options}</div>
        <textarea placeholder="Optional human note">${escapeHtml(row.human_notes)}</textarea>
        <div class="button-row"><button class="primary" data-save="${row.example_id}-${row.model_code}">Save correction</button></div>
      </div>
    </td>
  </tr>`;
}

function bindActions() {
  document.querySelectorAll("[data-accept]").forEach((button) => button.addEventListener("click", () => {
    const [example_id, model_code] = button.dataset.accept.split("-");
    save([{ example_id, model_code, action: "accept" }]);
  }));
  document.querySelectorAll("[data-edit]").forEach((button) => button.addEventListener("click", () => {
    document.querySelector(`[data-panel="${button.dataset.edit}"]`).classList.toggle("hidden");
  }));
  document.querySelectorAll("[data-save]").forEach((button) => button.addEventListener("click", () => {
    const key = button.dataset.save;
    const [example_id, model_code] = key.split("-");
    const panel = document.querySelector(`[data-panel="${key}"]`);
    const human_labels = [...panel.querySelectorAll("input:checked")].map((input) => input.value);
    const human_notes = panel.querySelector("textarea").value;
    save([{ example_id, model_code, action: "correct", human_labels, human_notes }]);
  }));
  document.querySelectorAll("[data-reset]").forEach((button) => button.addEventListener("click", () => {
    const [example_id, model_code] = button.dataset.reset.split("-");
    save([{ example_id, model_code, action: "reset" }]);
  }));
  document.querySelectorAll("[data-accept-example]").forEach((button) => button.addEventListener("click", () => {
    const example_id = button.dataset.acceptExample;
    const decisions = state.rows.filter((row) => row.example_id === example_id)
      .map((row) => ({ example_id, model_code: row.model_code, action: "accept" }));
    save(decisions);
  }));
}

async function save(decisions) {
  const message = byId("save-message");
  message.className = "";
  message.textContent = "Saving locally…";
  try {
    const response = await fetch("/api/decisions", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decisions })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Save failed");
    message.className = "success";
    message.textContent = "Saved locally";
    await loadReview();
  } catch (error) {
    message.className = "error";
    message.textContent = error.message;
  }
}

byId("pending-only").addEventListener("change", (event) => { state.pendingOnly = event.target.checked; renderExamples(); });
loadReview().catch((error) => { byId("review-list").textContent = error.message; });
