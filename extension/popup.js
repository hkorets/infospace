// ── State ─────────────────────────────────────────────────────────────────────
const DEFAULT_API = "http://localhost:8000";

let apiBase = DEFAULT_API;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const inputText   = document.getElementById("inputText");
const verifyBtn   = document.getElementById("verifyBtn");
const charCount   = document.getElementById("charCount");
const loading     = document.getElementById("loading");
const loadingStep = document.getElementById("loadingStep");
const results     = document.getElementById("results");
const errorBox    = document.getElementById("errorBox");
const statusDot   = document.getElementById("statusDot");
const statusText  = document.getElementById("statusText");
const apiUrlInput = document.getElementById("apiUrl");
const saveBtn     = document.getElementById("saveBtn");

// ── Init ──────────────────────────────────────────────────────────────────────
function init() {
  const saved = localStorage.getItem("infospace_api");
  if (saved) {
    apiBase = saved;
    apiUrlInput.value = saved;
  } else {
    apiUrlInput.value = DEFAULT_API;
  }
  checkApiStatus();
}

// ── API health check ──────────────────────────────────────────────────────────
async function checkApiStatus() {
  statusDot.className = "status-dot";
  statusText.textContent = "checking...";
  try {
    const res = await fetch(`${apiBase}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      statusDot.classList.add("online");
      statusText.textContent = "online";
    } else {
      throw new Error();
    }
  } catch {
    statusDot.classList.add("offline");
    statusText.textContent = "offline";
  }
}

// ── Char counter ──────────────────────────────────────────────────────────────
inputText.addEventListener("input", () => {
  const n = inputText.value.length;
  charCount.textContent = `${n} chars`;
});

// ── Save API URL ──────────────────────────────────────────────────────────────
saveBtn.addEventListener("click", () => {
  apiBase = apiUrlInput.value.trim().replace(/\/$/, "");
  localStorage.setItem("infospace_api", apiBase);
  saveBtn.textContent = "Saved ✓";
  setTimeout(() => { saveBtn.textContent = "Save"; }, 1500);
  checkApiStatus();
});

// ── Verify ────────────────────────────────────────────────────────────────────
verifyBtn.addEventListener("click", verify);
inputText.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) verify();
});

async function verify() {
  const text = inputText.value.trim();
  if (!text) return;

  // Reset UI
  setLoading(true);
  hideResults();
  hideError();
  verifyBtn.disabled = true;

  try {
    // Step 1 — extracting
    setLoadingStep("Extracting claims...");
    await sleep(300); // small delay so user sees the step

    const res = await fetch(`${apiBase}/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: AbortSignal.timeout(60000),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    setLoadingStep("Verifying against knowledge base...");
    const data = await res.json();

    setLoadingStep("Generating summary...");
    await sleep(200);

    renderResults(data);

  } catch (err) {
    showError(err.message.includes("Failed to fetch")
      ? `Cannot reach API at ${apiBase}. Is the server running?`
      : err.message
    );
  } finally {
    setLoading(false);
    verifyBtn.disabled = false;
  }
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  // Summary
  document.getElementById("summaryText").textContent = data.summary || "—";

  // Stats
  const lc = data.label_counts || {};
  const statsRow = document.getElementById("statsRow");
  statsRow.innerHTML = `
    <div class="stat-pill supported">✓ ${lc.SUPPORTED || 0} supported</div>
    <div class="stat-pill notsup">✗ ${lc.NOT_SUPPORTED || 0} not supported</div>
    <div class="stat-pill insuf">? ${lc.INSUFFICIENT || 0} insufficient</div>
  `;

  // Claims
  const list = document.getElementById("claimsList");
  list.innerHTML = "";

  (data.results || []).forEach((r, i) => {
    const item = document.createElement("div");
    item.className = "claim-item";

    const label = r.label || "INSUFFICIENT";

    item.innerHTML = `
      <div class="claim-header">
        <span class="badge ${label}">${labelShort(label)}</span>
        <span class="claim-text">${escHtml(r.claim)}</span>
        <span class="chevron">▾</span>
      </div>
      <div class="claim-detail">
        ${r.reasoning ? `
          <div class="detail-section">
            <div class="detail-label">Reasoning</div>
            <div class="detail-text">${escHtml(r.reasoning)}</div>
          </div>` : ""}
        ${r.evidence ? `
          <div class="detail-section">
            <div class="detail-label">Evidence</div>
            <div class="detail-text">${escHtml(r.evidence)}</div>
            ${r.source ? `<div class="source-tag">📄 ${escHtml(r.source)}${r.tier ? ` · tier ${r.tier}` : ""}</div>` : ""}
          </div>` : ""}
      </div>
    `;

    // Toggle expand
    item.querySelector(".claim-header").addEventListener("click", () => {
      item.classList.toggle("open");
    });

    list.appendChild(item);
  });

  results.classList.add("visible");
  results.style.display = "block";
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function labelShort(label) {
  return { SUPPORTED: "✓ OK", NOT_SUPPORTED: "✗ NO", INSUFFICIENT: "? ?" }[label] || label;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function setLoading(on) {
  loading.classList.toggle("visible", on);
}

function setLoadingStep(text) {
  loadingStep.textContent = text;
}

function hideResults() {
  results.classList.remove("visible");
  results.style.display = "none";
}

function hideError() {
  errorBox.classList.remove("visible");
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.add("visible");
}

// ── Start ─────────────────────────────────────────────────────────────────────
init();
