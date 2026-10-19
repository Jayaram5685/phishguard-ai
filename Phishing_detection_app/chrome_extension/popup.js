const DEFAULT_API_URL = "http://127.0.0.1:5000/predict";
const REQUEST_TIMEOUT_MS = 15000;

const ui = {
  resultContainer: document.getElementById("result-container"),
  result: document.getElementById("result"),
  button: document.getElementById("checkBtn"),
  urlBox: document.getElementById("urlDisplay"),
  icon: document.getElementById("main-icon"),
  reasons: document.getElementById("reasons"),
};

const MAX_REASONS_SHOWN = 4;

function setIcon(glyph) {
  ui.icon.textContent = glyph;
}

function setState(state) {
  ui.result.className = state;
}

function renderReasons(reasons) {
  ui.reasons.replaceChildren();
  if (!Array.isArray(reasons)) return;
  for (const text of reasons.slice(0, MAX_REASONS_SHOWN)) {
    const item = document.createElement("li");
    item.textContent = text;
    ui.reasons.appendChild(item);
  }
}

function scoreSuffix(score) {
  if (typeof score !== "number") return "";
  return ` (phishing score ${Math.round(score * 100)}%)`;
}

function paintLoading(url) {
  ui.resultContainer.classList.add("show");
  ui.result.textContent = "Scanning securely...";
  setState("loading-text");
  setIcon("...");
  ui.icon.classList.add("loading-icon");
  ui.icon.style.background = "rgba(255, 255, 255, 0.05)";
  ui.icon.style.borderColor = "rgba(255, 255, 255, 0.1)";
  ui.icon.style.boxShadow = "none";
  ui.urlBox.textContent = url;
  renderReasons(null);
  ui.button.disabled = true;
  ui.button.textContent = "Analyzing...";
}

function paintVerdict({ glyph, text, state, background, border, glow }) {
  ui.icon.classList.remove("loading-icon");
  setIcon(glyph);
  ui.icon.style.background = background;
  ui.icon.style.borderColor = border;
  ui.icon.style.boxShadow = glow;
  ui.result.textContent = text;
  setState(state);
}

const VERDICTS = {
  phishing: {
    glyph: "!",
    text: "Phishing Detected!",
    state: "phishing",
    background: "rgba(239, 68, 68, 0.1)",
    border: "rgba(239, 68, 68, 0.3)",
    glow: "0 0 20px rgba(239, 68, 68, 0.2)",
  },
  safe: {
    glyph: "OK",
    text: "Safe Website",
    state: "safe",
    background: "rgba(16, 185, 129, 0.1)",
    border: "rgba(16, 185, 129, 0.3)",
    glow: "0 0 20px rgba(16, 185, 129, 0.2)",
  },
  info: {
    glyph: "-",
    text: "Not analysed",
    state: "loading-text",
    background: "rgba(148, 163, 184, 0.1)",
    border: "rgba(148, 163, 184, 0.3)",
    glow: "none",
  },
  error: {
    glyph: "!",
    text: "Check failed",
    state: "phishing",
    background: "rgba(239, 68, 68, 0.1)",
    border: "rgba(239, 68, 68, 0.3)",
    glow: "0 0 20px rgba(239, 68, 68, 0.2)",
  },
};

function paintVerdictKind(verdict) {
  const kind = verdict.kind;
  const painted = { ...VERDICTS[kind] };
  if (verdict.detail) painted.text = verdict.detail;
  else painted.text = painted.text + scoreSuffix(verdict.score);
  painted.text = painted.text.trim();
  paintVerdict(painted);
  renderReasons(verdict.reasons);
}

async function readConfig() {
  const config = { apiUrl: DEFAULT_API_URL, apiKey: null };
  if (!chrome.storage || !chrome.storage.local) return config;
  const stored = await chrome.storage.local.get(["apiUrl", "apiKey"]);
  if (typeof stored.apiUrl === "string" && stored.apiUrl.trim()) {
    config.apiUrl = stored.apiUrl.trim();
  }
  if (typeof stored.apiKey === "string" && stored.apiKey.trim()) {
    config.apiKey = stored.apiKey.trim();
  }
  return config;
}

async function scanUrl(url, config) {
  const headers = { "Content-Type": "application/json" };
  if (config.apiKey) headers["X-API-Key"] = config.apiKey;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response;
  try {
    response = await fetch(config.apiUrl, {
      method: "POST",
      headers,
      body: JSON.stringify({ url }),
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeout);
  }

  let body = null;
  try {
    body = await response.json();
  } catch (parseError) {
    body = null;
  }

  if (!response.ok || (body && body.error)) {
    const code = (body && body.error && body.error.code) || `http_${response.status}`;
    const retryAfter = response.headers.get("Retry-After");
    const suffix = retryAfter ? ` (retry in ${retryAfter}s)` : "";
    return { kind: "error", detail: `Error: ${code}${suffix}` };
  }

  if (body && body.prediction === 1) {
    return { kind: "phishing", score: body.score, reasons: body.reasons };
  }
  if (body && body.analysis === "whitelisted") {
    return { kind: "info", detail: `Known-safe domain: ${body.domain}` };
  }
  if (body && body.analysis === "skipped") {
    return { kind: "info", detail: "Not applicable to this address" };
  }
  return { kind: "safe", score: body ? body.score : null, reasons: body ? body.reasons : null };
}

ui.button.addEventListener("click", async () => {
  let tab;
  try {
    [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  } catch (queryError) {
    paintVerdictKind({ kind: "error", detail: "Error: no_active_tab" });
    ui.button.textContent = "Scan Again";
    return;
  }

  const url = tab && tab.url;
  if (!url) {
    paintVerdictKind({ kind: "error", detail: "Error: url_unavailable" });
    ui.button.textContent = "Scan Again";
    return;
  }

  paintLoading(url);

  try {
    const config = await readConfig();
    const verdict = await scanUrl(url, config);
    paintVerdictKind(verdict);
  } catch (error) {
    const detail =
      error && error.name === "AbortError" ? "Error: api_timeout" : "Error: api_unreachable";
    paintVerdictKind({ kind: "error", detail });
    console.error("PhishGuard scan failed:", error);
  }

  ui.button.disabled = false;
  ui.button.textContent = "Scan Again";
});
