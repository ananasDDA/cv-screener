/* Chat client and MCP Apps host. No build step, no dependencies.
   The page answers the widgets' JSON-RPC over postMessage and forwards tools/call to /api/tool. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const transcript = $("#transcript");
const hero = $("#hero");
const input = $("#input");
const sendBtn = $("#send");
const dropzone = $("#dropzone");

const mounted = []; // { frame, spec, html }
const history = []; // {role, content} pairs sent back with the next question
let busy = false;

const WIDGET_TITLES = {
  results: "Search results",
  card: "Candidate profile",
  deck: "Candidate collection",
};
const HISTORY_TURNS = 6;

/* ---------- text ---------------------------------------------------------- */

const esc = (s) =>
  String(s === null || s === undefined ? "" : s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );

/* Tiny markdown: HTML is escaped first, then bold/italic/code, lists and line breaks. */
function markdown(src) {
  const inline = (s) =>
    s
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  let html = "";
  let list = null;
  let para = [];
  const closeList = () => {
    if (list) {
      html += "</" + list + ">";
      list = null;
    }
  };
  const flush = () => {
    if (para.length) {
      html += "<p>" + inline(para.join("<br>")) + "</p>";
      para = [];
    }
  };
  for (const raw of esc(src).split(/\r?\n/)) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*(?:[-*]|&#39;|•)\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const heading = line.match(/^\s*#{1,4}\s+(.*)$/);
    if (bullet || numbered) {
      flush();
      const want = bullet ? "ul" : "ol";
      if (list !== want) {
        closeList();
        html += "<" + want + ">";
        list = want;
      }
      html += "<li>" + inline((bullet || numbered)[1]) + "</li>";
    } else if (heading) {
      flush();
      closeList();
      html += "<p><strong>" + inline(heading[1]) + "</strong></p>";
    } else if (!line.trim()) {
      flush();
      closeList();
    } else {
      closeList();
      para.push(line);
    }
  }
  flush();
  closeList();
  return html;
}

function fmtArgs(args) {
  const text = Object.entries(args || {})
    .map(([k, v]) => k + "=" + JSON.stringify(v))
    .join(", ");
  return text.length > 140 ? text.slice(0, 137) + "…" : text;
}

/* ---------- theme --------------------------------------------------------- */

const THEME_KEY = "cv-screener-theme";
const themeOf = () =>
  document.documentElement.classList.contains("theme-dark") ? "dark" : "light";

function setTheme(theme, persist) {
  document.documentElement.classList.toggle("theme-dark", theme === "dark");
  $("#theme-label").textContent = theme === "dark" ? "Light" : "Dark";
  $("#theme").setAttribute(
    "aria-label",
    theme === "dark" ? "Switch to light theme" : "Switch to dark theme",
  );
  if (persist) {
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch (_) {
      /* private mode */
    }
  }
  // Widgets read the theme once, at ui/initialize: reload them so they follow.
  mounted.forEach((entry) => {
    entry.frame.srcdoc = entry.html;
  });
}

/* The theme class is already on <html> (inline script in the head); only sync the button. */
setTheme(themeOf(), false);

$("#theme").addEventListener("click", () => setTheme(themeOf() === "dark" ? "light" : "dark", true));

/* ---------- model picker -------------------------------------------------- */

const MODEL_KEY = "cv-screener-model";
const select = $("#model");
const badge = $("#model-badge");
const note = $("#model-note");
let catalogue = [];

const price = (m) =>
  m.free
    ? "free"
    : "$" + m.prompt_price.toFixed(2) + " / $" + m.completion_price.toFixed(2) + " per 1M";
const shortName = (m) => String(m.name || m.id).replace(/\s*\(free\)\s*$/i, "");

function optionFor(model) {
  const option = document.createElement("option");
  option.value = model.id;
  option.textContent = model.free ? shortName(model) : shortName(model) + " · " + price(model);
  return option;
}

function refreshBadge() {
  const model = catalogue.find((m) => m.id === select.value);
  badge.hidden = !model;
  note.hidden = !model || model.free;
  if (!model) return;
  badge.textContent = model.free ? "free" : price(model);
  badge.className = model.free ? "chip tiny free" : "chip tiny";
}

async function loadModels() {
  let payload;
  try {
    payload = await (await fetch("/api/models")).json();
  } catch (_) {
    select.innerHTML = '<option value="">default model</option>';
    return;
  }
  catalogue = payload.models || [];
  let saved = null;
  try {
    saved = localStorage.getItem(MODEL_KEY);
  } catch (_) {
    /* private mode */
  }
  const free = catalogue.filter((m) => m.free);
  const paid = catalogue.filter((m) => !m.free);
  select.innerHTML = "";
  for (const [label, group] of [
    ["Free", free],
    ["Paid", paid],
  ]) {
    if (!group.length) continue;
    const optgroup = document.createElement("optgroup");
    optgroup.label = label;
    group.forEach((m) => optgroup.appendChild(optionFor(m)));
    select.appendChild(optgroup);
  }
  const wanted = catalogue.some((m) => m.id === saved) ? saved : payload.default;
  select.value = catalogue.some((m) => m.id === wanted) ? wanted : (catalogue[0] || {}).id || "";
  refreshBadge();
}

select.addEventListener("change", () => {
  try {
    localStorage.setItem(MODEL_KEY, select.value);
  } catch (_) {
    /* private mode */
  }
  refreshBadge();
});

loadModels();

/* ---------- MCP Apps host ------------------------------------------------- */

const widgetHtml = {};

function loadWidgetHtml(name) {
  if (!widgetHtml[name]) {
    widgetHtml[name] = fetch("/widgets/" + encodeURIComponent(name) + ".html").then((r) => {
      if (!r.ok) throw new Error("widget " + name + " is unavailable");
      return r.text();
    });
  }
  return widgetHtml[name];
}

async function mountWidget(slot, spec) {
  slot.hidden = false;
  let html;
  try {
    html = await loadWidgetHtml(spec.name);
  } catch (err) {
    slot.innerHTML = '<p class="meta">' + esc(String(err.message || err)) + "</p>";
    return;
  }
  const frame = document.createElement("iframe");
  frame.className = "widget";
  frame.title = WIDGET_TITLES[spec.name] || "Candidate widget";
  frame.setAttribute("sandbox", "allow-scripts");
  frame.setAttribute("scrolling", "no");
  slot.appendChild(frame);
  mounted.push({ frame, spec, html });
  frame.srcdoc = html;
}

function toolResult(data) {
  return { content: [{ type: "text", text: JSON.stringify(data) }], structuredContent: data };
}

async function callTool(name, args) {
  const r = await fetch("/api/tool", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name: name, arguments: args }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error((data && data.error) || "tool " + name + " failed");
  return data;
}

window.addEventListener("message", async (event) => {
  const entry = mounted.find((m) => m.frame.contentWindow === event.source);
  if (!entry) return;
  const msg = event.data;
  if (!msg || msg.jsonrpc !== "2.0" || !msg.method) return;
  const win = entry.frame.contentWindow;
  const post = (payload) => win && win.postMessage(payload, "*");
  const reply = (result) => post({ jsonrpc: "2.0", id: msg.id, result: result });
  const fail = (message) =>
    post({ jsonrpc: "2.0", id: msg.id, error: { code: -32000, message: message } });
  const params = msg.params || {};

  switch (msg.method) {
    case "ui/initialize":
      reply({
        protocolVersion: "2026-01-26",
        capabilities: {},
        serverInfo: { name: "cv-screener-web", version: "0.1.0" },
        hostContext: { theme: themeOf() },
      });
      break;
    case "ui/notifications/initialized":
      post({
        jsonrpc: "2.0",
        method: "ui/notifications/tool-input",
        params: { toolName: entry.spec.tool, arguments: entry.spec.args || {} },
      });
      post({
        jsonrpc: "2.0",
        method: "ui/notifications/tool-result",
        params: toolResult(entry.spec.data),
      });
      break;
    case "ui/notifications/size-changed":
      if (params.height) {
        entry.frame.style.height = Math.max(80, Math.ceil(params.height) + 8) + "px";
      }
      break;
    case "tools/call":
      try {
        reply(toolResult(await callTool(params.name, params.arguments || {})));
      } catch (err) {
        fail(String((err && err.message) || err));
      }
      break;
    case "ui/message":
      reply({});
      if (params.content && params.content.text) submit(params.content.text);
      break;
    case "ui/open-link":
      reply({});
      if (params.url) window.open(params.url, "_blank", "noopener");
      break;
    default:
      if (msg.id !== undefined) fail("unsupported method " + msg.method);
  }
});

/* ---------- transcript ---------------------------------------------------- */

function hideHero() {
  if (hero && !hero.hidden) hero.hidden = true;
}

function scrollDown() {
  transcript.scrollTop = transcript.scrollHeight;
}

function addUser(text, label) {
  hideHero();
  const el = document.createElement("div");
  el.className = "user enter";
  el.innerHTML =
    '<div class="bubble">' +
    (label ? '<span class="file">' + esc(label) + "</span>" : esc(text)) +
    "</div>";
  transcript.appendChild(el);
  scrollDown();
}

function addTurn() {
  hideHero();
  const el = document.createElement("section");
  el.className = "turn enter";
  el.innerHTML =
    '<details class="trace" open hidden><summary><span class="count"></span></summary>' +
    '<div class="trace-lines"></div></details>' +
    '<div class="widget-slot" hidden></div>' +
    '<div class="answer"><span class="thinking"><i></i><i></i><i></i>' +
    "<span>searching the collection…</span></span></div>" +
    '<div class="meta" hidden></div>';
  transcript.appendChild(el);
  scrollDown();
  return el;
}

function addTrace(turnEl, tool) {
  const details = turnEl.querySelector(".trace");
  details.hidden = false;
  const lines = details.querySelector(".trace-lines");
  const line = document.createElement("div");
  line.className = "trace-line";
  line.innerHTML =
    "<b>" +
    esc(tool.name) +
    "</b>(" +
    esc(fmtArgs(tool.args)) +
    ') <span class="n">→ ' +
    esc(tool.count) +
    "</span>";
  lines.appendChild(line);
  const n = lines.children.length;
  details.querySelector(".count").textContent = n === 1 ? "1 tool call" : n + " tool calls";
  const label = turnEl.querySelector(".thinking span");
  if (label) label.textContent = "reading the results…";
  scrollDown();
}

function showAnswer(turnEl, payload) {
  turnEl.querySelector(".answer").innerHTML =
    markdown(payload.text) || '<p class="meta">The model returned an empty answer.</p>';
  const meta = turnEl.querySelector(".meta");
  meta.hidden = false;
  let html = payload.model ? '<span class="model">' + esc(payload.model) + "</span>" : "";
  if (payload.grounded === false) {
    html +=
      '<span class="chip warn">ungrounded: ' +
      esc((payload.ungrounded_names || []).join(", ")) +
      "</span>";
  }
  meta.innerHTML = html;
  scrollDown();
}

function showError(turnEl, message) {
  turnEl.querySelector(".answer").innerHTML =
    '<p><span class="chip err">' + esc(message) + "</span></p>";
  scrollDown();
}

/* ---------- one turn over SSE --------------------------------------------- */

async function streamTurn(message, turnEl) {
  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      message: message,
      model: select.value || undefined,
      history: history.slice(-2 * HISTORY_TURNS),
    }),
  });
  if (!resp.ok || !resp.body) {
    let detail = "the server refused the request";
    try {
      detail = (await resp.json()).error || detail;
    } catch (_) {
      /* not JSON */
    }
    showError(turnEl, detail);
    return null;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let name = "message";
  let data = [];
  let answer = null;

  const dispatch = () => {
    if (!data.length) return;
    let payload = {};
    try {
      payload = JSON.parse(data.join("\n"));
    } catch (_) {
      payload = {};
    }
    data = [];
    if (name === "tool") addTrace(turnEl, payload);
    else if (name === "widget") mountWidget(turnEl.querySelector(".widget-slot"), payload);
    else if (name === "answer") {
      answer = payload;
      showAnswer(turnEl, payload);
    } else if (name === "error") showError(turnEl, payload.message || "the agent failed");
    name = "message";
  };

  for (;;) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).replace(/\r$/, "");
      buffer = buffer.slice(nl + 1);
      if (line === "") {
        dispatch();
        continue;
      }
      if (line.startsWith(":")) continue;
      const colon = line.indexOf(":");
      const field = colon === -1 ? line : line.slice(0, colon);
      let value = colon === -1 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") name = value;
      else if (field === "data") data.push(value);
    }
  }
  dispatch();
  return answer;
}

/* ---------- sending ------------------------------------------------------- */

async function submit(text, opts) {
  opts = opts || {};
  const shown = String(text || "").trim();
  if (busy || !shown) return;
  busy = true;
  sendBtn.disabled = true;
  addUser(shown, opts.label);
  const turnEl = addTurn();
  const sent = opts.send || shown;
  try {
    const answer = await streamTurn(sent, turnEl);
    history.push({ role: "user", content: opts.historyText || shown });
    history.push({ role: "assistant", content: (answer && answer.text) || "" });
  } catch (err) {
    showError(turnEl, String((err && err.message) || err));
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value;
  input.value = "";
  input.style.height = "auto";
  submit(text);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("#composer").requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
});

document.querySelectorAll(".sample").forEach((chip) => {
  chip.addEventListener("click", () => submit(chip.textContent.trim()));
});

/* ---------- collection and uploads ---------------------------------------- */

$("#browse").addEventListener("click", () => {
  hideHero();
  const el = document.createElement("section");
  el.className = "turn enter";
  el.innerHTML =
    '<details class="trace" open><summary><span class="count">the collection</span></summary>' +
    '<div class="trace-lines"><div class="trace-line"><b>get_deck</b>() ' +
    '<span class="n">→ deck widget</span></div></div></details>' +
    '<div class="widget-slot"></div>';
  transcript.appendChild(el);
  mountWidget(el.querySelector(".widget-slot"), {
    name: "deck",
    tool: "list_candidates",
    args: {},
    data: {},
  });
  scrollDown();
});

$("#add").addEventListener("click", () => $("#file").click());
$("#file").addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  e.target.value = "";
  if (file) uploadResume(file);
});

async function uploadResume(file) {
  if (busy) return;
  const body = new FormData();
  body.append("file", file);
  let data;
  try {
    const r = await fetch("/api/upload", { method: "POST", body: body });
    data = await r.json();
    if (!r.ok) throw new Error((data && data.error) || "upload failed");
  } catch (err) {
    hideHero();
    showError(addTurn(), String((err && err.message) || err));
    return;
  }
  submit("Add this résumé to the collection", {
    send: data.prompt,
    label: "Uploaded " + data.filename + " · " + data.chars + " characters",
    historyText: "I uploaded the résumé " + data.filename + " and asked to add it.",
  });
}

let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  if (!e.dataTransfer || !Array.from(e.dataTransfer.types || []).includes("Files")) return;
  dragDepth += 1;
  dropzone.hidden = false;
  dropzone.setAttribute("aria-hidden", "false");
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) dropzone.hidden = true;
});
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  dropzone.hidden = true;
  const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) uploadResume(file);
});

input.focus();
