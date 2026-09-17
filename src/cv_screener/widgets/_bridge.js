/* Minimal MCP Apps bridge: JSON-RPC over postMessage to the host (spec 2026-01-26). */
const Host = (() => {
  const pending = new Map();
  let nextId = 1;
  const listeners = { toolResult: [], toolInput: [] };
  function send(method, params) {
    return new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      window.parent.postMessage({ jsonrpc: "2.0", id, method, params }, "*");
    });
  }
  function notify(method, params) {
    window.parent.postMessage({ jsonrpc: "2.0", method, params: params || {} }, "*");
  }
  window.addEventListener("message", (e) => {
    const m = e.data;
    if (!m || m.jsonrpc !== "2.0") return;
    if (m.id !== undefined && m.method === undefined) {
      const p = pending.get(m.id);
      if (!p) return;
      pending.delete(m.id);
      m.error ? p.reject(m.error) : p.resolve(m.result);
      return;
    }
    if (m.method === "ui/notifications/tool-result") listeners.toolResult.forEach((f) => f(m.params || {}));
    if (m.method === "ui/notifications/tool-input") listeners.toolInput.forEach((f) => f(m.params || {}));
  });
  async function init(name) {
    let res = null;
    try {
      res = await send("ui/initialize", {
        capabilities: {},
        clientInfo: { name, version: "0.1.0" },
        protocolVersion: "2026-01-26",
      });
    } catch (_) { /* not inside a host */ }
    const theme = res && res.hostContext && res.hostContext.theme;
    if (theme === "dark") document.documentElement.classList.add("theme-dark");
    notify("ui/notifications/initialized");
    new ResizeObserver(() => {
      notify("ui/notifications/size-changed", { width: document.body.scrollWidth, height: document.body.scrollHeight });
    }).observe(document.body);
    return res;
  }
  /* Tool results arrive as CallToolResult; lists are wrapped as {result: [...]} by the Python SDK. */
  function unwrap(result) {
    let data = result.structuredContent;
    if (data === undefined && result.content && result.content[0] && result.content[0].text) {
      try { data = JSON.parse(result.content[0].text); } catch (_) { data = null; }
    }
    if (data && typeof data === "object" && !Array.isArray(data) && Object.keys(data).length === 1 && "result" in data) data = data.result;
    return data;
  }
  return {
    init,
    onToolResult: (f) => listeners.toolResult.push(f),
    onToolInput: (f) => listeners.toolInput.push(f),
    callTool: (name, args) => send("tools/call", { name, arguments: args }).then(unwrap),
    say: (text) => send("ui/message", { role: "user", content: { type: "text", text } }),
    openLink: (url) => send("ui/open-link", { url }),
    unwrap,
  };
})();
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const initials = (name) => String(name || "?").split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase();
