const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const titles = {
  status: ["Status", "Worker health and NPU readiness."],
  config: ["Gateway", "Host, port, retries, and project sandbox."],
  plugins: ["Plugins", "Native workers and MCP servers."],
  themes: ["Themes", "Style the control panel and GTK launcher."],
};

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    el.hidden = true;
  }, 3200);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) {
    const detail = data?.detail || data || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function setTab(name) {
  $$(".nav-item").forEach((btn) => btn.classList.toggle("is-active", btn.dataset.tab === name));
  $$(".panel").forEach((panel) => panel.classList.toggle("is-active", panel.id === `tab-${name}`));
  const [title, lede] = titles[name];
  $("#page-title").textContent = title;
  $("#page-lede").textContent = lede;
  history.replaceState(null, "", `#${name}`);
}

$$(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => setTab(btn.dataset.tab));
});

function pill(label, ok, detail = "", state = null) {
  const kind = state || (ok ? "ok" : "bad");
  const labelText =
    kind === "ok" ? "Online" : kind === "warn" ? "Driver only" : "Offline";
  return `<article class="pill">
    <div class="label">${label}</div>
    <div class="value"><span class="dot ${kind}"></span>${labelText}</div>
    ${detail ? `<div class="muted" style="font-size:.82rem">${detail}</div>` : ""}
  </article>`;
}

function applyThemeCss() {
  const link = $("#theme-css") || document.querySelector('link[href*="theme.css"]');
  if (!link) return;
  link.href = `/theme.css?t=${Date.now()}`;
}

async function loadStatus() {
  const data = await api("/api/status");
  const grid = $("#status-grid");
  const npuState = data.npu ? "ok" : data.npu_driver ? "warn" : "bad";
  grid.innerHTML = [
    pill("NPU", data.npu, data.npu_detail || "", npuState),
    pill("LM Studio", data.lmstudio),
    pill("ComfyUI", data.comfyui),
    pill("Claude", data.claude),
    pill("Cursor", data.cursor),
    pill("Lemonade", data.lemonade),
    pill("Gateway", data.gateway, data.local_only ? "Local-only mode" : "All workers allowed"),
  ].join("");
}

async function loadConfig() {
  const data = await api("/api/config");
  const form = $("#config-form");
  form.host.value = data.host;
  form.port.value = data.port;
  form.max_retries.value = data.max_retries;
  form.local_only.checked = !!data.local_only;
  form.require_confirmation_for_modifications.checked = !!data.require_confirmation_for_modifications;
  form.allowed_project_roots.value = (data.allowed_project_roots || []).join("\n");
  $("#config-note").textContent = data.note || "";
}

$("#config-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  const body = {
    host: form.host.value.trim(),
    port: Number(form.port.value),
    max_retries: Number(form.max_retries.value),
    local_only: form.local_only.checked,
    require_confirmation_for_modifications: form.require_confirmation_for_modifications.checked,
    allowed_project_roots: form.allowed_project_roots.value
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean),
  };
  try {
    const saved = await api("/api/config", { method: "PUT", body: JSON.stringify(body) });
    $("#config-note").textContent = saved.note || "Saved.";
    toast(saved.restart_required ? "Saved. Restart service for port/host." : "Settings saved.");
  } catch (err) {
    toast(err.message);
  }
});

function settingInputs(plugin) {
  const schema = plugin.settings_schema || [];
  if (!schema.length) return "";
  return `<div class="settings-grid" data-plugin-settings="${plugin.id}">
    ${schema
      .map((field) => {
        const val = plugin.settings?.[field.key] ?? field.default ?? "";
        if (field.type === "boolean") {
          return `<label class="check"><input type="checkbox" name="${field.key}" ${val ? "checked" : ""}/><span>${field.label}</span></label>`;
        }
        const type = field.type === "integer" ? "number" : "text";
        return `<label><span>${field.label}</span><input name="${field.key}" type="${type}" value="${String(val).replaceAll('"', "&quot;")}" /></label>`;
      })
      .join("")}
    <button type="button" class="btn ghost save-settings" data-id="${plugin.id}">Save plugin settings</button>
  </div>`;
}

async function loadPlugins() {
  const plugins = await api("/api/plugins?health=true");
  const list = $("#plugin-list");
  list.innerHTML = plugins
    .map((p) => {
      const health = p.health;
      const healthText = health ? `${health.ok ? "Healthy" : "Down"} - ${health.detail || ""}` : "";
      return `<article class="card">
        <div class="card-head">
          <div>
            <h3>${p.name}</h3>
            <p class="muted" style="margin:.25rem 0 0">${p.description || ""}</p>
            <p class="muted" style="margin:.35rem 0 0;font-size:.85rem">${healthText}</p>
          </div>
          <div class="row">
            <span class="badge ${p.kind === "mcp" ? "mcp" : ""}">${p.kind}</span>
            ${p.worker_id ? `<span class="badge">${p.worker_id}</span>` : ""}
          </div>
        </div>
        <div class="card-actions">
          <button type="button" class="btn ${p.enabled ? "ghost" : "primary"} toggle-plugin" data-id="${p.id}" data-enabled="${p.enabled}">
            ${p.enabled ? "Disable" : "Enable"}
          </button>
          ${p.removable ? `<button type="button" class="btn danger remove-plugin" data-id="${p.id}">Remove</button>` : ""}
        </div>
        ${settingInputs(p)}
      </article>`;
    })
    .join("");
}

$("#plugin-list").addEventListener("click", async (e) => {
  const toggle = e.target.closest(".toggle-plugin");
  const remove = e.target.closest(".remove-plugin");
  const save = e.target.closest(".save-settings");
  try {
    if (toggle) {
      const enabled = toggle.dataset.enabled !== "true";
      await api(`/api/plugins/${toggle.dataset.id}/enable`, {
        method: "POST",
        body: JSON.stringify({ enabled }),
      });
      toast(`${toggle.dataset.id} ${enabled ? "enabled" : "disabled"}`);
      await loadPlugins();
      await loadStatus();
    }
    if (remove) {
      await api(`/api/plugins/${remove.dataset.id}`, { method: "DELETE" });
      toast("Plugin removed");
      await loadPlugins();
    }
    if (save) {
      const wrap = save.closest("[data-plugin-settings]");
      const body = {};
      wrap.querySelectorAll("input[name]").forEach((input) => {
        body[input.name] = input.type === "checkbox" ? input.checked : input.value;
      });
      await api(`/api/plugins/${save.dataset.id}/settings`, {
        method: "PUT",
        body: JSON.stringify(body),
      });
      toast("Plugin settings saved");
    }
  } catch (err) {
    toast(err.message);
  }
});

$("#reload-plugins").addEventListener("click", async () => {
  try {
    await api("/api/plugins/reload", { method: "POST", body: "{}" });
    await loadPlugins();
    toast("Plugins reloaded");
  } catch (err) {
    toast(err.message);
  }
});

$("#mcp-install-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  let args = [];
  try {
    args = JSON.parse(form.args.value || "[]");
  } catch {
    toast("Args must be a JSON array");
    return;
  }
  const body = {
    id: form.id.value.trim(),
    name: form.name.value.trim() || form.id.value.trim(),
    command: form.command.value.trim(),
    args,
    health_tool: form.health_tool.value.trim() || "server_info",
    run_tool: form.run_tool.value.trim() || "generate_image",
    worker_id: form.worker_id.value.trim() || null,
    cloud: form.cloud.checked,
  };
  try {
    await api("/api/plugins/install/mcp", { method: "POST", body: JSON.stringify(body) });
    form.reset();
    form.args.value = "[]";
    form.health_tool.value = "server_info";
    form.run_tool.value = "generate_image";
    toast("MCP plugin installed");
    await loadPlugins();
  } catch (err) {
    toast(err.message);
  }
});

async function loadThemes() {
  const themes = await api("/api/themes");
  $("#theme-list").innerHTML = themes
    .map((t) => {
      const colors = t.preview_colors || {};
      const swatches = ["bg", "surface", "accent", "text"]
        .filter((k) => colors[k])
        .map((k) => `<span class="swatch" style="background:${colors[k]}" title="${k}"></span>`)
        .join("");
      return `<article class="theme-card ${t.active ? "is-active" : ""}">
        <div class="swatches">${swatches}</div>
        <div>
          <strong>${t.name}</strong>
          <div class="muted" style="font-size:.85rem">${t.author} · v${t.version}</div>
          <p class="muted" style="margin:.4rem 0 0;font-size:.9rem">${t.description || ""}</p>
        </div>
        <div class="row">
          <button type="button" class="btn ${t.active ? "ghost" : "primary"} activate-theme" data-id="${t.id}" ${t.active ? "disabled" : ""}>
            ${t.active ? "Active" : "Use theme"}
          </button>
          ${["default", "midnight", "paper"].includes(t.id) ? "" : `<button type="button" class="btn danger remove-theme" data-id="${t.id}">Remove</button>`}
        </div>
      </article>`;
    })
    .join("");
}

$("#theme-list").addEventListener("click", async (e) => {
  const activate = e.target.closest(".activate-theme");
  const remove = e.target.closest(".remove-theme");
  try {
    if (activate) {
      await api("/api/themes/active", { method: "PUT", body: JSON.stringify({ id: activate.dataset.id }) });
      applyThemeCss();
      toast("Theme activated — web + Super+Space launcher");
      await loadThemes();
    }
    if (remove) {
      await api(`/api/themes/${remove.dataset.id}`, { method: "DELETE" });
      toast("Theme removed");
      await loadThemes();
    }
  } catch (err) {
    toast(err.message);
  }
});

$("#theme-install-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = e.currentTarget.file.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch("/api/themes/install", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Install failed");
    toast(`Installed ${data.name}`);
    e.currentTarget.reset();
    await loadThemes();
  } catch (err) {
    toast(err.message);
  }
});

async function refreshAll() {
  await Promise.all([loadStatus(), loadConfig(), loadPlugins(), loadThemes()]);
}

$("#refresh-btn").addEventListener("click", () => {
  refreshAll().then(() => toast("Refreshed")).catch((err) => toast(err.message));
});

const initial = (location.hash || "#status").slice(1);
setTab(titles[initial] ? initial : "status");
refreshAll().catch((err) => toast(err.message));
