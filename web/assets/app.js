/* Keylane control panel. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const titles = {
  status: ["Status", "Worker health, NPU readiness and what the assistant is doing."],
  assistant: ["Assistant", "What the on-device model may do, and the tools it can reach."],
  plugins: ["Plugins", "Workers, MCP servers and tool providers."],
  skills: ["Skills", "Instruction packs that extend the assistant."],
  models: ["Models", "Device selection, downloads and per-worker defaults."],
  themes: ["Themes", "Style this panel and reshape the Super+Space popup."],
  config: ["Gateway", "Host, port, retries and the project sandbox."],
};

const esc = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    el.hidden = true;
  }, 3600);
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
  ensureTab(name).catch((err) => toast(err.message));
}

$$(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => setTab(btn.dataset.tab));
});

function applyThemeCss() {
  const link = $("#theme-css");
  if (link) link.href = `/theme.css?t=${Date.now()}`;
}

/* Lazy tab data: cold boot used to fan out every control-plane probe at once
   (worker health, Hugging Face catalogs, GitHub skill checks). Only the active
   section loads now; others hydrate on first visit. */
const loadedTabs = new Set();

async function ensureTab(name) {
  if (loadedTabs.has(name)) return;
  loadedTabs.add(name);
  switch (name) {
    case "status":
      await loadStatus();
      break;
    case "assistant":
      await Promise.all([loadAssistant(), loadTools()]);
      break;
    case "plugins":
      await Promise.all([loadPlugins({ health: false }), loadCatalog()]);
      loadPlugins({ health: true }).catch(() => {});
      break;
    case "skills":
      await loadSkills();
      loadSkillCatalog().catch(() => {});
      break;
    case "models":
      await loadModels();
      refreshHfJobs().catch(() => {});
      break;
    case "themes":
      await loadThemes();
      break;
    case "config":
      await Promise.all([
        loadConfig(),
        loadProjects(),
        loadWorkerEndpoints(),
        loadSystemInfo(),
      ]);
      break;
    default:
      break;
  }
}

/* ————————————————————————————————————————————————— status ——— */

function pill(label, ok, detail = "", state = null) {
  const kind = state || (ok ? "ok" : "bad");
  const text = kind === "ok" ? "Online" : kind === "warn" ? "Driver only" : "Offline";
  return `<article class="pill">
    <div class="label">${esc(label)}</div>
    <div class="value"><span class="dot ${kind}"></span>${text}</div>
    ${detail ? `<div class="detail">${esc(detail)}</div>` : ""}
  </article>`;
}

async function loadStatus() {
  const data = await api("/api/status");
  const grid = $("#status-grid");
  grid.removeAttribute("aria-busy");
  const npuState = data.npu ? "ok" : data.npu_driver ? "warn" : "bad";
  grid.innerHTML = [
    pill("NPU", data.npu, data.npu_detail || "", npuState),
    pill(
      "Assistant",
      data.assistant,
      data.assistant
        ? `${data.tool_count} tools ready`
        : `No NPU model loaded — ${data.tool_count} tools ready, keyword fallback in use`,
      data.assistant ? "ok" : "warn"
    ),
    pill("LM Studio", data.lmstudio),
    pill("ComfyUI", data.comfyui),
    pill("Claude Code", data.claude),
    pill("Cursor", data.cursor),
    pill("Lemonade", data.lemonade),
    pill("Gateway", data.gateway, data.local_only ? "Local-only mode" : "All workers allowed"),
  ].join("");

  $("#count-tools").textContent = data.tool_count || "";

  const banner = $("#status-banner");
  const partial = data.incomplete_models || [];
  if (!data.assistant && partial.length) {
    // A download that stopped early leaves the graph and not the weights.
    // Telling someone to download a model they already downloaded is the
    // least useful thing the panel could say, so name the file and resume it.
    const m = partial[0];
    banner.innerHTML = `<div class="banner warn">
      <svg viewBox="0 0 24 24"><use href="#i-alert" /></svg>
      <div><strong>${esc(m.id)}</strong> downloaded only part way — ${esc(
        m.missing.join(", ")
      )} ${m.missing.length === 1 ? "is" : "are"} missing, so it cannot load and
      Keylane is falling back to keyword matching.
      <button class="link rec-download" data-repo="${esc(m.repo_id)}"
        data-target="router">Resume the download</button></div>
    </div>`;
  } else if (!data.assistant && data.npu) {
    banner.innerHTML = `<div class="banner warn">
      <svg viewBox="0 0 24 24"><use href="#i-alert" /></svg>
      <div>The NPU is available but no router model is loaded, so Keylane is using
      keyword matching instead of the model.${
        data.assistant_note ? ` ${esc(data.assistant_note)}` : ""
      } Download an OpenVINO export under
      <strong>Models</strong> to switch the assistant on.</div>
    </div>`;
  } else if (!data.tools_enabled) {
    banner.innerHTML = `<div class="banner warn">
      <svg viewBox="0 0 24 24"><use href="#i-alert" /></svg>
      <div>The assistant tool layer is switched off — every request goes straight to a worker.</div>
    </div>`;
  } else {
    banner.innerHTML = "";
  }
  return data;
}

/* ————————————————————————————————————————————— activity ——— */

// Which task rows the user has opened. Snapshots arrive continuously and
// replace the list wholesale, so expansion has to live outside the markup or
// every row would slam shut a few times a second.
const expandedTasks = new Set();
// The most recent snapshot, kept so a purely visual change (opening a row)
// can re-render without asking the gateway again.
let lastSnapshot = null;

function argumentList(args) {
  const entries = Object.entries(args || {}).filter(
    ([, value]) => value !== "" && value !== null && value !== undefined
  );
  if (!entries.length) return `<p class="muted small">No arguments.</p>`;
  return `<dl class="arg-list">${entries
    .map(([key, value]) => {
      const shown = typeof value === "string" ? value : JSON.stringify(value);
      return `<div><dt>${esc(key)}</dt><dd><code>${esc(shown)}</code></dd></div>`;
    })
    .join("")}</dl>`;
}

function elapsed(task) {
  const start = Date.parse(task.started_at || "");
  const end = Date.parse(task.finished_at || task.updated_at || "") || Date.now();
  if (!start || end < start) return "";
  const seconds = Math.round((end - start) / 1000);
  return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function stepRow(step) {
  const mark = step.ok
    ? `<span class="dot ok"></span>`
    : `<span class="dot bad"></span>`;
  const args = Object.keys(step.arguments || {}).length
    ? argumentList(step.arguments)
    : "";
  return `<li class="step">
    <div class="step-head">
      ${mark}
      <span class="step-index">${esc(step.index)}</span>
      ${step.tool ? `<code class="step-tool">${esc(step.tool)}</code>` : ""}
      ${step.thought ? `<span class="step-thought">${esc(step.thought)}</span>` : ""}
    </div>
    ${args}
    ${
      step.observation
        ? `<pre class="step-observation">${esc(step.observation)}</pre>`
        : ""
    }
  </li>`;
}

// A tool name on its own is not enough to consent to. Show the arguments, and
// make approving and refusing equally reachable.
function approvalBlock(task) {
  return `<div class="approval">
    <div class="approval-head">
      <svg viewBox="0 0 24 24"><use href="#i-alert" /></svg>
      <div>
        <strong>Waiting for you.</strong>
        Keylane wants to run
        <code>${esc(task.pending_tool || "an action")}</code>
        ${task.worker ? `via <strong>${esc(task.worker)}</strong>` : ""}.
      </div>
    </div>
    ${argumentList(task.pending_arguments)}
    <div class="approval-actions">
      <button class="btn primary task-approve" data-task="${esc(task.task_id)}">Allow</button>
      <button class="btn danger task-deny" data-task="${esc(task.task_id)}">Deny</button>
    </div>
  </div>`;
}

function activityRow(task, { live }) {
  const waiting = task.status === "waiting_confirmation";
  const icon = live
    ? `<span class="spinner"></span>`
    : `<span class="dot ${
        task.status === "completed" ? "ok" : task.status === "failed" ? "bad" : "warn"
      }"></span>`;
  const when = task.finished_at || task.updated_at || task.started_at || "";
  const time = when ? new Date(when).toLocaleTimeString() : "";
  const steps = task.steps || [];
  const open = expandedTasks.has(task.task_id);
  const took = elapsed(task);

  // Only offer the disclosure when there is genuinely something behind it.
  const hasDetail = steps.length > 0 || waiting || !!task.error;

  return `<div class="activity-row ${open ? "is-open" : ""} ${waiting ? "needs-you" : ""}"
    data-task="${esc(task.task_id)}">
    <div class="activity-head ${hasDetail ? "expandable" : ""}"
      ${hasDetail ? `data-expand="${esc(task.task_id)}"` : ""}>
      ${icon}
      <span class="title">${esc(task.title || "(untitled)")}</span>
      ${task.worker ? `<span class="badge">${esc(task.worker)}</span>` : ""}
      <span class="badge ${waiting ? "warn" : ""}">${esc(task.step || task.status)}</span>
      ${steps.length ? `<span class="badge muted">${steps.length} step${steps.length === 1 ? "" : "s"}</span>` : ""}
      ${took ? `<span class="muted small">${esc(took)}</span>` : ""}
      <time>${esc(time)}</time>
      ${live && !waiting ? `<button class="btn small task-cancel" data-task="${esc(task.task_id)}">Stop</button>` : ""}
      ${hasDetail ? `<span class="chevron">${open ? "▾" : "▸"}</span>` : ""}
    </div>
    ${
      open
        ? `<div class="activity-detail">
            ${waiting ? approvalBlock(task) : ""}
            ${task.error ? `<p class="error-line">${esc(task.error)}</p>` : ""}
            ${
              steps.length
                ? `<ol class="step-list">${steps.map(stepRow).join("")}</ol>`
                : waiting
                  ? ""
                  : `<p class="muted small">No steps recorded yet.</p>`
            }
          </div>`
        : ""
    }
  </div>`;
}

function renderActivity(snapshot) {
  lastSnapshot = snapshot;
  const active = snapshot.active || [];
  const recent = snapshot.recent || [];

  // A task that is blocked on a person should never be hiding behind a
  // disclosure triangle — open it the first time it appears.
  for (const task of active) {
    if (task.status === "waiting_confirmation") expandedTasks.add(task.task_id);
  }

  $("#activity-active").innerHTML = active.length
    ? active.map((t) => activityRow(t, { live: t.status !== "waiting_confirmation" })).join("")
    : `<p class="empty">Nothing running. Press <kbd>Super</kbd>+<kbd>Space</kbd> to ask for something.</p>`;

  $("#activity-recent").innerHTML = recent.length
    ? recent.map((t) => activityRow(t, { live: false })).join("")
    : `<p class="empty">No tasks yet this session.</p>`;

  $("#activity-meta").textContent = snapshot.busy
    ? `${snapshot.active_count} running`
    : "Idle";

  const rail = $("#rail-activity");
  const kind = snapshot.needs_attention ? "warn" : snapshot.busy ? "ok" : "";
  const label = snapshot.needs_attention
    ? `${snapshot.needs_attention} awaiting approval`
    : snapshot.busy
      ? `${snapshot.active_count} running`
      : "Idle";
  rail.innerHTML = `<span class="dot ${kind}"></span>${esc(label)}`;
}

// Expanding a row is a local concern, so re-render from the snapshot already
// in hand rather than paying a round trip to learn what we were just told.
async function refreshActivity({ local = false } = {}) {
  if (local && lastSnapshot) {
    renderActivity(lastSnapshot);
    return;
  }
  try {
    renderActivity(await api("/api/activity"));
  } catch {
    /* the stream will catch up */
  }
}

// One delegated handler for both lists: rows are replaced on every snapshot,
// so anything bound to a row directly would not survive the next frame.
for (const id of ["#activity-active", "#activity-recent"]) {
  $(id).addEventListener("click", async (e) => {
    const expand = e.target.closest("[data-expand]");
    const approve = e.target.closest(".task-approve");
    const deny = e.target.closest(".task-deny");
    const cancel = e.target.closest(".task-cancel");

    if (approve || deny || cancel) {
      const button = approve || deny || cancel;
      const taskId = button.dataset.task;
      const [path, verb] = approve
        ? [`/api/tasks/${taskId}/approve`, "Allowed"]
        : deny
          ? [`/api/tasks/${taskId}/deny`, "Denied"]
          : [`/api/tasks/${taskId}/cancel`, "Stopped"];
      // Approving runs the tool, which can take a while — make it obvious the
      // click landed rather than leaving a live-looking button.
      button.disabled = true;
      button.textContent = approve ? "Running…" : "…";
      try {
        await api(path, { method: "POST" });
        toast(`${verb}.`);
      } catch (err) {
        toast(err.message);
      } finally {
        refreshActivity();
      }
      return;
    }

    if (expand) {
      const taskId = expand.dataset.expand;
      if (expandedTasks.has(taskId)) expandedTasks.delete(taskId);
      else expandedTasks.add(taskId);
      refreshActivity({ local: true });
    }
  });
}

function connectActivityStream() {
  let source;
  try {
    source = new EventSource("/api/events");
  } catch {
    setInterval(() => api("/api/activity").then(renderActivity).catch(() => {}), 4000);
    return;
  }
  source.addEventListener("snapshot", (event) => {
    try {
      renderActivity(JSON.parse(event.data));
    } catch {
      /* ignore a malformed frame */
    }
  });
  source.onerror = () => {
    // EventSource reconnects on its own; keep a slow poll as a safety net.
    api("/api/activity").then(renderActivity).catch(() => {});
  };
}

/* ————————————————————————————————————————————— assistant ——— */

let toolsCache = [];
let speechDevices = [];

function toolCard(tool) {
  const props = Object.keys(tool.parameters?.properties || {});
  const blocked = !tool.enabled;
  const unusable = blocked || !tool.available;
  // "Unavailable" means the machine lacks something; "off" means policy says no.
  const missing = tool.available === false && !blocked;
  return `<article class="tool-card ${tool.danger} ${unusable ? "is-off" : ""}" data-tool="${esc(tool.name)}">
    <div class="row between">
      <span class="name">${esc(tool.name)}</span>
      <div class="row">
        <span class="badge">${esc(tool.danger)}</span>
        ${tool.requires_confirmation ? `<span class="badge warn">asks first</span>` : ""}
      </div>
    </div>
    <p class="desc">${esc(tool.description)}</p>
    ${props.length ? `<p class="args">${esc(props.join(", "))}</p>` : ""}
    ${missing ? `<p class="args">unavailable: ${esc(tool.unavailable_reason || "")}</p>` : ""}
    <div class="tool-controls">
      <label class="check">
        <input type="checkbox" class="tool-enabled" ${blocked ? "" : "checked"} />
        <span>Enabled</span>
      </label>
      <label class="check">
        <input type="checkbox" class="tool-autoconfirm" ${tool.requires_confirmation ? "" : "checked"}
               ${tool.danger === "safe" ? "disabled" : ""} />
        <span>Run without asking</span>
      </label>
      <button type="button" class="btn ghost small tool-run" ${unusable ? "disabled" : ""}>Try it</button>
    </div>
  </article>`;
}

function renderTools(filter = "") {
  const needle = filter.trim().toLowerCase();
  const shown = needle
    ? toolsCache.filter(
        (t) =>
          t.name.toLowerCase().includes(needle) ||
          t.description.toLowerCase().includes(needle) ||
          t.category.toLowerCase().includes(needle)
      )
    : toolsCache;

  if (!shown.length) {
    $("#tool-grid").innerHTML = `<p class="empty">No tools match “${esc(filter)}”.</p>`;
    return;
  }

  const groups = new Map();
  for (const tool of shown) {
    if (!groups.has(tool.category)) groups.set(tool.category, []);
    groups.get(tool.category).push(tool);
  }
  $("#tool-grid").innerHTML = [...groups.entries()]
    .map(
      ([category, tools]) =>
        `<h4 class="tool-group-title">${esc(category)} · ${tools.length}</h4>` +
        tools.map(toolCard).join("")
    )
    .join("");
}

async function loadTools() {
  const data = await api("/api/tools");
  toolsCache = data.tools || [];
  renderTools($("#tool-search").value);
  $("#count-tools").textContent = toolsCache.filter((t) => t.enabled && t.available).length;
}

$("#tool-search").addEventListener("input", (e) => renderTools(e.target.value));

$("#tool-grid").addEventListener("change", async (e) => {
  const card = e.target.closest("[data-tool]");
  if (!card) return;
  const name = card.dataset.tool;
  const body = {};
  if (e.target.classList.contains("tool-enabled")) body.enabled = e.target.checked;
  else if (e.target.classList.contains("tool-autoconfirm")) body.auto_confirm = e.target.checked;
  else return;
  try {
    await api(`/api/tools/${encodeURIComponent(name)}/policy`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    toast(`${name} updated`);
    await Promise.all([loadTools(), loadAssistant(), loadStatus()]);
  } catch (err) {
    toast(err.message);
    await loadTools();
  }
});

$("#tool-grid").addEventListener("click", async (e) => {
  const btn = e.target.closest(".tool-run");
  if (!btn) return;
  const name = e.target.closest("[data-tool]").dataset.tool;
  const tool = toolsCache.find((t) => t.name === name);
  const props = tool?.parameters?.properties || {};
  const required = tool?.parameters?.required || [];

  const args = {};
  for (const [key, schema] of Object.entries(props)) {
    const hint = schema.description ? `\n${schema.description}` : "";
    const value = prompt(`${name} — ${key}${required.includes(key) ? " (required)" : ""}${hint}`, "");
    if (value === null) return;
    if (value === "" && !required.includes(key)) continue;
    args[key] = schema.type === "integer" ? Number(value)
      : schema.type === "boolean" ? /^(true|yes|1|on)$/i.test(value)
      : value;
  }

  btn.disabled = true;
  try {
    let res = await api("/api/tools/call", {
      method: "POST",
      body: JSON.stringify({ tool: name, arguments: args }),
    });
    if (res.requires_confirmation) {
      if (!confirm(`${name} is a ${res.danger} tool. Run it now?`)) return;
      res = await api("/api/tools/call", {
        method: "POST",
        body: JSON.stringify({ tool: name, arguments: args, confirmed: true }),
      });
    }
    alert(res.ok ? (res.output || "(no output)").slice(0, 4000) : `Failed: ${res.error}`);
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
  }
});

$("#refresh-tools").addEventListener("click", async (e) => {
  e.target.closest("button").disabled = true;
  try {
    const res = await api("/api/tools/refresh", { method: "POST", body: "{}" });
    await loadTools();
    await loadAssistant();
    toast(`${res.count} tools (${res.mcp_tools} from MCP servers)`);
  } catch (err) {
    toast(err.message);
  } finally {
    e.target.closest("button").disabled = false;
  }
});

let speechEngines = [];

function renderVoices(engineId, selected) {
  const engine = speechEngines.find((e) => e.id === engineId);
  const sel = $("#speech-voice");
  const voices = engine?.voices || [];
  sel.innerHTML = voices.length
    ? voices.map((v) => `<option value="${esc(v.id)}">${esc(v.name)}</option>`).join("")
    : `<option value="">— no voices —</option>`;
  if (selected && voices.some((v) => v.id === selected)) sel.value = selected;
}

function renderSpeechDevices(devices, selected, resolved) {
  const sel = $("#speech-device");
  // Unavailable devices stay in the list, disabled, carrying their reason.
  // Hiding a GPU that is physically present just looks like missing hardware.
  const options = [
    `<option value="auto">Automatic${resolved ? ` (using ${esc(resolved)})` : ""}</option>`,
    ...devices.map(
      (d) =>
        `<option value="${esc(d.id)}" ${d.available ? "" : "disabled"}>${esc(d.name)}${
          d.available ? (d.detail ? ` — ${esc(d.detail)}` : "") : " — unavailable"
        }</option>`
    ),
  ];
  sel.innerHTML = options.join("");
  const wanted = selected || "auto";
  sel.value = devices.some((d) => d.id === wanted && d.available) ? wanted : "auto";

  const chosen = devices.find((d) => d.id === sel.value);
  const blocked = devices.filter((d) => !d.available && d.detail);
  $("#speech-device-help").textContent = chosen?.detail
    ? chosen.detail
    : blocked.length
      ? `${blocked[0].name}: ${blocked[0].detail}`
      : "where synthesis happens";
}

async function loadSpeech(settings) {
  const data = await api("/api/speech");
  speechEngines = data.engines || [];
  speechDevices = data.devices || [];
  const s = settings || data.settings || {};
  const form = $("#assistant-form");

  form.speech_enabled.checked = !!s.enabled;
  form.auto_speak.checked = !!s.auto_speak;
  form.speech_rate.value = s.rate ?? 100;
  form.speech_pitch.value = s.pitch ?? 50;

  const usable = speechEngines.filter((e) => e.available);
  $("#speech-engine").innerHTML = usable.length
    ? usable.map((e) => `<option value="${esc(e.id)}">${esc(e.name)}</option>`).join("")
    : `<option value="">— none installed —</option>`;
  if (s.engine && usable.some((e) => e.id === s.engine)) $("#speech-engine").value = s.engine;
  renderVoices($("#speech-engine").value, s.voice);
  renderSpeechDevices(speechDevices, s.device, data.resolved_device);

  // Audio8 has no speed or pitch control. Leaving the inputs live would let
  // someone set a rate that silently does nothing, so grey them out and say so
  // rather than pretending.
  const chosen = speechEngines.find((e) => e.id === $("#speech-engine").value);
  for (const [field, supported] of [
    ["speech_rate", chosen?.supports_rate],
    ["speech_pitch", chosen?.supports_pitch],
  ]) {
    const input = form[field];
    if (!input) continue;
    input.disabled = chosen ? !supported : false;
    const help = input.closest("label")?.querySelector(".field-help");
    if (help && chosen && !supported) help.textContent = "not adjustable on this engine";
  }

  // Say what is missing and how to get it, rather than showing an empty list.
  const unavailable = speechEngines.filter((e) => !e.available && e.install_hint);
  $("#speech-note").textContent = usable.length
    ? chosen?.detail || ""
    : unavailable.length
      ? `${unavailable[0].name}: ${unavailable[0].detail} — ${unavailable[0].install_hint}`
      : "No speech engine is available.";
}

$("#speech-engine").addEventListener("change", (e) => renderVoices(e.target.value, ""));
$("#speech-device").addEventListener("change", (e) => {
  const chosen = speechDevices.find((d) => d.id === e.target.value);
  $("#speech-device-help").textContent =
    chosen?.detail || "picked automatically";
});

$("#speech-test").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  const note = $("#speech-note");
  const previous = note.textContent;
  note.textContent = "Speaking…";
  try {
    await api("/api/speech/speak", {
      method: "POST",
      body: JSON.stringify({
        text: "Keylane is ready. This is how answers will sound.",
        engine: $("#speech-engine").value,
        voice: $("#speech-voice").value,
        device: $("#speech-device").value,
        rate: Number($("#assistant-form").speech_rate.value) || 100,
        pitch: Number($("#assistant-form").speech_pitch.value) || 50,
      }),
    });
    note.textContent = previous;
  } catch (err) {
    note.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
});

async function loadAssistant() {
  const data = await api("/api/assistant");
  const s = data.settings || {};
  const form = $("#assistant-form");

  form.tools_enabled.checked = !!s.tools?.enabled;
  form.confirm_danger_at.value = s.tools?.confirm_danger_at || "sensitive";
  form.max_steps.value = s.tools?.max_steps ?? 6;
  form.persona.value = s.persona || "";

  form.delegation_enabled.checked = !!s.delegation?.enabled;
  form.delegation_follow_up.checked = !!s.delegation?.follow_up;
  form.max_delegations.value = s.delegation?.max_delegations ?? 2;
  form.agent_enabled.checked = s.agent?.enabled !== false;
  form.supervisor_backend.value = s.supervisor?.backend || "auto";
  form.supervisor_fallback.value = s.supervisor?.fallback_worker || "auto";

  form.search_engine.value = s.search?.engine || "duckduckgo";
  form.searxng_url.value = s.search?.searxng_url || "";
  form.search_max_results.value = s.search?.max_results ?? 5;

  form.shell_enabled.checked = !!s.shell?.enabled;
  form.shell_allowlist.value = (s.shell?.allowlist || []).join("\n");
  form.shell_working_directory.value = s.shell?.working_directory || "~";

  form.email_enabled.checked = !!s.email?.enabled;
  form.smtp_host.value = s.email?.smtp_host || "";
  form.smtp_port.value = s.email?.smtp_port ?? 587;
  form.use_tls.checked = !!s.email?.use_tls;
  form.email_username.value = s.email?.username || "";
  form.email_password.value = s.email?.password || "";
  form.from_address.value = s.email?.from_address || "";
  form.from_name.value = s.email?.from_name || "";
  form.allowed_recipients.value = (s.email?.allowed_recipients || []).join("\n");

  await loadSpeech(s.speech);

  $("#assistant-device").textContent = data.model_loaded
    ? `NPU model on ${data.device || "device"}`
    : "no model — keyword fallback";
  $("#system-prompt").textContent = data.system_prompt || "";

  const banner = $("#assistant-banner");
  banner.innerHTML = data.model_loaded
    ? ""
    : `<div class="banner">
        <svg viewBox="0 0 24 24"><use href="#i-alert" /></svg>
        <div>No OpenVINO router model is loaded, so the assistant recognises only a
        few obvious requests (open an app, search the web, system status) and hands
        everything else to a worker. Download a router model under <strong>Models</strong>
        to get the full plan–act–verify loop.</div>
      </div>`;
}

$("#assistant-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  const lines = (value) =>
    value
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);

  const body = {
    persona: form.persona.value,
    tools: {
      enabled: form.tools_enabled.checked,
      confirm_danger_at: form.confirm_danger_at.value,
      max_steps: Number(form.max_steps.value),
    },
    delegation: {
      enabled: form.delegation_enabled.checked,
      follow_up: form.delegation_follow_up.checked,
      max_delegations: Number(form.max_delegations.value),
    },
    supervisor: {
      backend: form.supervisor_backend.value,
      fallback_worker: form.supervisor_fallback.value,
    },
    agent: {
      enabled: form.agent_enabled.checked,
    },
    search: {
      engine: form.search_engine.value,
      searxng_url: form.searxng_url.value.trim(),
      max_results: Number(form.search_max_results.value),
    },
    shell: {
      enabled: form.shell_enabled.checked,
      allowlist: lines(form.shell_allowlist.value),
      working_directory: form.shell_working_directory.value.trim() || "~",
    },
    speech: {
      enabled: form.speech_enabled.checked,
      auto_speak: form.auto_speak.checked,
      engine: form.speech_engine.value,
      voice: form.speech_voice.value,
      device: form.speech_device.value,
      rate: Number(form.speech_rate.value) || 100,
      pitch: Number(form.speech_pitch.value) || 50,
    },
    email: {
      enabled: form.email_enabled.checked,
      smtp_host: form.smtp_host.value.trim(),
      smtp_port: Number(form.smtp_port.value),
      use_tls: form.use_tls.checked,
      username: form.email_username.value.trim(),
      password: form.email_password.value,
      from_address: form.from_address.value.trim(),
      from_name: form.from_name.value.trim(),
      allowed_recipients: lines(form.allowed_recipients.value),
    },
  };

  try {
    await api("/api/assistant", { method: "PUT", body: JSON.stringify(body) });
    $("#assistant-note").textContent = "Saved.";
    toast("Assistant settings saved");
    await Promise.all([loadAssistant(), loadTools(), loadStatus()]);
  } catch (err) {
    toast(err.message);
  }
});

/* ——————————————————————————————————————————————— plugins ——— */

function settingInputs(plugin) {
  const schema = plugin.settings_schema || [];
  if (!schema.length) return "";
  return `<div class="settings-grid" data-plugin-settings="${esc(plugin.id)}">
    ${schema
      .map((field) => {
        const val = plugin.settings?.[field.key] ?? field.default ?? "";
        if (field.type === "boolean") {
          return `<label class="check"><input type="checkbox" name="${esc(field.key)}" ${val ? "checked" : ""}/><span>${esc(field.label)}</span></label>`;
        }
        const type =
          field.type === "integer" ? "number" : field.type === "secret" ? "password" : "text";
        return `<label><span>${esc(field.label)}</span>
          <input name="${esc(field.key)}" type="${type}" value="${esc(val)}" />
          ${field.description ? `<span class="field-help">${esc(field.description)}</span>` : ""}
        </label>`;
      })
      .join("")}
    <button type="button" class="btn ghost save-settings" data-id="${esc(plugin.id)}">Save plugin settings</button>
  </div>`;
}

async function loadCatalog() {
  const data = await api("/api/plugins/catalog");
  const entries = data.plugins || [];
  $("#catalog-meta").textContent = `${data.installed} of ${data.count} installed`;
  $("#catalog-grid").innerHTML = entries.length
    ? entries
        .map(
          (e) => `<article class="tool-card ${e.installed ? "safe" : ""}" data-catalog="${esc(e.id)}">
        <div class="row between">
          <span class="name">${esc(e.name)}</span>
          <div class="row">
            ${e.cloud ? `<span class="badge warn">cloud</span>` : `<span class="badge">local</span>`}
            <span class="badge">${esc(e.kind)}</span>
          </div>
        </div>
        <p class="desc">${esc(e.description)}</p>
        ${e.tags?.length ? `<p class="args">${e.tags.map(esc).join(" · ")}</p>` : ""}
        <div class="tool-controls">
          ${
            e.installed
              ? `<span class="badge ok">installed</span>
                 <button type="button" class="btn danger small catalog-remove">Remove</button>`
              : `<button type="button" class="btn primary small catalog-install">Install</button>`
          }
          ${e.homepage ? `<a class="btn ghost small" href="${esc(e.homepage)}" target="_blank" rel="noopener">Homepage</a>` : ""}
        </div>
      </article>`
        )
        .join("")
    : `<p class="empty">No catalog entries found under <code>plugins/catalog/</code>.</p>`;
}

$("#catalog-grid").addEventListener("click", async (e) => {
  const card = e.target.closest("[data-catalog]");
  if (!card) return;
  const id = card.dataset.catalog;
  const install = e.target.closest(".catalog-install");
  const remove = e.target.closest(".catalog-remove");
  if (!install && !remove) return;

  const btn = install || remove;
  btn.disabled = true;
  try {
    if (install) {
      await api(`/api/plugins/catalog/${encodeURIComponent(id)}/install`, {
        method: "POST",
        body: "{}",
      });
      toast(`${id} installed`);
    } else {
      if (!confirm(`Remove ${id}? Its settings are kept.`)) {
        btn.disabled = false;
        return;
      }
      await api(`/api/plugins/catalog/${encodeURIComponent(id)}`, { method: "DELETE" });
      toast(`${id} removed`);
    }
    await Promise.all([loadCatalog(), loadPlugins(), loadTools(), loadStatus()]);
  } catch (err) {
    toast(err.message);
    btn.disabled = false;
  }
});

async function loadPlugins({ health = true } = {}) {
  const plugins = await api(`/api/plugins?health=${health ? "true" : "false"}`);
  $("#count-plugins").textContent = plugins.filter((p) => p.enabled).length;
  $("#plugin-list").innerHTML = plugins
    .map((p) => {
      const healthInfo = p.health;
      const healthBadge = !p.enabled
        ? `<span class="badge">disabled</span>`
        : healthInfo
          ? `<span class="badge ${healthInfo.ok ? "ok" : "bad"}">${healthInfo.ok ? "healthy" : "unreachable"}</span>`
          : health
            ? ""
            : `<span class="badge">checking</span>`;
      const toolCount = (p.tools || []).length;
      return `<article class="card">
        <div class="card-head">
          <div>
            <div class="row"><h3>${esc(p.name)}</h3>${healthBadge}</div>
            <p class="muted hint" style="margin-top:4px">${esc(p.description || "")}</p>
            ${healthInfo?.detail ? `<p class="muted hint" style="margin-top:4px">${esc(healthInfo.detail)}</p>` : ""}
          </div>
          <div class="row wrap" style="justify-content:flex-end">
            <span class="badge ${p.kind === "mcp" ? "mcp" : ""}">${esc(p.kind)}</span>
            ${p.worker_id ? `<span class="badge mono">${esc(p.worker_id)}</span>` : ""}
            ${p.cloud ? `<span class="badge warn">cloud</span>` : ""}
            ${toolCount ? `<span class="badge">${toolCount} tools</span>` : ""}
          </div>
        </div>
        <div class="card-actions">
          <button type="button" class="btn ${p.enabled ? "ghost" : "primary"} toggle-plugin"
                  data-id="${esc(p.id)}" data-enabled="${p.enabled}">
            ${p.enabled ? "Disable" : "Enable"}
          </button>
          ${p.homepage ? `<a class="btn ghost" href="${esc(p.homepage)}" target="_blank" rel="noopener">Homepage</a>` : ""}
          ${p.removable ? `<button type="button" class="btn danger remove-plugin" data-id="${esc(p.id)}">Remove</button>` : ""}
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
      await Promise.all([loadPlugins(), loadStatus(), loadTools()]);
    }
    if (remove) {
      await api(`/api/plugins/${remove.dataset.id}`, { method: "DELETE" });
      toast("Plugin removed");
      await Promise.all([loadPlugins(), loadTools()]);
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
      await loadPlugins();
    }
  } catch (err) {
    toast(err.message);
  }
});

$("#reload-plugins").addEventListener("click", async () => {
  try {
    const res = await api("/api/plugins/reload", { method: "POST", body: "{}" });
    await Promise.all([loadPlugins(), loadCatalog(), loadTools(), loadSkills()]);
    toast(`${res.count} plugins · ${res.tools} tools · ${res.skills} skills`);
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
    await Promise.all([loadPlugins(), loadTools()]);
  } catch (err) {
    toast(err.message);
  }
});

/* ———————————————————————————————————————————————— skills ——— */

let skillsCache = [];

const skillForm = $("#skill-form");

function showSkillForm(skill = null) {
  const f = skillForm;
  f.hidden = false;
  f.original.value = skill?.name || "";
  f.name.value = skill?.name || "";
  f.description.value = skill?.description || "";
  f.triggers.value = (skill?.triggers || []).join(", ");
  f.always.checked = !!skill?.always;
  f.enabled.checked = skill ? !!skill.enabled : true;
  f.content.value = skill?.content || "";
  $("#skill-form-title").textContent = skill ? `Edit “${skill.name}”` : "New skill";
  $("#skill-form-source").textContent = skill?.source || "user";
  $("#skill-note").textContent = "";
  f.scrollIntoView({ behavior: "smooth", block: "nearest" });
  f.name.focus();
}

function hideSkillForm() {
  skillForm.hidden = true;
  skillForm.reset();
}

$("#new-skill").addEventListener("click", () => showSkillForm(null));
$("#cancel-skill").addEventListener("click", hideSkillForm);

skillForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.currentTarget;
  const original = f.original.value;
  const body = {
    name: f.name.value.trim(),
    description: f.description.value.trim(),
    triggers: f.triggers.value
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean),
    always: f.always.checked,
    enabled: f.enabled.checked,
    content: f.content.value,
  };
  try {
    await api(original ? `/api/skills/${encodeURIComponent(original)}` : "/api/skills", {
      method: original ? "PUT" : "POST",
      body: JSON.stringify(body),
    });
    toast(original ? "Skill updated" : "Skill created");
    hideSkillForm();
    await loadSkills();
  } catch (err) {
    $("#skill-note").textContent = err.message;
  }
});

async function loadSkills() {
  const data = await api("/api/skills");
  skillsCache = data.skills || [];
  $("#skills-dir").textContent = data.directory || "skills/";
  $("#count-skills").textContent = skillsCache.length || "";

  $("#skill-list").innerHTML = skillsCache.length
    ? skillsCache
        .map((s) => {
          const fromPlugin = s.source !== "user";
          const inert = !s.always && !(s.triggers || []).length;
          return `<article class="skill-card" data-skill="${esc(s.name)}">
        <div class="row between">
          <div class="row">
            <strong>${esc(s.name)}</strong>
            ${s.enabled ? "" : `<span class="badge">off</span>`}
            ${s.always ? `<span class="badge ok">always on</span>` : ""}
            ${fromPlugin ? `<span class="badge mcp">${esc(s.source)}</span>` : ""}
          </div>
          <div class="row">
            <label class="check" title="Enable or disable this skill">
              <input type="checkbox" class="skill-enabled" ${s.enabled ? "checked" : ""}
                     ${fromPlugin ? "disabled" : ""} />
              <span>On</span>
            </label>
            ${fromPlugin ? "" : `<button type="button" class="btn ghost small skill-edit">Edit</button>`}
            ${fromPlugin ? "" : `<button type="button" class="btn danger small skill-delete">Delete</button>`}
          </div>
        </div>
        ${s.description ? `<p class="muted hint">${esc(s.description)}</p>` : ""}
        ${
          (s.triggers || []).length
            ? `<div class="trigger-list">${s.triggers.map((x) => `<span class="badge mono">${esc(x)}</span>`).join("")}</div>`
            : `<p class="muted hint">${
                s.always
                  ? "No triggers needed — always applied."
                  : "No triggers and not always-on, so this skill never activates."
              }</p>`
        }
        ${inert && !s.always ? `<p class="muted hint">⚠ add a trigger or turn on “always”.</p>` : ""}
        <pre>${esc(s.content)}</pre>
      </article>`;
        })
        .join("")
    : `<p class="empty">No skills yet. Press <strong>New skill</strong>, or drop a markdown file into <code>${esc(data.directory)}</code>.</p>`;
}

$("#skill-list").addEventListener("click", async (e) => {
  const card = e.target.closest("[data-skill]");
  if (!card) return;
  const name = card.dataset.skill;
  const skill = skillsCache.find((s) => s.name === name);

  if (e.target.closest(".skill-edit")) {
    showSkillForm(skill);
    return;
  }
  if (e.target.closest(".skill-delete")) {
    if (!confirm(`Delete the skill “${name}”? The markdown file is removed.`)) return;
    try {
      await api(`/api/skills/${encodeURIComponent(name)}`, { method: "DELETE" });
      toast("Skill deleted");
      await loadSkills();
    } catch (err) {
      toast(err.message);
    }
  }
});

$("#skill-list").addEventListener("change", async (e) => {
  if (!e.target.classList.contains("skill-enabled")) return;
  const name = e.target.closest("[data-skill]").dataset.skill;
  try {
    await api(`/api/skills/${encodeURIComponent(name)}/enable`, {
      method: "POST",
      body: JSON.stringify({ enabled: e.target.checked }),
    });
    await loadSkills();
  } catch (err) {
    toast(err.message);
    await loadSkills();
  }
});

async function loadSkillCatalog() {
  let data;
  try {
    data = await api("/api/skills/catalog");
  } catch (err) {
    $("#skill-catalog").innerHTML = `<p class="empty">${esc(err.message)}</p>`;
    return;
  }
  const skills = data.skills || [];
  $("#skill-catalog-meta").textContent = `${skills.length} suggested`;
  $("#skill-catalog").innerHTML = skills.length
    ? skills
        .map((s) => {
          const state = s.installed
            ? `<span class="badge ok">installed</span>`
            : s.unreachable
              ? `<span class="badge">source unreachable</span>`
              : s.verified
                ? `<button type="button" class="btn primary small skill-catalog-install"
                     data-repo="${esc(s.repo)}" data-path="${esc(s.path)}">Install</button>`
                : `<span class="badge warn">moved upstream</span>`;
          return `<article class="tool-card safe">
        <div class="row between">
          <span class="name">${esc(s.name)}</span>
          <span class="badge">${esc(s.installs || "")}</span>
        </div>
        <p class="desc">${esc(s.desc || "")}</p>
        <p class="args">${esc(s.source || s.repo)} · ${esc(s.repo)}</p>
        <div class="tool-controls">${state}</div>
      </article>`;
        })
        .join("")
    : `<p class="empty">No catalog entries under <code>skills/catalog/</code>.</p>`;
}

$("#skill-catalog").addEventListener("click", async (e) => {
  const btn = e.target.closest(".skill-catalog-install");
  if (!btn) return;
  btn.disabled = true;
  btn.textContent = "Installing…";
  try {
    const res = await api("/api/skills/import", {
      method: "POST",
      body: JSON.stringify({ repo: btn.dataset.repo, paths: [btn.dataset.path] }),
    });
    const failed = res.failed || [];
    toast(
      failed.length
        ? `Failed: ${failed[0].reason}`
        : "Installed — switch it on below when you want it active"
    );
    await Promise.all([loadSkills(), loadSkillCatalog()]);
  } catch (err) {
    toast(err.message);
    btn.disabled = false;
    btn.textContent = "Install";
  }
});

let discovered = { repo: "", skills: [] };

$("#skill-discover-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const repo = e.currentTarget.repo.value.trim();
  const btn = $("#discover-btn");
  const note = $("#discover-note");
  btn.disabled = true;
  note.textContent = "Reading the repository…";
  $("#discover-results").innerHTML = "";
  $("#import-actions").hidden = true;
  try {
    const data = await api("/api/skills/discover", {
      method: "POST",
      body: JSON.stringify({ repo }),
    });
    discovered = data;
    note.textContent = `${data.count} skill${data.count === 1 ? "" : "s"} in ${data.repo}@${data.ref}`;
    $("#discover-results").innerHTML = data.skills
      .map(
        (s) => `<label class="skill-card check-row">
        <input type="checkbox" class="import-pick" value="${esc(s.path)}" />
        <div>
          <strong>${esc(s.name)}</strong>
          <p class="muted hint">${esc(s.description)}</p>
          <p class="args">${esc(s.path)}</p>
        </div>
      </label>`
      )
      .join("");
    $("#import-actions").hidden = data.skills.length === 0;
  } catch (err) {
    note.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
});

$("#select-all-skills").addEventListener("click", () => {
  const boxes = $$(".import-pick");
  const target = !boxes.every((b) => b.checked);
  boxes.forEach((b) => (b.checked = target));
});

$("#import-selected").addEventListener("click", async (e) => {
  const paths = $$(".import-pick").filter((b) => b.checked).map((b) => b.value);
  if (!paths.length) {
    toast("Pick at least one skill");
    return;
  }
  e.target.disabled = true;
  try {
    const res = await api("/api/skills/import", {
      method: "POST",
      body: JSON.stringify({ repo: discovered.repo, paths }),
    });
    const failed = res.failed || [];
    toast(
      `Imported ${res.installed.length}${failed.length ? `, ${failed.length} failed` : ""} — all disabled until you switch them on`
    );
    $("#discover-note").textContent = failed.length
      ? failed.map((f) => `${f.path}: ${f.reason}`).join("; ")
      : res.note || "";
    await loadSkills();
  } catch (err) {
    toast(err.message);
  } finally {
    e.target.disabled = false;
  }
});

$("#reload-skills").addEventListener("click", async () => {
  try {
    const res = await api("/api/skills/reload", { method: "POST", body: "{}" });
    await loadSkills();
    toast(`${res.count} skills loaded`);
  } catch (err) {
    toast(err.message);
  }
});

/* ———————————————————————————————————————————————— themes ——— */

const BUILTIN_THEMES = ["default", "midnight", "panel", "paper", "studio", "orb"];

async function loadThemes() {
  const themes = await api("/api/themes");
  $("#theme-list").innerHTML = themes
    .map((t) => {
      const colors = t.preview_colors || {};
      const swatches = ["bg", "surface", "accent", "text"]
        .filter((k) => colors[k])
        .map((k) => `<span class="swatch" style="background:${esc(colors[k])}" title="${k}"></span>`)
        .join("");
      const popup = t.popup || {};
      return `<article class="theme-card ${t.active ? "is-active" : ""}">
        <div class="popup-preview" data-mode="${esc(popup.mode || "panel")}" aria-hidden="true">
          <div class="shape"></div>
        </div>
        <div class="swatches">${swatches}</div>
        <div>
          <div class="row between">
            <strong>${esc(t.name)}</strong>
            <span class="badge mono">${esc(popup.mode || "panel")}</span>
          </div>
          <div class="theme-meta">${esc(t.author)} · v${esc(t.version)}</div>
          <p class="muted hint" style="margin-top:6px">${esc(t.description || "")}</p>
        </div>
        <div class="card-actions">
          <button type="button" class="btn ${t.active ? "ghost" : "primary"} activate-theme"
                  data-id="${esc(t.id)}" ${t.active ? "disabled" : ""}>
            ${t.active ? "Active" : "Use theme"}
          </button>
          ${BUILTIN_THEMES.includes(t.id) ? "" : `<button type="button" class="btn danger remove-theme" data-id="${esc(t.id)}">Remove</button>`}
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
      await api("/api/themes/active", {
        method: "PUT",
        body: JSON.stringify({ id: activate.dataset.id }),
      });
      applyThemeCss();
      toast("Theme activated — this panel and the popup");
      await loadThemes();
    }
    if (remove) {
      await api(`/api/themes/${remove.dataset.id}`, { method: "DELETE" });
      applyThemeCss();
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

/* ——————————————————————————————————————————————— gateway ——— */

async function loadConfig() {
  const data = await api("/api/config");
  const form = $("#config-form");
  form.host.value = data.host;
  form.port.value = data.port;
  form.max_retries.value = data.max_retries;
  form.docs_url.value = data.docs_url || "/docs";
  if (form.result_corner) form.result_corner.value = data.result_corner || "top-right";
  form.local_only.checked = !!data.local_only;
  form.require_confirmation_for_modifications.checked =
    !!data.require_confirmation_for_modifications;
  form.allowed_project_roots.value = (data.allowed_project_roots || []).join("\n");
  $("#config-note").textContent = data.note || "";

  const link = $("#docs-link");
  const url = (data.docs_url || "/docs").trim();
  link.href = url.endsWith("/") || url.includes("#") ? url : `${url}/`;
}

$("#config-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  const body = {
    host: form.host.value.trim(),
    port: Number(form.port.value),
    max_retries: Number(form.max_retries.value),
    docs_url: form.docs_url.value.trim() || "/docs",
    result_corner: form.result_corner ? form.result_corner.value : undefined,
    local_only: form.local_only.checked,
    require_confirmation_for_modifications:
      form.require_confirmation_for_modifications.checked,
    allowed_project_roots: form.allowed_project_roots.value
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean),
  };
  try {
    const saved = await api("/api/config", { method: "PUT", body: JSON.stringify(body) });
    $("#config-note").textContent = saved.note || "Saved.";
    toast(saved.restart_required ? "Saved — restart the service for host/port." : "Settings saved.");
    await loadConfig();
  } catch (err) {
    toast(err.message);
  }
});

/* ——————————————————————————————————————————————— projects ——— */

function projectRow(name = "", path = "") {
  return `<div class="project-row row wrap">
    <label class="grow"><span>Name</span><input class="project-name" value="${esc(name)}" placeholder="aurora" /></label>
    <label class="grow" style="flex:2"><span>Path</span><input class="project-path" value="${esc(path)}" placeholder="~/Documents/Code/aurora" /></label>
    <button type="button" class="btn danger small remove-project">Remove</button>
  </div>`;
}

async function loadProjects() {
  const data = await api("/api/projects");
  const rows = (data.projects || []).map((p) => projectRow(p.name, p.path));
  $("#project-rows").innerHTML = rows.length ? rows.join("") : projectRow();
}

$("#add-project").addEventListener("click", () => {
  $("#project-rows").insertAdjacentHTML("beforeend", projectRow());
});

$("#project-rows").addEventListener("click", (e) => {
  if (e.target.closest(".remove-project")) e.target.closest(".project-row").remove();
});

$("#save-projects").addEventListener("click", async () => {
  const projects = $$(".project-row")
    .map((row) => ({
      name: $(".project-name", row).value.trim(),
      path: $(".project-path", row).value.trim(),
    }))
    .filter((p) => p.name && p.path);
  try {
    const res = await api("/api/projects", {
      method: "PUT",
      body: JSON.stringify({ projects }),
    });
    const rejected = res.rejected || [];
    $("#projects-note").textContent = rejected.length
      ? `Saved ${res.projects.length}. Rejected: ${rejected.map((r) => `${r.name} (${r.reason})`).join("; ")}`
      : `Saved ${res.projects.length} project${res.projects.length === 1 ? "" : "s"}.`;
    toast(rejected.length ? "Saved, some entries rejected" : "Projects saved");
    await loadProjects();
  } catch (err) {
    toast(err.message);
  }
});

/* ——————————————————————————————————————— worker endpoints ——— */

const WORKER_FIELDS = [
  "lmstudio_base_url", "lmstudio_default_model", "lmstudio_timeout_seconds",
  "lemonade_base_url", "lemonade_default_model", "lemonade_timeout_seconds",
  "comfyui_base_url", "comfyui_output_dir", "comfyui_timeout_seconds",
  "claude_command", "claude_timeout_seconds",
  "cursor_command", "cursor_timeout_seconds",
  "audio_sample_rate", "audio_channels",
];

async function loadWorkerEndpoints() {
  const { settings } = await api("/api/workers");
  const form = $("#workers-form");
  for (const key of WORKER_FIELDS) {
    if (form[key]) form[key].value = settings[key] ?? "";
  }
}

$("#workers-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  const body = {};
  for (const key of WORKER_FIELDS) {
    const el = form[key];
    if (!el) continue;
    const raw = el.value.trim();
    if (raw === "") continue;
    body[key] = el.type === "number" ? Number(raw) : raw;
  }
  try {
    const res = await api("/api/workers", { method: "PUT", body: JSON.stringify(body) });
    $("#workers-note").textContent = res.note || "Saved.";
    toast("Worker endpoints saved");
    await Promise.all([loadWorkerEndpoints(), loadStatus()]);
  } catch (err) {
    toast(err.message);
  }
});

/* ————————————————————————————————————————— system info ——— */

async function loadSystemInfo() {
  const d = await api("/api/system");
  const row = (label, value) =>
    `<div class="row between info-row"><span class="muted">${esc(label)}</span>` +
    `<code>${esc(value)}</code></div>`;

  const pipes = d.pipelines || {};
  const counts = d.counts || {};
  const paths = d.paths || {};
  const hw = d.hardware || {};

  $("#system-info").innerHTML = [
    row("Version", d.version),
    row("Python", d.python),
    row("Platform", d.platform),
    hw.summary ? row("Hardware", hw.summary) : "",
    row(
      "Router model",
      pipes.router?.loaded ? `loaded on ${pipes.router.device}` : "not loaded"
    ),
    row(
      "Verifier model",
      pipes.verifier?.loaded ? `loaded on ${pipes.verifier.device}` : "shares the router"
    ),
    row(
      "Counts",
      `${counts.plugins} plugins · ${counts.tools} tools · ${counts.skills} skills · ${counts.projects} projects`
    ),
    ...Object.entries(paths).map(([k, v]) => row(k, v)),
  ]
    .filter(Boolean)
    .join("");
}

/* ———————————————————————————————————————————————— models ——— */

let modelsCache = null;

function fillSelect(sel, items, selected, valueKey = "id", labelFn = null, emptyLabel = "— none available —") {
  if (!sel) return;
  const list = items || [];
  const current = selected || sel.value;
  if (!list.length) {
    sel.innerHTML = `<option value="">${esc(emptyLabel)}</option>`;
    sel.value = "";
    return;
  }
  sel.innerHTML = list
    .map((item) => {
      const value = typeof item === "string" ? item : item[valueKey];
      const label = labelFn
        ? labelFn(item)
        : typeof item === "string"
          ? item
          : `${item.name}${item.recommended ? " ★" : ""}`;
      return `<option value="${esc(value)}">${esc(label)}</option>`;
    })
    .join("");
  if (current && [...sel.options].some((o) => o.value === current)) sel.value = current;
  else sel.selectedIndex = 0;
}

function repoFromUrl(url) {
  const match = String(url || "").match(/huggingface\.co\/([^/]+\/[^/?#]+)/);
  return match ? match[1] : "";
}

function recCard(item, kind) {
  const ready = kind === "router" ? !!item.installed : !!item.available;
  const repo = repoFromUrl(item.hf_url);
  const badge = ready
    ? `<span class="badge ok">downloaded</span>`
    : item.recommended
      ? `<span class="badge">suggested</span>`
      : "";
  const link = item.hf_url
    ? `<a class="muted hint" href="${esc(item.hf_url)}" target="_blank" rel="noopener">Hugging Face</a>`
    : "";

  // The button says what it will actually do. A dead "Not downloaded" told
  // the user the problem and gave them no way to fix it.
  let action;
  if (ready) {
    action = `<button type="button" class="btn ghost small use-rec" data-kind="${kind}" data-id="${esc(item.id)}">Use</button>`;
  } else if (item.gated) {
    action = `<a class="btn ghost small" href="${esc(item.hf_url)}" target="_blank" rel="noopener"
                 title="This repository needs its licence accepted on Hugging Face first">Get access</a>`;
  } else if (repo) {
    action = `<button type="button" class="btn primary small rec-download"
                data-repo="${esc(repo)}" data-target="${kind === "router" ? "router" : "chat"}"
                data-id="${esc(item.id)}">Download</button>`;
  } else {
    action = `<button type="button" class="btn ghost small" disabled
                title="No download source for this model">Load it in LM Studio</button>`;
  }

  return `<article class="rec-card ${ready ? "is-rec" : ""}" data-rec="${esc(item.id)}">
    <div class="row between"><strong>${esc(item.name)}</strong>${badge}</div>
    <p class="muted meta">${esc(item.size_hint || "")} ${item.quant ? `· ${esc(item.quant)}` : ""}</p>
    <p class="notes">${esc(item.notes || item.reason || "")}</p>
    <div class="row between actions">
      <span class="muted reason">${esc(item.reason || "")}</span>
      <div class="row">${link}${action}</div>
    </div>
  </article>`;
}

function fillWorkerModelSelect(sel, items, selected) {
  if (!sel) return;
  const ids = (items || []).map((m) => (typeof m === "string" ? m : m.id)).filter(Boolean);
  const current = selected || sel.value || "";
  const opts = ['<option value="">— leave blank for auto —</option>'];
  if (!ids.length) opts.push('<option value="" disabled>No models detected from this worker</option>');
  for (const id of ids) opts.push(`<option value="${esc(id)}">${esc(id)}</option>`);
  sel.innerHTML = opts.join("");
  sel.value = current && ids.includes(current) ? current : "";
}

function findRec(kind, id) {
  return (modelsCache?.recommendations?.[kind] || []).find((m) => m.id === id);
}

async function loadModels() {
  const data = await api("/api/models");
  modelsCache = data;
  const form = $("#models-form");
  if (!form) return;
  const s = data.settings || {};
  const hw = data.hardware || {};
  const rec = data.recommendations || {};

  const summary = $("#hardware-summary");
  if (summary) summary.textContent = hw.summary || "Hardware profile unavailable";
  const chips = $("#hardware-chips");
  if (chips) {
    chips.innerHTML = (hw.devices || [])
      .map((d) => `<span class="chip ${esc(d.kind)}">${esc((d.kind || "").toUpperCase())} · ${esc(d.name)}</span>`)
      .join("");
  }
  const tips = $("#hardware-tips");
  if (tips) tips.innerHTML = (rec.guidance || []).map((t) => `<li>${esc(t)}</li>`).join("");

  const available = data.available || {};
  const routerModels = available.router || rec.installed || [];
  fillSelect(
    form.router_model_id,
    routerModels,
    s.router_model_id,
    "id",
    (m) => `${m.name || m.id}${m.size_hint ? ` (${m.size_hint})` : ""}`,
    "— no OpenVINO models downloaded —"
  );

  fillWorkerModelSelect(form.lmstudio_model, available.lmstudio || [], s.lmstudio_model || s.chat_model_id || "");
  fillWorkerModelSelect(form.lemonade_model, available.lemonade || [], s.lemonade_model || "");
  fillWorkerModelSelect(form.comfyui_model, available.comfyui || [], s.comfyui_model || "");

  form.primary_device.value = s.primary_device || "auto";
  form.npu_device.value = s.npu_device || "NPU";
  form.gpu_device.value = s.gpu_device || "GPU";
  form.fallback_device.value = s.fallback_device || "CPU";
  form.verifier_model_id.value = s.verifier_model_id || "";
  form.verifier_model_path.value = s.verifier_model_path || "";
  form.preferred_chat_worker.value = s.preferred_chat_worker || "auto";
  form.lmstudio_mode.value = s.lmstudio_mode || "auto";
  form.lemonade_mode.value = s.lemonade_mode || "auto";
  form.comfyui_mode.value = s.comfyui_mode || "auto";
  if (form.lemonade_base_url)
    form.lemonade_base_url.value = s.lemonade_base_url || "http://127.0.0.1:13305/api/v1";
  if (form.chat_model_id) form.chat_model_id.value = form.lmstudio_model.value || s.chat_model_id || "";
  form.whisper_model.value = s.whisper_model || "tiny";
  form.autostart_gateway.checked = !!s.autostart_gateway;
  form.autostart_launcher.checked = !!s.autostart_launcher;
  form.open_panel_on_login.checked = !!s.open_panel_on_login;

  const selectedRouter = routerModels.find((m) => m.id === form.router_model_id.value);
  form.router_model_path.value = selectedRouter?.path || s.router_model_path || "";

  const resolved = $("#resolved-device");
  if (resolved) resolved.textContent = data.resolved_device || "—";
  const auto = data.autostart || {};
  const autoEl = $("#autostart-status");
  if (autoEl) {
    autoEl.textContent =
      `Login — gateway: ${auto.gateway_enabled ? "on" : "off"}, ` +
      `launcher: ${auto.launcher_enabled ? "on" : "off"}, ` +
      `panel: ${auto.panel_autostart ? "on" : "off"}`;
  }

  const rr = $("#rec-router");
  if (rr) rr.innerHTML = (rec.router || []).slice(0, 4).map((m) => recCard(m, "router")).join("");
  const rc = $("#rec-chat");
  if (rc) rc.innerHTML = (rec.chat || []).slice(0, 4).map((m) => recCard(m, "chat")).join("");

  // "./models" is relative to the *install*, not to wherever the user has a
  // checkout open — which is exactly the confusion worth heading off.
  const paths = data.paths || {};
  const where = $("#models-location");
  if (where) {
    where.innerHTML = paths.router
      ? `Downloads go to <code>${esc(paths.router)}</code>`
      : "";
  }

  const installed = rec.installed || [];
  const inst = $("#installed-models");
  if (inst) {
    inst.innerHTML = installed.length
      ? installed
          .map(
            (m) => `<article class="rec-card${m.ready === false ? " incomplete" : ""}">
              <strong>${esc(m.name)}</strong>
              <p class="muted meta" title="${esc(m.absolute || m.path)}">${esc(m.absolute || m.path)}</p>
              ${
                m.ready === false
                  ? `<p class="warn-text">Incomplete — missing ${esc(
                      (m.missing || []).join(", ")
                    )}</p>
                     <button type="button" class="btn small rec-download"
                       data-repo="${esc(m.repo_id)}" data-target="router">Resume download</button>`
                  : `<button type="button" class="btn ghost small use-installed" data-path="${esc(
                      m.path
                    )}" data-id="${esc(m.id)}">Use as router</button>`
              }
            </article>`
          )
          .join("")
      : `<p class="empty">Nothing downloaded yet. Use a <strong>Download</strong> button above, or search Hugging Face.</p>`;
  }
}

function applyRouterChoice(item) {
  const form = $("#models-form");
  if (!form || !item) return false;
  const installed = modelsCache?.available?.router || modelsCache?.recommendations?.installed || [];
  const match =
    installed.find((m) => m.id === item.id || m.path === item.path || m.path === item.path_hint) ||
    (item.installed && (item.path || item.id) ? item : null);
  if (!match) {
    toast("That router model is not downloaded yet — use the Hugging Face search.");
    return false;
  }
  if (![...form.router_model_id.options].some((o) => o.value === match.id)) {
    const opt = document.createElement("option");
    opt.value = match.id;
    opt.textContent = match.name || match.id;
    form.router_model_id.append(opt);
  }
  form.router_model_id.value = match.id;
  form.router_model_path.value = match.path || item.path_hint || "";
  return true;
}

function applyChatChoice(item) {
  const form = $("#models-form");
  if (!form || !item) return false;
  const sel = form.lmstudio_model;
  const available = (modelsCache?.available?.lmstudio || []).map((m) => m.id);
  let mid = available.includes(item.id) ? item.id : "";
  if (!mid) {
    const token = String(item.id || "").toLowerCase();
    mid =
      available.find((id) => {
        const lower = id.toLowerCase();
        return lower.includes(token) || token.includes(lower);
      }) || "";
  }
  if (!mid) {
    toast("That chat model is not loaded in LM Studio yet.");
    return false;
  }
  if (sel) sel.value = mid;
  if (form.chat_model_id) form.chat_model_id.value = mid;
  form.lmstudio_mode.value = "fixed";
  form.preferred_chat_worker.value = "lmstudio";
  return true;
}

$("#models-form")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  const body = {
    primary_device: form.primary_device.value,
    npu_device: form.npu_device.value.trim(),
    gpu_device: form.gpu_device.value.trim(),
    fallback_device: form.fallback_device.value.trim(),
    router_model_id: form.router_model_id.value,
    router_model_path: form.router_model_path.value.trim(),
    verifier_model_id: form.verifier_model_id.value.trim(),
    verifier_model_path: form.verifier_model_path.value.trim(),
    preferred_chat_worker: form.preferred_chat_worker.value,
    lmstudio_mode: form.lmstudio_mode.value,
    lmstudio_model: form.lmstudio_model.value,
    lemonade_mode: form.lemonade_mode.value,
    lemonade_model: form.lemonade_model.value,
    lemonade_base_url: form.lemonade_base_url.value.trim(),
    comfyui_mode: form.comfyui_mode.value,
    comfyui_model: form.comfyui_model.value,
    chat_model_id: form.lmstudio_model.value,
    chat_backend:
      form.preferred_chat_worker.value === "auto" ? "lmstudio" : form.preferred_chat_worker.value,
    whisper_model: form.whisper_model.value,
    autostart_gateway: form.autostart_gateway.checked,
    autostart_launcher: form.autostart_launcher.checked,
    open_panel_on_login: form.open_panel_on_login.checked,
  };
  try {
    const saved = await api("/api/models", { method: "PUT", body: JSON.stringify(body) });
    const note = $("#models-note");
    if (note) note.textContent = saved.note || "Saved.";
    const resolved = $("#resolved-device");
    if (resolved) resolved.textContent = saved.resolved_device || "—";
    toast(saved.restart_required ? "Saved — reload models to apply." : "Model settings saved.");
    await loadModels();
  } catch (err) {
    toast(err.message);
  }
});

$("#reload-models-btn")?.addEventListener("click", async () => {
  try {
    const res = await api("/api/models/reload", { method: "POST", body: "{}" });
    toast(
      res.status === "reloaded"
        ? `Reloaded${res.device ? ` on ${res.device}` : ""} — router ${res.loaded ? "loaded" : "not found"}`
        : res.note || "Restart required"
    );
    await Promise.all([loadStatus(), loadAssistant()]);
  } catch (err) {
    toast(err.message);
  }
});

$("#apply-top-recs")?.addEventListener("click", () => {
  const installed = modelsCache?.available?.router || modelsCache?.recommendations?.installed || [];
  const router = installed[0] || (modelsCache?.recommendations?.router || []).find((m) => m.installed);
  const chat = (modelsCache?.recommendations?.chat || []).find((m) => m.available);
  let applied = 0;
  if (router && applyRouterChoice(router)) applied += 1;
  if (chat && applyChatChoice(chat)) applied += 1;
  const suggestion = modelsCache?.recommendations?.primary_suggestion;
  if (suggestion) $("#models-form").primary_device.value = suggestion;
  toast(
    applied
      ? "Applied downloaded models — click Save to keep them."
      : "Nothing downloaded to apply yet."
  );
});

document.addEventListener("click", (e) => {
  const useRec = e.target.closest(".use-rec");
  if (useRec) {
    const kind = useRec.dataset.kind === "chat" ? "chat" : "router";
    const item = findRec(kind, useRec.dataset.id);
    const ok = kind === "chat" ? applyChatChoice(item) : applyRouterChoice(item);
    if (ok) toast(`Selected ${item?.name || useRec.dataset.id}`);
  }
  const useInst = e.target.closest(".use-installed");
  if (useInst) {
    const form = $("#models-form");
    form.router_model_path.value = useInst.dataset.path;
    if (![...form.router_model_id.options].some((o) => o.value === useInst.dataset.id)) {
      const opt = document.createElement("option");
      opt.value = useInst.dataset.id;
      opt.textContent = useInst.dataset.id;
      form.router_model_id.append(opt);
    }
    form.router_model_id.value = useInst.dataset.id;
    toast("Router path set — click Save.");
  }
});

$("#models-form")?.router_model_id?.addEventListener("change", (e) => {
  const installed = modelsCache?.available?.router || modelsCache?.recommendations?.installed || [];
  const match = installed.find((m) => m.id === e.target.value);
  if (match?.path) $("#models-form").router_model_path.value = match.path;
});

/* ————————————————————————————————— Hugging Face downloads ——— */

let hfPollTimer = null;

function formatBytes(n) {
  if (!n || n < 1) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function hfResultCard(item) {
  const badge = item.compatible
    ? `<span class="badge ok">compatible</span>`
    : `<span class="badge">low match</span>`;
  const installed = item.installed ? `<span class="badge">installed</span>` : "";
  const params = item.params_b != null ? `${item.params_b}B · ` : "";
  return `<article class="rec-card ${item.compatible ? "is-rec" : ""}" data-repo="${esc(item.repo_id)}">
    <div class="row between">
      <strong>${esc(item.repo_id)}</strong>
      <div class="row">${badge}${installed}</div>
    </div>
    <p class="muted meta">${params}${(item.downloads || 0).toLocaleString()} downloads → <code>${esc(item.dest_hint || "")}</code></p>
    <p class="notes">${esc(item.reason || "")}</p>
    <div class="row between actions">
      <a class="muted hint" href="${esc(item.hf_url)}" target="_blank" rel="noopener">Hugging Face</a>
      <button type="button" class="btn ghost small hf-download-btn"
        data-repo="${esc(item.repo_id)}" data-target="${esc(item.target)}">
        ${item.installed ? "Re-download" : "Download"}
      </button>
    </div>
  </article>`;
}

function renderHfJobs(jobs) {
  const el = $("#hf-downloads");
  if (!el) return;
  const active = (jobs || []).slice(0, 6);
  if (!active.length) {
    el.innerHTML = "";
    return;
  }
  el.innerHTML =
    `<h4>Downloads</h4>` +
    active
      .map((j) => {
        const pct = Math.round((j.progress || 0) * 100);
        const size = formatBytes(j.bytes_downloaded);
        return `<div class="hf-job ${esc(j.status)}">
        <div class="row between">
          <strong>${esc(j.repo_id)}</strong>
          <span class="muted hint">${esc(j.status)}${size ? ` · ${size}` : ""}</span>
        </div>
        <div class="hf-progress"><div style="width:${pct}%"></div></div>
        <p class="muted meta">${esc(j.message || j.error || "")}</p>
        ${j.dest ? `<p class="args" title="${esc(j.dest)}">${esc(j.dest)}</p>` : ""}
      </div>`;
      })
      .join("");
}

async function refreshHfJobs() {
  try {
    const data = await api("/api/models/hf/downloads");
    renderHfJobs(data.jobs || []);
    const busy = (data.jobs || []).some((j) => j.status === "queued" || j.status === "running");
    if (busy && !hfPollTimer) {
      hfPollTimer = setInterval(() => refreshHfJobs().catch(() => {}), 2000);
    }
    if (!busy && hfPollTimer) {
      clearInterval(hfPollTimer);
      hfPollTimer = null;
      await loadModels();
    }
  } catch {
    /* ignore while offline */
  }
}

$("#hf-search-form")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  const btn = $("#hf-search-btn");
  const results = $("#hf-results");
  const meta = $("#hf-search-meta");
  btn.disabled = true;
  results.innerHTML = `<p class="muted hint">Searching Hugging Face…</p>`;
  try {
    const params = new URLSearchParams({
      q: form.q.value.trim(),
      target: form.target.value,
      limit: "12",
    });
    const data = await api(`/api/models/hf/search?${params}`);
    meta.textContent = `${data.hardware_summary || ""} · scored for ${data.tier || "this device"} · ${data.count || 0} results`;
    results.innerHTML = (data.results || []).length
      ? data.results.map(hfResultCard).join("")
      : `<p class="empty">No matching models. Try a different query or target.</p>`;
  } catch (err) {
    results.innerHTML = "";
    toast(err.message || "Search failed");
  } finally {
    btn.disabled = false;
  }
});

document.addEventListener("click", async (e) => {
  const btn = e.target.closest(".rec-download");
  if (!btn) return;
  btn.disabled = true;
  btn.textContent = "Starting…";
  try {
    const job = await api("/api/models/hf/download", {
      method: "POST",
      body: JSON.stringify({ repo_id: btn.dataset.repo, target: btn.dataset.target }),
    });
    toast(
      btn.dataset.target === "chat"
        ? `Downloading ${job.repo_id} into LM Studio`
        : `Downloading ${job.repo_id} → ${job.dest || "models/router"}`
    );
    setTab("models");
    await refreshHfJobs();
  } catch (err) {
    toast(err.message || "Download failed to start");
    btn.disabled = false;
    btn.textContent = "Download";
  }
});

document.addEventListener("click", async (e) => {
  const btn = e.target.closest(".hf-download-btn");
  if (!btn) return;
  btn.disabled = true;
  try {
    const job = await api("/api/models/hf/download", {
      method: "POST",
      body: JSON.stringify({ repo_id: btn.dataset.repo, target: btn.dataset.target }),
    });
    toast(`Download started: ${job.repo_id}`);
    await refreshHfJobs();
  } catch (err) {
    toast(err.message || "Download failed to start");
    btn.disabled = false;
  }
});

/* ————————————————————————————————————————————————— boot ——— */

async function refreshCurrentTab() {
  const hash = (location.hash || "#status").slice(1);
  const tab = titles[hash] ? hash : "status";
  loadedTabs.delete(tab);
  // Status counts are cheap and keep the rail honest after a refresh.
  if (tab !== "status") {
    loadStatus().catch(() => {});
  }
  await ensureTab(tab);
}

async function warmRailCounts() {
  // Fire-and-forget metadata so nav badges fill without blocking first paint.
  api("/api/plugins?health=false")
    .then((plugins) => {
      $("#count-plugins").textContent = plugins.filter((p) => p.enabled).length;
    })
    .catch(() => {});
  api("/api/skills")
    .then((data) => {
      $("#count-skills").textContent = (data.skills || []).length || "";
    })
    .catch(() => {});
}

$("#refresh-btn").addEventListener("click", (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  refreshCurrentTab()
    .then(() => toast("Refreshed"))
    .catch((err) => toast(err.message))
    .finally(() => {
      btn.disabled = false;
    });
});

const initial = (location.hash || "#status").slice(1);
const startTab = titles[initial] ? initial : "status";
setTab(startTab);
connectActivityStream();
warmRailCounts();
if (startTab === "models") {
  refreshHfJobs().catch(() => {});
}