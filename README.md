<p align="center">
  <img src="assets/logo.png" alt="Keylane logo" width="128" height="128" />
</p>

<h1 align="center">Keylane</h1>

<p align="center">
  <strong>Your Jarvis — always on, on your NPU</strong><br />
  Press Super+Space. A ~9B model on the Intel NPU helps you think, search, plan, and act.
</p>

## What is Keylane?

Keylane is a personal AI assistant that lives on your desktop:

- **Super+Space** opens a Spotlight-style bar that collapses into a floating orb while working,
  then expands into an answer HUD in the top-right corner
- The orb and HUD stay **above every other window**, and take up only their own small
  window — the rest of the screen stays fully clickable while Keylane works
- Answers render as a **formatted canvas** — headline, sections, key/value rows, steps,
  and code blocks — not one wrapped paragraph
- **Ask a follow-up** directly in the answer HUD without reopening the bar
- A **curated 3–14B model** runs **always-on** on your **Intel NPU**, through either
  **OpenVINO GenAI** or **ONNX Runtime GenAI** — pick the runtime in Settings
- **Import any model from Hugging Face** that either runtime can load
- **Persistent memory** — Keylane remembers facts about you across sessions and recalls
  them before answering anything personal
- **Reminders and watchers** that survive restarts — "remind me to call Sam at 6",
  or an opt-in morning sweep of your calendar and unanswered mail
- **Agentic web research** — pluggable search (SearXNG / DDGS fallback), evidence compression, cited answers
- **Todo list**, **goals**, **background jobs**, **subagents**, and an **inbox** of
  results you have not seen yet
- **Voice input** (mic button) via Whisper, and **screenshot capture** to ask about what is on screen
- **MCP servers** over stdio *or* Streamable HTTP — including Mailspring for mail and calendar
- **Audio8 TTS** for spoken answers and optional notify speech

Everything binds to `127.0.0.1`. Nothing leaves your machine unless a tool or MCP server you configured does.

## Quick start

```bash
# Install system deps (Fedora)
sudo dnf install python3-gobject gtk4 gtk4-layer-shell libnotify portaudio ffmpeg

# Python env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start daemon (NPU warm-up may take minutes first run)
PYTHONPATH=. python -m daemon.main

# In another terminal — Spotlight UI
PYTHONPATH=. python ui/main.py

# Bind Super+Space to toggle
PYTHONPATH=. python ui/main.py --toggle
```

Or use `./scripts/install.sh` for a user-level systemd install.

## Settings

Open **Settings** from the gear icon in the Spotlight footer, or press **`Ctrl+,`**.

| Tab | Options |
| --- | --- |
| General | Assistant name, your name, iteration budget |
| Model | Inference runtime, device, active model, switch / download, Hugging Face import |
| Web | Search backend (searxng / ddgs), SearXNG URL, Playwright, fallback |
| Speech | TTS on notify, read aloud, test buttons |
| Security | Shell allowlist, permitted read directories, permission modes per tool |
| MCP | Servers over stdio (command, arguments, environment) or HTTP (URL, bearer token) |
| Appearance (in General) | Theme, and light / dark / system |

CLI:

```bash
PYTHONPATH=. python scripts/keylane-settings list
PYTHONPATH=. python scripts/keylane-settings set research.search_backend ddgs
PYTHONPATH=. python scripts/keylane-settings wizard web
```

Settings persist to `data/settings.json` (merged over `config/*.toml` defaults).

## Runtimes and the model catalog

Keylane can run a local model through two inference stacks. Pick one in
**Settings → Model → Runtime**; the model list filters to what it can load.

| Runtime | Loads | Install |
| --- | --- | --- |
| **OpenVINO GenAI** (default) | OpenVINO IR exports — repos named `*-int4-ov` | in `requirements.txt` |
| **ONNX Runtime GenAI** | ONNX exports with a `genai_config.json` | `pip install onnxruntime-genai onnxruntime-openvino` |

The runtime is a property of the export, not a preference: an `*-int4-ov` repo
holds OpenVINO IR and only OpenVINO GenAI can load it. ONNX Runtime reaches the
same NPU through the **OpenVINO execution provider**, which is what the second
package is for — without it, ONNX models run on CPU only.

> Installing `onnxruntime-openvino` may pin a different OpenVINO version than
> the one `openvino-genai` wants. If OpenVINO GenAI stops loading afterwards,
> install the two runtimes in separate virtualenvs and keep the one you use.

**Device** is chosen per runtime in the same panel (`NPU` / `GPU` / `CPU`, plus
`AUTO` on ONNX Runtime, which keeps whatever provider the model shipped with).
Changing it invalidates the compile cache, so the next load is slow either way.

### Curated models

Pick one in **Settings → Model** or via `POST /models/select`. They auto-download
from Hugging Face on first activation.

| ID | Model | Runtime |
| --- | --- | --- |
| `qwen2.5-7b-instruct` | Qwen 2.5 7B Instruct (default) | OpenVINO |
| `qwen2.5-7b-npu` | Qwen 2.5 7B NPU-tested | OpenVINO |
| `qwen2.5-coder-7b` | Qwen 2.5 Coder 7B | OpenVINO |
| `qwen3-8b` | Qwen 3 8B | OpenVINO |
| `qwen3.5-9b` | Qwen 3.5 9B (VLM) | OpenVINO |
| `gemma-2-9b-it` | Gemma 2 9B Instruct | OpenVINO |
| `mistral-7b-instruct-v03` | Mistral 7B Instruct v0.3 | OpenVINO |
| `deepseek-r1-qwen-7b` | DeepSeek R1 Distill Qwen 7B | OpenVINO |
| `phi-4-mini-instruct` | Phi 4 Mini Instruct | OpenVINO |
| `phi-3.5-mini-instruct` | Phi 3.5 Mini Instruct | OpenVINO |
| `phi-4-mini-onnx` | Phi 4 Mini Instruct | ONNX Runtime |
| `phi-3.5-mini-onnx` | Phi 3.5 Mini Instruct (AWQ) | ONNX Runtime |
| `phi-4-mini-reasoning-onnx` | Phi 4 Mini Reasoning | ONNX Runtime |
| `llama-3.2-3b-onnx` | Llama 3.2 3B Instruct | ONNX Runtime |
| `mistral-7b-onnx` | Mistral 7B Instruct v0.2 | ONNX Runtime |
| `phi-4-onnx` | Phi 4 (14B) | ONNX Runtime |

Vision models run on OpenVINO GenAI only.

### Importing from Hugging Face

Paste a repo id or URL into **Settings → Model → Import from Hugging Face**, or:

```bash
curl -X POST localhost:9100/models/import \
  -H 'content-type: application/json' \
  -d '{"repo": "OpenVINO/Qwen3-8B-int4-ov"}'
```

Keylane reads the repo's file listing first and refuses anything it could not
load, so a 15 GB download never starts on a guess:

- `openvino_model.xml` → OpenVINO GenAI
- `genai_config.json` → ONNX Runtime GenAI
- neither → rejected with the reason (a PyTorch or GGUF repo needs converting first)

ONNX repos usually ship four or five builds of the same model — `cpu-int4`,
`cuda-fp16`, `directml`, `qnn` — so the import ranks them and takes the one that
runs on Intel hardware. Pass `"subfolder"` to override the choice. Imported
models sit alongside the curated ones and can be removed with **Forget**
(`DELETE /models/imported/{id}`), which keeps the downloaded files.

## Web search

Recommended: self-hosted SearXNG:

```bash
cd deploy && podman compose up -d
```

Keylane uses an agentic research pipeline: query planning → search (SearXNG with
optional DDGS fallback) → BM25 pre-rank → page extraction → evidence compression
→ synthesis. Sources travel beside the answer rather than inside it, so the model
writes no citation markers and the HUD renders attribution.

Test connectivity in **Settings → Web → Test SearXNG**, or:

```bash
curl -s http://127.0.0.1:9100/research/health | jq
```

Search results and fetched pages are framed as untrusted data on every result.
Outbound fetches are checked before each hop: HTTP(S) only, no credentials in the
URL, and any host resolving to a non-public address is refused — Keylane's own
API is on `127.0.0.1:9100`, so this is not hypothetical.

### Troubleshooting web search

1. **No results** — ensure SearXNG is running (`podman ps`) or switch backend to `ddgs` in Settings
2. **Thin / empty pages** — enable Playwright fetch in Settings if you have a sidecar at `playwright_url`
3. **Wrong answers** — try **thorough** depth by asking explicitly; check NPU model is loaded (`curl /health`)

## Memory and background work

Keylane keeps two kinds of memory:

- **`data/memory/USER.md`** — a profile you write by hand, always in context.
- **A fact store in SQLite** — individual things Keylane learned, added one at a time with
  the `remember` tool and searched with `recall`. Facts are deduplicated, and the most
  recent ones ride in the `<session_context>` block beside the conversation.

Inspect or edit them from the API:

```bash
curl -s http://127.0.0.1:9100/memories | jq
curl -X DELETE http://127.0.0.1:9100/memories/<id>
```

Reminders and watchers are written to SQLite before they are armed, and replayed when the
daemon starts — so a reminder set this morning still fires after a reboot. A one-shot missed
while the machine was off fires late (once, marked as missed) if it was due within 12 hours.

Longer work runs as a **background job** with an id, so it can be listed, read,
and stopped (`job_list`, `job_output`, `job_kill`) — and cannot nest without end.
A **subagent** delegates a self-contained task to a child agent with a restricted
tool set on the `background` route, and returns only its result.

```bash
curl -X POST http://127.0.0.1:9100/tasks/reminder \
  -H 'content-type: application/json' \
  -d '{"text": "call the dentist", "when": "tomorrow at 9am"}'
curl -s http://127.0.0.1:9100/tasks | jq
```

Recurring checks (`watch_create`) are opt-in: Keylane calls `ask_user` and waits for you to
agree before setting it up. Results from background work land in the inbox as well as a
desktop notification, so nothing is lost if you miss the popup.

## Skills

Skills are reusable instructions kept out of the prompt until needed. The catalog
shows names and descriptions; the body arrives only when something asks for it.

Roots are searched in rank order — `.keylane/skills` in the project, `data/skills`
for your own, then the bundled `skills/` — and a skill may be a `<name>/SKILL.md`
bundle or a flat `<name>.md`. Names are kebab-case. Frontmatter controls
invocation:

| Field | Effect |
| --- | --- |
| `description` | the one line shown in the catalog |
| `when-to-use` | optional extra routing hint |
| `enabled: false` | off entirely |
| `disable-model-invocation: true` | you may invoke it; the model may not |
| `user-invocable: false` | the model may load it; you cannot |

Typing `/skill-name` in the Spotlight bar injects that skill's instructions
directly — the only way to reach one the model is not allowed to load.

## Security

- **Outbound fetches** are validated per redirect hop; non-public addresses are
  refused, so a page the model just read cannot redirect it at your LAN.
- **Shell commands** are checked by argument, not just by name. Every file
  argument must resolve inside `security.shell_read_roots` (the Keylane checkout
  by default), and flags that read arbitrary files — `grep -f` — are refused.
- **Writing a skill file** goes through the permission gate, and Keylane never
  writes one on its own initiative unless `auto_learn_skills` is on.

## Display backend

Keylane prefers **wlr-layer-shell** (Sway, Hyprland, river): a layer surface is genuinely
always-on-top and anchors to a screen edge.

**GNOME/Mutter has never implemented wlr-layer-shell**, so there Keylane restarts itself on
**XWayland** and uses small floating windows with `_NET_WM_STATE_ABOVE` instead — the only
way a client can stay on top and place itself on that desktop. This needs `wmctrl`:

```bash
sudo dnf install wmctrl
```

Override the choice with `KEYLANE_BACKEND=layer` or `KEYLANE_BACKEND=x11` if you need to.

## Themes

Keylane ships three themes, each with a light and a dark scheme:

| Theme | Looks like |
| --- | --- |
| `glass-console` | Dark glass over the desktop, one cyan accent, hairline structure. The default. |
| `paper-terminal` | Ink on warm paper: flat surfaces, hairline rules, serif answers, monospace labels. |
| `aurora` | Translucent material, no borders, large radii, violet and cyan light. |

Pick one in **Settings → General → Appearance**, or from the CLI:

```bash
keylane-settings theme list
keylane-settings theme use paper-terminal
```

### Writing your own

GTK CSS has no variables, so a theme is a table of tokens that fills
`ui/spotlight.css.in` — the stylesheet template. Both schemes render into one
sheet, so switching light/dark stays a class swap with no reload.

```bash
keylane-settings theme new midnight --from aurora
```

That writes `data/themes/midnight.toml` with every token spelled out at its
inherited value. Delete the lines you do not want to change — `extends` fills
in the rest — then `keylane-settings theme use midnight`. A file with no
`extends` inherits `glass-console`, so the shortest useful theme is:

```toml
[theme]
name = "Amber"

[dark]
pill-text = "#ffd479"
pill-bg = "rgba(190, 140, 30, 0.26)"
orb-accent = "#ffb648"
```

Tokens come in three tables:

| Table | Holds | Examples |
| --- | --- | --- |
| `[common]` | Shape and type, shared by both schemes | `radius-panel`, `radius-control`, `font-ui`, `font-answer`, `font-label`, `font-mono` |
| `[light]`, `[dark]` | Every colour, per scheme | `panel-bg`, `panel-shadow`, `hud-*` (the answer panel), `entry-*`, `badge-*`, `btn-*` |

A few tokens hold a whole CSS value rather than a colour: `panel-shadow`,
`hud-shadow`, `shell-shadow`, `entry-focus-ring`, `control-shadow`,
`segment-checked-shadow` and `progress-fill`. The `orb-*` tokens must be plain
hex — the working orb paints itself in Cairo, so it reads them directly rather
than through CSS.

`ui/themes/glass-console.toml` lists every token with a comment per group; it
is the file to read when you want to know what something controls. A theme that
is missing a token, or names a base that does not exist, is skipped with a
warning rather than leaving the window unstyled.

## MCP servers

Add them in **Settings → MCP** — pick the transport, then a command with its arguments, or a URL with its token — or edit `config/mcp.toml` directly. Both transports are supported:

```toml
# stdio
[[servers]]
id = "fs"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/home/you"]

# Streamable HTTP — Mailspring's built-in server gives Keylane mail and calendar.
# Enable it in Mailspring: Preferences → MCP Server, then paste the token.
# A bare UUID is fine; Keylane adds the "Bearer " scheme itself.
[[servers]]
id = "mailspring"
transport = "http"
url = "http://127.0.0.1:2587/mcp"
auth_header = ""
```

Servers added in Settings are stored in `data/settings.json` and merged over the TOML by id. MCP tools appear as `mcp.<id>.<tool_name>`. Disable individual tools via Settings (stored in `data/settings.json` under `mcp.disabled_tools`).

> The client lives in `mcpbridge/`, **not** `mcp/` — a local package called `mcp` shadows the
> official SDK on `sys.path` and silently breaks every MCP server.

## API highlights

| Endpoint | Purpose |
| --- | --- |
| `POST /chat/stream` | SSE agent run with status, tool, research, token, permission events |
| `GET/PATCH /settings` | Read / write user settings |
| `GET /settings/health` | NPU, SearXNG, MCP health |
| `GET /models` · `GET /runtimes` | Catalog (filter with `?runtime=`) and installed runtimes |
| `POST /models/import` · `DELETE /models/imported/{id}` | Add or forget a Hugging Face model |
| `GET /sessions` | Session history for UI |
| `GET /tasks` · `POST /tasks/reminder` · `DELETE /tasks/{id}` | Reminders, watchers, background jobs |
| `GET/POST /memories` · `DELETE /memories/{id}` | The fact store |
| `GET /inbox` · `POST /inbox/read` | Results from background work |
| `POST /permissions/respond` | Approve/deny tool permission prompts |

## Architecture

```text
Super+Space → GTK Spotlight → floating orb → answer HUD (click-through)
                    ↓
              FastAPI daemon (:9100)
                    ↓
         AIAgent (ReAct + tools + MCP)
                    ↓
              capability seams (seams/)
      llm · web · skills · jobs · subagents · goals · spill
                    ↓
   NPU via runtimes/  ·  optional GPU model over the OpenAI API
   (OpenVINO GenAI · ONNX Runtime GenAI + OpenVINO EP)
```

Every capability is reached through a registry in `seams/`, not by importing one
implementation: an interface, one or more providers, and a consumer (usually the
model-facing tool). That is what lets a provider be swapped, restricted per
agent, or stubbed in a test without every call site knowing.

### Model routes

Call sites ask for a *purpose*, not a model, and `config/models.toml` maps each
route to the adapters to try in order:

| Route | Serves | Default |
| --- | --- | --- |
| `interactive` | the turn the HUD is waiting on | `npu` |
| `background` | subagents, scheduled work, research synthesis | `gpu` → `npu` |
| `utility` | query planning, URL selection | `npu` → `gpu` |

The `gpu` adapter speaks the OpenAI chat-completions API, so LM Studio,
llama.cpp's server, Ollama and vLLM all work. Set its `model`, flip `enabled`,
and background work moves off the NPU without touching any call site.

### Runtimes

`runtimes/` is the same idea one level down: an interface (`RuntimeBackend`) for
recognising an export on disk, validating its download, compiling it, budgeting
a prompt and streaming tokens, with one module per stack behind it. A catalog
entry names its runtime and everything else is asked of that runtime, so adding
a third is a new module rather than a new branch in every function that touches
a model.

### The system prompt

The prompt is assembled from registered sections rather than written as one
string. Each capability contributes its own paragraph next to its tool
registration, so disabling a tool removes its guidance too and the prompt cannot
promise something that is not there.

Facts that change every turn — the clock, the memory digest, the skill catalog,
the todo list, the current goal — are **not** in the system message. They are
appended as one `<session_context>` block and re-emitted only when their content
changes, which keeps the system prefix byte-identical across turns.

## License

MIT — see [LICENSE](LICENSE).
