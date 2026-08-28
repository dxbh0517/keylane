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

- **Super+Space** opens a Spotlight-style bar that morphs into a corner progress widget while working
- A **curated ~7–9B OpenVINO model** runs **always-on** on your **Intel NPU**
- **Settings UI** (gear icon or `Ctrl+,`) — model picker, web search backend, speech, security, MCP status
- **Agentic web research** — pluggable search (SearXNG / DDGS fallback), evidence compression, cited answers
- **Todo list**, **background tasks**, **scheduled jobs**, and **proactive notifications**
- **Hermes-inspired** memory (`USER.md`, `MEMORY.md`), skills, and cross-session recall (`Ctrl+H`)
- **Voice input** (mic button) via Whisper
- **MCP servers** — persistent sessions, configurable in `config/mcp.toml`
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
| General | Assistant name, iteration budget |
| Model | Active NPU model, switch / download |
| Web | Search backend (searxng / ddgs), SearXNG URL, Playwright, fallback |
| Speech | TTS on notify, read aloud, test buttons |
| Security | Shell allowlist, shell permission mode (auto / ask / deny) |
| MCP | Configured servers (edit `config/mcp.toml`) |

CLI:

```bash
PYTHONPATH=. python scripts/keylane-settings list
PYTHONPATH=. python scripts/keylane-settings set research.search_backend ddgs
PYTHONPATH=. python scripts/keylane-settings wizard web
```

Settings persist to `data/settings.json` (merged over `config/*.toml` defaults).

## Model catalog

Pick a model in **Settings → Model** or via API `POST /models/select`. Models auto-download from Hugging Face:

| ID | Model |
| --- | --- |
| `qwen2.5-7b-instruct` | Qwen 2.5 7B Instruct (default) |
| `qwen2.5-7b-npu` | Qwen 2.5 7B NPU-tested |
| `qwen2.5-coder-7b` | Qwen 2.5 Coder 7B |
| `qwen3.5-9b` | Qwen 3.5 9B |
| `gemma-2-9b-it` | Gemma 2 9B Instruct |

## Web search

Recommended: self-hosted SearXNG:

```bash
cd deploy && podman compose up -d
```

Keylane uses an agentic research pipeline: query planning → search (SearXNG with optional DDGS fallback) → BM25 pre-rank → page extraction → evidence compression → synthesis with `[1]` citations.

Test connectivity in **Settings → Web → Test SearXNG**, or:

```bash
curl -s http://127.0.0.1:9100/research/health | jq
```

### Troubleshooting web search

1. **No results** — ensure SearXNG is running (`podman ps`) or switch backend to `ddgs` in Settings
2. **Thin / empty pages** — enable Playwright fetch in Settings if you have a sidecar at `playwright_url`
3. **Wrong answers** — try **thorough** depth by asking explicitly; check NPU model is loaded (`curl /health`)

## MCP servers

Edit `config/mcp.toml`:

```toml
[[servers]]
id = "fs"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/home/you"]
```

MCP tools appear as `mcp.<id>.<tool_name>`. Disable individual tools via Settings (stored in `data/settings.json` under `mcp.disabled_tools`).

## API highlights

| Endpoint | Purpose |
| --- | --- |
| `POST /chat/stream` | SSE agent run with status, tool, research, token, permission events |
| `GET/PATCH /settings` | Read / write user settings |
| `GET /settings/health` | NPU, SearXNG, MCP health |
| `GET /sessions` | Session history for UI |
| `POST /permissions/respond` | Approve/deny tool permission prompts |

## Architecture

```text
Super+Space → GTK Spotlight → corner progress widget
                    ↓
              FastAPI daemon (:9100)
                    ↓
         AIAgent (ReAct + tools + MCP)
                    ↓
    research providers (searxng / ddgs / local extract)
                    ↓
           OpenVINO GenAI on NPU
```

## License

MIT — see [LICENSE](LICENSE).
