<p align="center">
  <img src="assets/logo.png" alt="Keylane logo" width="128" height="128" />
</p>

<h1 align="center">Keylane</h1>

<p align="center">
  <strong>Super-key AI gateway for Fedora</strong><br />
  Route desktop prompts through an Intel NPU control plane to local and cloud workers.
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#control-panel">Control panel</a> ·
  <a href="#plugins--mcp">Plugins</a> ·
  <a href="#themes">Themes</a> ·
  <a href="#api">API</a> ·
  <a href="#npu-status">NPU</a>
</p>

---

## What is Keylane?

**Keylane** is a local-first desktop AI gateway for Fedora Linux. Press **Super+Space**, type or dictate an instruction, and a small OpenVINO model on the Intel Core Ultra **NPU** classifies the request and routes it to the right worker:

| Worker | Role |
|--------|------|
| **LM Studio** | General local LLM answers |
| **Claude Code** | Advanced coding / repo work (cloud) |
| **Cursor CLI** | Coding / repo work (cloud) |
| **ComfyUI** | Image generation via MCP (`comfy-mcp`) |
| **Lemonade** | Optional local OpenAI-compatible server |

A separate **NPU verifier** checks worker evidence and can retry failed tasks. The gateway—not the LLM—is the authority: models emit structured intents; Python validates and executes them.

Design specification: [`fedora_local_ai_gateway.md`](fedora_local_ai_gateway.md).

```text
Super+Space
    │
    ▼
GTK4 / libadwaita launcher  (Keylane popup)
    │  text · microphone
    ▼
FastAPI gateway  127.0.0.1:9100
    │
    ▼
OpenVINO NPU router  (fallback: heuristic on CPU)
    ├── LM Studio
    ├── Claude Code
    ├── Cursor CLI
    ├── ComfyUI (MCP)
    └── …plugins
              │
              ▼
         Task evidence
              │
              ▼
      OpenVINO NPU verifier
              │
         success / retry
```

## Features

- **Super-key launcher** — GTK4 + libadwaita popup, themable, meant for Super+Space
- **NPU control plane** — intent routing + verification when OpenVINO exposes `NPU`
- **Plugin system** — native workers and **MCP** servers (install from the UI)
- **Themes** — shared styles for the web control panel and the GTK popup
- **Local-only mode** — hard-block cloud workers at the gateway
- **Project sandbox** — allowed roots + confirmation for modifying actions
- **systemd user services** — gateway + optional always-on launcher
- **OpenAI-compatible** entrypoint at `/v1/chat/completions`

## Requirements

- Fedora (GTK4 / libadwaita)
- Python 3.12+ recommended
- Intel Core Ultra (or similar) with `intel_vpu` for the NPU path — CPU heuristics work without it
- Optional workers: LM Studio, Claude Code CLI, Cursor agent CLI, ComfyUI + `comfy-mcp`, Lemonade

## Quick start

### 1. System packages

```bash
sudo dnf install -y \
  git python3 python3-pip python3-devel \
  gcc gcc-c++ make pkg-config \
  gtk4-devel libadwaita-devel \
  gobject-introspection-devel cairo-gobject-devel \
  python3-gobject python3-gobject-base \
  gtk4 libadwaita \
  portaudio-devel ffmpeg
```

> Do **not** `pip install PyGObject`. Use Fedora’s `python3-gobject` and create the venv with `--system-site-packages`.

### 2. Dev environment

```bash
git clone https://github.com/dxbh0517/keylane.git
cd keylane
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run

```bash
# Terminal A — gateway (9100; 9000 is often taken by Lemonade’s lemond)
uvicorn app.main:app --host 127.0.0.1 --port 9100

# Terminal B — launcher
python launcher/main.py
```

Control panel: [http://127.0.0.1:9100/](http://127.0.0.1:9100/)

### 4. Install as a user service

```bash
chmod +x scripts/install.sh
./scripts/install.sh
systemctl --user enable --now ai-launcher.service
```

Installs into `~/.local/share/ai-gateway` and enables `ai-gateway.service`.

**GNOME shortcut:** Settings → Keyboard → Custom Shortcuts

| Field | Value |
|-------|--------|
| Name | Keylane |
| Command | `~/.local/share/ai-gateway/.venv/bin/python ~/.local/share/ai-gateway/launcher/main.py` |
| Shortcut | Super+Space |

## Architecture

| Layer | Path | Responsibility |
|-------|------|----------------|
| Launcher | `launcher/` | Capture prompt / mic; call gateway |
| Gateway | `app/main.py` | FastAPI, localhost only |
| Orchestrator | `app/orchestrator.py` | Route → execute → verify → retry |
| Router / verifier | `app/npu/` | OpenVINO GenAI or heuristic fallback |
| Plugins | `app/plugins/` | Native + MCP workers |
| Themes | `app/themes.py`, `themes/` | Web + GTK CSS |
| Control panel | `web/` | Status, config, plugins, themes |
| Config | `config/*.toml` | Gateway, workers, plugins, projects, themes |

**Principles**

1. **NPU = control plane** — small, frequent classification/verification — not the main coding or Flux engine
2. **Gateway = authority** — structured intents are validated before any worker runs
3. **Verifier ≠ executor** — success is judged from evidence, not worker self-report alone

## Control panel

[http://127.0.0.1:9100/](http://127.0.0.1:9100/)

- **Status** — NPU (OpenVINO vs driver-only), LM Studio, ComfyUI, Claude, Cursor, Lemonade
- **Gateway** — host/port, retries, local-only, project roots, confirmation policy
- **Plugins** — enable/disable, settings, install MCP servers
- **Themes** — activate built-ins or install community zips (web **and** Super+Space popup)

## Configuration

Primary file: `config/workers.toml`

```toml
[gateway]
host = "127.0.0.1"
port = 9100
max_retries = 3
local_only = false

[npu]
model_path = "./models/router"
device = "NPU"
fallback_device = "CPU"

[security]
allowed_project_roots = ["/home/you/Documents/Code"]
require_confirmation_for_modifications = true
```

Also:

- `config/projects.toml` — launcher project dropdown
- `config/plugins.toml` — plugin enablement / MCP settings
- `config/themes.toml` — active theme id

## Plugins & MCP

Guide: [`docs/PLUGINS.md`](docs/PLUGINS.md)

| Kind | Purpose |
|------|---------|
| `native` | In-process Python workers (LM Studio, Claude, Cursor, Lemonade, …) |
| `mcp` | External MCP over stdio (ComfyUI via `comfy-mcp`) |

Install MCP plugins from **Plugins → Install MCP plugin**, or drop a package under `plugins/community/<id>/`.

ComfyUI defaults to MCP (`comfy-mcp`), not a hand-rolled HTTP client.

## Themes

Guide: [`docs/THEMES.md`](docs/THEMES.md)

Built-ins: `default`, `midnight`, `paper`.

```text
my-theme/
  theme.toml
  web.css
  launcher.css
```

Activating a theme updates `/theme.css` for the control panel and `/api/themes/active/launcher.css` for the GTK popup (re-applied every time the launcher opens).

## NPU status

Keylane separates **driver presence** from **OpenVINO NPU readiness**:

| State | Meaning |
|-------|---------|
| Online | OpenVINO lists `NPU` |
| Driver only | `/dev/accel/accel0` (or similar) exists, but OpenVINO does not expose `NPU` yet |
| Offline | No accelerator device |

Typical Fedora gap: kernel `intel_vpu` is loaded, but **Level Zero NPU userspace** (`libze_intel_npu.so.1`) from [intel/linux-npu-driver](https://github.com/intel/linux-npu-driver/releases) is missing. Install that stack, ensure your user can access the accel device (often `render` group), then re-login.

```bash
python scripts/check_npu.py
curl -s http://127.0.0.1:9100/api/status | jq '{npu,npu_driver,npu_detail,openvino_devices}'
```

Until OpenVINO lists `NPU`, the router/verifier use **CPU / heuristic** fallbacks. Place a 1B–3B OpenVINO GenAI export under `models/router` when you have one.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Control panel |
| GET | `/theme.css` | Active web theme |
| GET | `/api/status` | Worker + NPU health |
| GET/PUT | `/api/config` | Gateway settings |
| GET/POST… | `/api/plugins…` | Plugin list, enable, MCP install |
| GET/PUT… | `/api/themes…` | Theme list / activate / install |
| GET | `/api/themes/active/launcher.css` | GTK theme CSS |
| POST | `/api/chat` | Route + execute (+ confirm/retry) |
| POST | `/api/route` | Route only |
| POST | `/api/transcribe` | Voice → text |
| GET | `/api/projects` | Project list |
| GET/POST | `/api/tasks/…` | Task status / cancel / confirm |
| POST | `/v1/chat/completions` | OpenAI-compatible entry |

Interactive docs: [http://127.0.0.1:9100/docs](http://127.0.0.1:9100/docs)

## Repository layout

```text
app/           FastAPI gateway, orchestrator, NPU, plugins, workers
launcher/      GTK4 / libadwaita Super-key UI + logo
web/           Control panel + favicon
themes/        Built-in themes (web.css + launcher.css)
config/        TOML configuration
plugins/       Community / drop-in plugin packages
workflows/     Approved ComfyUI graphs
models/router/ OpenVINO GenAI model (you provide)
systemd/       User unit files
scripts/       install.sh, check_npu.py, desktop entry
assets/        Brand logo
docs/          PLUGINS.md, THEMES.md
tests/         Pytest suite
fedora_local_ai_gateway.md   Design specification
```

## Tests

```bash
source .venv/bin/activate
pytest -q
```

## Security notes

- Gateway binds **127.0.0.1 only** by default — not exposed to the LAN
- Cloud CLIs are optional and blocked in **local-only** mode
- Project paths must sit under configured allowed roots
- Destructive / modifying actions can require explicit confirmation in the launcher

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| Theme doesn’t change in the browser | Hard refresh; confirm `/theme.css` after activate |
| Theme doesn’t change in Super+Space | Re-open launcher; confirm `/api/themes/active/launcher.css` |
| NPU shows “Driver only” | Install Level Zero NPU userspace; `render` group; re-login |
| Port already in use | Lemonade often owns `9000` — Keylane defaults to `9100` |
| Launcher can’t import `gi` | Install `python3-gobject`; use `--system-site-packages` venv |
| ComfyUI offline | Start ComfyUI; ensure `comfy-mcp` is on `PATH` |

## License

MIT — see [LICENSE](LICENSE).

## Credits

Built for Fedora desktop AI workflows around OpenVINO NPU routing. Design goals are documented in `fedora_local_ai_gateway.md`.
