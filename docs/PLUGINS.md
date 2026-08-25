# Writing plugins

Everything Keylane can talk to is a plugin: Claude Code, Cursor, ComfyUI, LM
Studio, Lemonade, and anything you add. A plugin can contribute any combination
of three things, and nothing else in the system needs to know which:

| Contribution | What it means | Example |
| --- | --- | --- |
| **Worker** | Can be routed a whole task (`worker_id` set) | Claude Code takes a coding job |
| **Tools** | Individual capabilities the assistant may call | `comfyui.generate_image` |
| **Skills** | Instruction packs added to the assistant's prompt | "always deploy with `make release`" |

Plugins can be enabled, disabled, configured, installed and removed while the
gateway is running. Enabling one immediately changes what the assistant can do —
no restart.

## Nothing is installed by default

A fresh Keylane ships with **no plugins installed**. It talks to nothing —
no local model, no cloud agent, no image server — until you say so.

Everything Keylane ships with lives in [`plugins/catalog/`](../plugins/catalog),
one folder per plugin, and is installed on request:

**Control panel → Plugins → Catalog**, or:

```bash
curl -s     http://127.0.0.1:9100/api/plugins/catalog
curl -X POST http://127.0.0.1:9100/api/plugins/catalog/lmstudio/install
curl -X DELETE http://127.0.0.1:9100/api/plugins/catalog/lmstudio          # keeps settings
curl -X DELETE 'http://127.0.0.1:9100/api/plugins/catalog/lmstudio?purge=true'
```

Installing records the plugin in `config/plugins.toml` and registers it
immediately — the assistant's tool list changes without a restart. Removing it
unregisters it and keeps your settings unless you purge.

| Catalog entry | Worker | Reaches |
| --- | --- | --- |
| `lmstudio` | `lmstudio` | Your machine |
| `lemonade` | `lemonade` | Your machine |
| `comfyui` | `comfyui` | Your machine (MCP) |
| `comfyui-http` | `comfyui` | Your machine (legacy HTTP) |
| `mailspring` | — (tools only) | Your machine (Mailspring MCP) |
| `gnome-calendar` | — (tools only) | Fedora / GNOME Calendar (Evolution Data Server) |
| `caldav` | — (tools only) | Remote CalDAV (Nextcloud, Fastmail, …) |
| `claude` | `claude` | **Anthropic** |
| `cursor` | `cursor` | **Cursor** |

> **Note**: Upgrading keeps what you had. Anything already recorded in
> `config/plugins.toml` is treated as installed, so an existing setup is not
> emptied out by this change.

### Adding your own to the catalog

Drop a folder into `plugins/catalog/<id>/plugin.toml`:

```toml
id = "ollama"
name = "Ollama"
kind = "native"
description = "Local models served by Ollama."
homepage = "https://ollama.com"
worker_id = "ollama"
cloud = false
tags = "worker, local, chat"

entry = "plugin.py"        # or "builtin:<id>" for an implementation Keylane ships
```

`entry = "builtin:<id>"` points at a class already inside Keylane. Anything
else is loaded from the folder — a `plugin.py`, or an MCP `command`.

## The four kinds

| Kind | Runs as | Use it when |
| --- | --- | --- |
| `mcp` | External process speaking [MCP](https://modelcontextprotocol.io) over stdio | The tool already speaks MCP — **start here** |
| `native` | Python in the gateway process | You need a worker with custom evidence collection |
| `tool` | Python, tools only, no worker | You are adding capabilities, not a destination for tasks |
| `utility` | Python, health and config only | A daemon you want to monitor but never route to |

## Quick start: wrap an MCP server

If your tool speaks MCP, you are already done. Give Keylane the command and it
discovers the rest.

**Control panel → Plugins → Install an MCP plugin**

| Field | Meaning |
| --- | --- |
| ID | Unique slug (`browser`, `notion`, `home-assistant`) |
| Command | Executable on `PATH` |
| Args | JSON array of CLI arguments |
| Health tool | A cheap tool used as a ping — often `server_info` or `list_*` |
| Run tool | Which tool a routed *task* calls; leave the default if this is tools-only |
| Worker ID | Set it to make the plugin a routing destination; leave blank for tools only |

Every tool the server exposes then appears under **Assistant → Tools**,
namespaced as `<plugin-id>.<tool-name>`, and the assistant can call it.

### Or drop in a manifest

Create `plugins/community/<id>/plugin.toml`:

```toml
id = "browser"
name = "Browser control"
kind = "mcp"
author = "you"
version = "0.1.0"
homepage = "https://example.com/browser-mcp"
worker_id = ""            # blank = contributes tools but is never routed a task
cloud = false             # true = blocked in local-only mode
command = "browser-mcp"
args = []
health_tool = "list_tabs"
run_tool = "navigate"
env = { BROWSER_PROFILE = "default" }
```

Then `POST /api/plugins/reload`, or press **Reload** on the Plugins tab.

> **Note**: `cloud = true` marks a plugin as talking to a remote service. Local-only
> mode then refuses to route to it and hides its worker from the assistant.

## Writing a Python plugin

Use this when MCP is not a good fit — for example a worker that needs to collect
git evidence, or tools that need in-process access to gateway state.

Create `plugins/community/<id>/plugin.toml`:

```toml
id = "notes"
name = "Notes"
kind = "tool"
version = "0.1.0"
author = "you"
entry = "plugin.py"       # optional, this is the default
```

And `plugins/community/<id>/plugin.py`:

```python
from app.plugins.base import BasePlugin, PluginHealth, PluginKind, SettingField, SettingType
from app.tools.base import BaseTool, ToolDanger, ToolResult, object_schema, string_prop


class AppendNoteTool(BaseTool):
    name = "append_note"
    description = "Append a line to the user's daily note file."
    danger = ToolDanger.SENSITIVE      # so it is confirmation-gated
    category = "notes"

    def __init__(self, plugin):
        self._plugin = plugin

    def parameters(self):
        return object_schema(
            {"text": string_prop("The line to append.")},
            required=["text"],
        )

    async def run(self, args):
        from datetime import date
        from pathlib import Path

        folder = Path(self._plugin.settings["folder"]).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{date.today():%Y-%m-%d}.md"
        with target.open("a", encoding="utf-8") as fh:
            fh.write(str(args.get("text", "")) + "\n")
        return ToolResult.success(f"Added a line to {target}.")


class Plugin(BasePlugin):
    id = "notes"
    name = "Notes"
    description = "Append lines to a daily markdown note."
    kind = PluginKind.TOOL
    worker_id = None                    # not a routing destination
    cloud = False

    def settings_schema(self):
        return [
            SettingField(
                key="folder",
                label="Notes folder",
                type=SettingType.PATH,
                default="~/Documents/notes",
                required=True,
            )
        ]

    async def health(self):
        from pathlib import Path

        folder = Path(self.settings.get("folder", "")).expanduser()
        ok = bool(self.settings.get("folder"))
        return PluginHealth(
            ok=ok,
            detail=f"Writing to {folder}" if ok else "No notes folder configured",
        )

    def tools(self):
        return [AppendNoteTool(self)]
```

The loader accepts either a `Plugin` class deriving from `BasePlugin`, or a
`create_plugin(settings) -> BasePlugin` factory if you need more control.

> **Warning**: A Python plugin runs inside the gateway process with your user's
> permissions. Treat community Python plugins exactly like any other script you
> would run on your machine — read it first.

## The `BasePlugin` contract

```python
class BasePlugin(ABC):
    id: str                       # unique slug
    name: str                     # shown in the control panel
    description: str
    kind: PluginKind              # NATIVE | MCP | TOOL | SKILL | UTILITY
    worker_id: str | None         # routing destination, or None
    cloud: bool                   # blocked in local-only mode
    removable: bool               # can be uninstalled from the UI
    homepage: str | None

    def settings_schema(self) -> list[SettingField]: ...
    def default_settings(self) -> dict: ...
    def update_settings(self, data: dict) -> dict: ...

    async def health(self) -> PluginHealth: ...          # required
    async def run(self, decision) -> WorkerResult: ...   # required for workers
    def tools(self) -> list[BaseTool]: ...               # optional
    def skills(self) -> list[Skill]: ...                 # optional
    def mcp_descriptor(self) -> dict | None: ...         # MCP plugins only
    async def close(self) -> None: ...
```

### Setting fields

`SettingField` renders a form in the control panel. Types: `string`, `integer`,
`boolean`, `path`, `select` (with `options`), `secret` (masked), `json`.

```python
SettingField(
    key="api_base",
    label="API base URL",
    type=SettingType.STRING,
    default="http://127.0.0.1:8080",
    description="Shown as help text under the field.",
    required=True,
)
```

Settings persist to `config/plugins.toml` and are handed back to your plugin on
the next load.

## Writing a worker plugin

A worker receives a whole `RouteDecision` and returns a `WorkerResult`. The
`WorkerEvidence` you attach is what the verifier judges, so populate it
honestly — an empty `changed_files` on a `modify_project` action is how Keylane
notices a worker claimed success without doing anything.

```python
class Plugin(BasePlugin):
    id = "ollama"
    name = "Ollama"
    kind = PluginKind.NATIVE
    worker_id = "ollama"
    cloud = False

    async def health(self):
        import httpx
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get("http://127.0.0.1:11434/api/tags")
                return PluginHealth(ok=r.status_code < 500, detail="Ollama reachable")
        except Exception as exc:
            return PluginHealth(ok=False, detail=str(exc))

    async def run(self, decision):
        import httpx
        from app.schemas import WorkerEvidence, WorkerResult

        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": decision.model or "llama3", "prompt": decision.instruction, "stream": False},
            )
            r.raise_for_status()
            answer = r.json()["response"]

        return WorkerResult(
            success=True,
            summary=answer,
            evidence=WorkerEvidence(
                worker="ollama",
                action=decision.action,
                response=answer,
                stdout=answer,
                exit_code=0,
            ),
        )
```

Registering a `worker_id` makes it a valid routing target automatically — the
router prompt and the assistant's `list_workers` tool both pick it up.

## Contributing skills

Return `Skill` objects from `skills()` to ship prompt guidance with your plugin:

```python
from app.skills import Skill

def skills(self):
    return [
        Skill(
            name="ollama-models",
            description="Which Ollama model to pick.",
            triggers=["ollama", "local model"],
            content="When delegating to Ollama, prefer llama3 for prose and "
                    "qwen2.5-coder for code.",
        )
    ]
```

See [Skills](SKILLS.md) for the file-based equivalent.

## Routing rules

- Only **enabled** plugins with a `worker_id` can be routed a task.
- `local_only` mode blocks every plugin with `cloud = true`.
- When two plugins share a `worker_id` (`comfyui` MCP and `comfyui-http`), the
  **MCP one wins**.
- A tool name that collides with a built-in is prefixed with the plugin id;
  MCP tools are always namespaced `<plugin-id>.<tool-name>`.

## HTTP API

```http
GET    /api/plugins?health=true
POST   /api/plugins/{id}/enable        {"enabled": true}
PUT    /api/plugins/{id}/settings      {...}
POST   /api/plugins/install/mcp        {id, command, args, ...}
DELETE /api/plugins/{id}
POST   /api/plugins/reload
POST   /api/tools/refresh
```

## Built-in plugins

| ID | Kind | Worker | Notes |
| --- | --- | --- | --- |
| `lmstudio` | native | `lmstudio` | OpenAI-compatible local LLM |
| `lemonade` | native | `lemonade` | Lemonade Server, port 13305 |
| `claude` | native (cloud) | `claude` | Claude Code CLI, needs a project |
| `cursor` | native (cloud) | `cursor` | Cursor Agent CLI, needs a project |
| `comfyui` | mcp | `comfyui` | Official `comfy-mcp`; exposes ~39 tools |
| `comfyui-http` | native | `comfyui` | Legacy direct HTTP client, off by default |

Built-ins live in `app/plugins/builtin.py` and are registered in
`create_builtin_plugins()`.

### The official ComfyUI MCP

Following [Local Comfy MCP Connection](https://docs.comfy.org/agent-tools/mcp#local-comfy-mcp-connection):

1. `pip install "comfy-cli>=1.14.0" comfy-mcp`
2. Set up a workspace (`comfy install` / `comfy set-default`) and start it (`comfy launch`)
3. Point Keylane at the stdio server, setting `COMFY_BIN` when the service `PATH`
   does not include it:

```toml
[plugins.comfyui]
enabled = true

[plugins.comfyui.settings]
command = "comfy-mcp"
health_tool = "server_info"
run_tool = "generate_image"
env = '{"COMFY_BIN": "/home/YOU/.local/bin/comfy"}'
```

Keylane auto-fills `COMFY_BIN` when it can find `comfy`, and
`ai-gateway.service` sets `PATH` and `COMFY_BIN` for user installs under
`~/.local/bin`.

## Security notes

- MCP commands and Python plugins run as your user, with your permissions.
- Do not point a plugin at a binary you did not install yourself.
- The gateway stays bound to `127.0.0.1`; a plugin should not change that.
- Mark anything that reaches the network `cloud = true` so local-only mode works.
