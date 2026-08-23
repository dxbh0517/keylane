# Plugin system

**Keylane** treats workers and integrations as **plugins**. Plugins can be:

| Kind | Purpose | Examples |
| --- | --- | --- |
| `native` | Python code in-process | LM Studio, Claude Code, Cursor CLI, Lemonade |
| `mcp` | External MCP server over **stdio** | ComfyUI via `comfy-mcp` |
| `utility` | Health/config only (not routed as a worker) | Lemonade clipboard daemon |

ComfyUI defaults to the **MCP** plugin (`comfy-mcp`), not a hand-rolled HTTP client. The older HTTP plugin remains available as `comfyui-http` (disabled by default).

## Control panel

Open `http://127.0.0.1:9100/` → **Plugins** to enable/disable, edit settings, install MCP plugins, or remove community ones.

State is stored in `config/plugins.toml`.

## MCP plugins (recommended for new tools)

If a tool already speaks [MCP](https://modelcontextprotocol.io), wrap it instead of writing a custom worker.

### Install from the UI

**Plugins → Install MCP plugin**

- **ID** — unique slug (`comfyui`, `browser`, …)
- **Command** — executable on `PATH` (`comfy-mcp`)
- **Health tool** — e.g. `server_info`
- **Run tool** — e.g. `generate_image` / `run_workflow`
- **Worker ID** — router name (same as ID unless you share a worker slot)

### Install from a manifest

Drop a folder under `plugins/community/<id>/plugin.toml`:

```toml
id = "comfyui"
name = "ComfyUI (MCP)"
kind = "mcp"
author = "Comfy-Org"
version = "0.1.0"
homepage = "https://docs.comfy.org/agent-tools/mcp"
worker_id = "comfyui"
cloud = false
command = "comfy-mcp"
args = []
health_tool = "server_info"
run_tool = "generate_image"
```

Then call `POST /api/plugins/reload` or restart the gateway.

### API

```http
GET    /api/plugins
POST   /api/plugins/{id}/enable          {"enabled": true}
PUT    /api/plugins/{id}/settings        {...}
POST   /api/plugins/install/mcp          {id, command, ...}
DELETE /api/plugins/{id}
POST   /api/plugins/reload
```

### How MCP execution works

1. Gateway starts the MCP command as a **stdio** subprocess (`mcp` Python SDK).
2. Health checks call `health_tool` (fallback: `list_tools`).
3. Routed tasks call `run_tool` with arguments derived from the route decision (`prompt`, dimensions, workflow, …).
4. Evidence (paths, text) is collected for the NPU verifier.

Dependency: `pip install mcp` (already listed in `requirements.txt`).

### Official ComfyUI MCP

You already have `/home/emul/.local/bin/comfy-mcp`. Ensure ComfyUI is running, then keep the built-in `comfyui` plugin enabled:

```toml
[plugins.comfyui]
enabled = true

[plugins.comfyui.settings]
command = "comfy-mcp"
health_tool = "server_info"
run_tool = "generate_image"
```

Optional env (JSON object in settings) can pass `COMFYUI_URL` / related vars — see Comfy docs.

## Native plugins

Built-ins live in `app/plugins/builtin.py` and implement `BasePlugin` (`app/plugins/base.py`):

```python
class BasePlugin(ABC):
    id: str
    name: str
    kind: PluginKind
    worker_id: str | None   # None => utility only
    cloud: bool

    def settings_schema(self) -> list[SettingField]: ...
    async def health(self) -> PluginHealth: ...
    async def run(self, decision: RouteDecision) -> WorkerResult: ...  # workers only
```

Register new built-ins in `create_builtin_plugins()`.

## Routing rules

- Only **enabled** plugins with a `worker_id` are routing candidates.
- `local_only` blocks plugins with `cloud = true` (Claude, Cursor).
- If both MCP and native plugins share a `worker_id` (e.g. `comfyui` + `comfyui-http`), **MCP wins**.

## Security notes

- MCP commands run as your user. Treat community MCP plugins like any local CLI.
- Do not point MCP plugins at untrusted remote binaries.
- Gateway remains bound to `127.0.0.1` by default.
