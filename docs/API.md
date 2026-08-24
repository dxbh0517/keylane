# HTTP API

The gateway listens on `127.0.0.1:9100` and is not reachable from the network.
An interactive OpenAPI explorer is served at
[`/api-docs`](http://127.0.0.1:9100/api-docs).

## Chat and tasks

### `POST /api/chat`

The main entry point. Runs the assistant loop, falling through to worker routing
when the assistant has nothing to contribute.

```json
{
  "message": "generate a hero image and drop it into the landing page",
  "project": "/home/you/code/site",
  "local_only": false,
  "confirmed": false,
  "task_id": null
}
```

Response:

```json
{
  "task_id": "5f069c5c2731",
  "status": "completed",
  "worker": "assistant",
  "result": "…",
  "requires_confirmation": false,
  "assistant_steps": [
    {"index": 1, "tool": "delegate_to_worker", "ok": true, "observation": "…"}
  ],
  "canvas": {"title": "…", "blocks": [{"type": "stats", "items": []}]},
  "pending_tool": null,
  "pending_arguments": {}
}
```

`status` is one of `pending`, `routing`, `waiting_confirmation`, `running`,
`verifying`, `retrying`, `completed`, `failed`, `cancelled`.

`canvas` is the structured answer — see [Canvas answers](canvas.html). `result`
carries the same content as plain text, so a client that cannot render a canvas
still has something to show.

**To approve a gated step**, resend with the same `task_id`, the same `message`,
and `"confirmed": true`.

### Other task endpoints

```http
POST /api/tasks                 alias for /api/chat
GET  /api/tasks/{id}            current state of a task
POST /api/tasks/{id}/cancel     request cancellation
POST /api/route                 route only — returns the decision, runs nothing
```

### `POST /v1/chat/completions`

A minimal OpenAI-compatible shim, so existing clients can point at Keylane. The
last user message becomes the request; the reply is the task result.

## Assistant and tools

```http
GET  /api/assistant             settings, live system prompt, model state
PUT  /api/assistant             update settings
GET  /api/tools                 every tool, with schema, danger and availability
GET  /api/tools?available_only=true
POST /api/tools/refresh         rebuild the registry, re-query MCP servers
POST /api/tools/call            invoke one tool directly
GET  /api/skills                every skill and its content
POST /api/skills                create one
PUT  /api/skills/{name}         update or rename one
POST /api/skills/{name}/enable  {"enabled": true}
DELETE /api/skills/{name}
POST /api/skills/reload         re-read skills/
POST /api/skills/discover       {"repo": "owner/repo"} -> what a GitHub repo contains
POST /api/skills/import         {"repo": "...", "paths": [...]} -> install a selection
```

`POST /api/tools/call`:

```json
{"tool": "web_search", "arguments": {"query": "openvino npu"}, "confirmed": false}
```

A gated tool answers `{"requires_confirmation": true, "danger": "sensitive", …}`
until you resend with `"confirmed": true`.

## Activity

```http
GET /api/activity     point-in-time snapshot
GET /api/events       Server-Sent Events stream
```

The snapshot is what drives the tray icon:

```json
{
  "busy": true,
  "active_count": 1,
  "needs_attention": 0,
  "active": [
    {"task_id": "ab12", "title": "refactor the client", "status": "running",
     "worker": "claude", "step": "attempt 1", "started_at": "…"}
  ],
  "recent": []
}
```

`/api/events` emits `event: snapshot` frames with that same payload, plus
`event: event` frames for individual steps, and a comment keepalive every 20
seconds.

## Status and config

```http
GET /api/status       NPU, workers, assistant, tool count, busy flag
GET /api/projects     configured project roots
GET /api/config       gateway settings
PUT /api/config       update them
GET /healthz          liveness
```

`GET /api/status`:

```json
{
  "npu": true, "npu_driver": true, "npu_openvino": true,
  "npu_detail": "OpenVINO NPU device available",
  "openvino_devices": ["CPU", "GPU", "NPU"],
  "lmstudio": true, "comfyui": true, "claude": true, "cursor": true,
  "lemonade": true, "gateway": true, "local_only": false,
  "assistant": true, "tools_enabled": true, "tool_count": 58, "busy": false,
  "plugins": {"lmstudio": true, "comfyui": true}
}
```

## Plugins

```http
GET    /api/plugins/catalog                    everything installable
POST   /api/plugins/catalog/{id}/install
DELETE /api/plugins/catalog/{id}?purge=false

GET    /api/plugins?health=true
POST   /api/plugins/{id}/enable      {"enabled": true}
PUT    /api/plugins/{id}/settings    {...}
POST   /api/plugins/install/mcp      {id, command, args, health_tool, run_tool, ...}
DELETE /api/plugins/{id}
POST   /api/plugins/reload
```

Enabling or disabling a plugin rebuilds the tool registry, so the assistant's
capabilities change immediately.

## Themes

```http
GET    /api/themes                      every theme with its popup spec
GET    /api/themes/active               active id, popup spec, colours
GET    /api/themes/active/popup.json    just the popup spec
GET    /api/themes/active/launcher.css  GTK CSS for the popup
GET    /api/themes/presets              built-in popup presets
GET    /theme.css                       control-panel CSS
PUT    /api/themes/active               {"id": "midnight"}
POST   /api/themes/install              multipart zip
DELETE /api/themes/{id}
```

## Models

```http
GET  /api/models                    settings, hardware, recommendations, available
GET  /api/models/available          what each worker reports as loaded
PUT  /api/models                    update settings
POST /api/models/reload             hot-reload every OpenVINO pipeline
GET  /api/models/hf/targets
GET  /api/models/hf/search?q=…&target=router|chat|comfy&limit=12
POST /api/models/hf/download        {"repo_id": "...", "target": "router"}
GET  /api/models/hf/downloads
GET  /api/models/hf/downloads/{job_id}
```

## Speech

```http
POST /api/transcribe    multipart WAV → {"text": "..."}
```

16 kHz mono is expected; the launcher resamples before uploading.

## Docs

The handbook you are reading is served as static files at `/docs`. Point the
control panel's Docs button somewhere else — your own docs subdomain, say — with
the `docs_url` setting under **Gateway**.
