# Fedora Local AI Gateway

## OpenVINO NPU Router + NPU Verifier + LM Studio + Claude Code + Cursor CLI + ComfyUI + Super-Key Launcher

## 1. Goal

Build a Fedora Linux desktop AI gateway available system-wide through the **Super/Windows key**.

Pressing Super opens a small GTK4/libadwaita launcher where you can type or dictate an instruction. A small model running through **OpenVINO on the Intel Core Ultra NPU** classifies and routes the request to the appropriate worker:

- **LM Studio** — general local LLM work
- **Claude Code** — advanced coding/repository work
- **Cursor CLI** — coding/repository work
- **ComfyUI** — image generation/editing

The gateway then collects evidence from the worker and sends it to a separate **NPU verifier**. The verifier determines whether the task actually succeeded. Failed tasks can be retried up to a fixed limit.

```text
Super
  |
  v
GTK4 / Libadwaita Launcher
  |---- text
  |---- microphone
  |
  v
FastAPI Gateway :9000
  |
  v
OpenVINO NPU Router
  |
  +---- LM Studio
  +---- Claude Code
  +---- Cursor CLI
  +---- ComfyUI
              |
              v
         Task Evidence
              |
              v
      OpenVINO NPU Verifier
              |
        +-----+-----+
        |           |
     SUCCESS      FAILED
        |           |
        |        retry <= 3
        |           |
        +-----------+
              |
              v
             User
```

## 2. Architecture principles

### NPU = control plane

Use the NPU for small, frequent inference tasks:

- intent classification
- worker selection
- parameter extraction
- structured routing
- lightweight task verification
- optional speech recognition
- optional OCR/embeddings

Do **not** use it as the main engine for large coding models or Flux image generation.

### Gateway = authority

The LLM should never directly control the computer. It outputs a constrained structured request; the Python gateway validates it and chooses what is actually executed.

```text
LLM -> structured request -> gateway validation -> worker
```

### Verifier != executor

The verifier evaluates evidence. It does not run commands or modify files. The gateway makes the final control-flow decision based on the verifier's structured result.

### Local by default

Bind the gateway to `127.0.0.1` and do not expose it to the LAN unless explicitly required.

---

# 3. Prerequisites

Target environment:

- Fedora Linux
- GNOME / Wayland
- Intel Core Ultra 200-series CPU with NPU
- Python 3
- Git
- systemd user services
- GTK4 / libadwaita
- OpenVINO
- LM Studio
- Claude Code
- Cursor CLI
- ComfyUI

Install base packages:

```bash
sudo dnf install -y \
  git python3 python3-pip python3-devel \
  gcc gcc-c++ make pkg-config \
  gtk4-devel libadwaita-devel \
  gobject-introspection-devel cairo-gobject-devel \
  portaudio-devel ffmpeg
```

Create the project:

```bash
mkdir -p ~/.local/share/ai-gateway
cd ~/.local/share/ai-gateway
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

---

# 4. NPU verification — do this first

Fedora/NPU support depends on the current Intel NPU driver, kernel, firmware and OpenVINO/runtime compatibility. Do not build the entire system until direct NPU inference works.

Inspect the system:

```bash
uname -r
lspci -nn
lsmod | grep -E 'intel_vpu|ivpu|vpu'
dmesg | grep -Ei 'vpu|npu|ivpu'
```

Install OpenVINO:

```bash
source ~/.local/share/ai-gateway/.venv/bin/activate
pip install openvino openvino-genai openvino-tokenizers
```

Check the runtime:

```bash
python -c "import openvino; print(openvino.__version__)"
```

Check available devices:

```python
from openvino import Core
core = Core()
print(core.available_devices)
```

You want to see an NPU device, typically alongside CPU/GPU.

If NPU is absent, stop here and resolve the Fedora Intel NPU driver/runtime issue before proceeding.

---

# 5. Router model

Use a small instruction-following model in roughly the **1B–3B** range, converted/quantized into a format supported by OpenVINO and the NPU.

The router does not need to be the smartest model. It needs to reliably produce structured routing decisions quickly.

Suitable model families can include small OpenVINO-compatible SmolLM/LFM-class models. Prefer a model with reliable structured output and low latency over maximum parameter count.

Recommended initial target:

```text
1B–3B instruction model
INT4 or other NPU-suitable quantization
OpenVINO format
NPU target
```

---

# 6. Direct NPU inference test

Before adding FastAPI or workers, prove that the model runs on the NPU.

Conceptually:

```python
import openvino_genai as ov_genai

pipeline = ov_genai.LLMPipeline(
    "./models/router",
    "NPU"
)

response = pipeline.generate(
    "Classify this request as coding, image_generation, or general_question: Fix my React component",
    max_new_tokens=64
)

print(response)
```

Only continue after this successfully performs inference on the NPU.

## Optional: OpenVINO Model Server

Once the prototype works, consider putting the router behind OpenVINO Model Server (OVMS):

```text
FastAPI Gateway -> OVMS -> NPU
```

Benefits:

- router process separated from gateway
- independent restarts
- HTTP/OpenAI-compatible integration options
- easier reuse by other local applications

For the first prototype, direct OpenVINO inference from Python is simpler.

---

# 7. Repository structure

```text
~/.local/share/ai-gateway/
|
├── .venv/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── router.py
│   ├── planner.py
│   ├── verifier.py
│   ├── permissions.py
│   ├── workers/
│   │   ├── __init__.py
│   │   ├── lmstudio.py
│   │   ├── claude.py
│   │   ├── cursor.py
│   │   └── comfyui.py
│   ├── npu/
│   │   ├── __init__.py
│   │   ├── router_model.py
│   │   └── verifier_model.py
│   └── audio/
│       ├── __init__.py
│       └── transcription.py
├── launcher/
│   ├── main.py
│   ├── window.py
│   └── assets/
├── models/
│   └── router/
├── workflows/
│   ├── flux_txt2img.json
│   ├── flux_img2img.json
│   ├── flux_inpaint.json
│   └── upscale.json
├── config/
│   ├── workers.toml
│   └── projects.toml
├── tests/
└── systemd/
    ├── ai-gateway.service
    └── ai-launcher.service
```

---

# 8. Python dependencies

`requirements.txt`:

```text
fastapi
uvicorn[standard]
httpx
pydantic
pydantic-settings
openvino
openvino-genai
openvino-tokenizers
PyGObject
sounddevice
numpy
```

Install:

```bash
pip install -r requirements.txt
```

If PyGObject installation through pip conflicts with Fedora's GTK bindings, use Fedora's system packages and make the virtual environment able to access the system bindings.

---

# 9. Worker configuration

`config/workers.toml`:

```toml
[gateway]
host = "127.0.0.1"
port = 9000
max_retries = 3

[npu]
model_path = "./models/router"
device = "NPU"

[lmstudio]
base_url = "http://127.0.0.1:1234/v1"
default_model = "local-model"

[comfyui]
base_url = "http://127.0.0.1:8188"

[claude]
command = "claude"

[cursor]
command = "cursor-agent"

[security]
allowed_project_roots = [
  "/home/YOUR_USER/projects"
]
```

Do not put API secrets in this file. Use environment variables or the appropriate application credential store if a worker requires credentials.

---

# 10. Gateway API

Implement these endpoints:

```text
POST /api/chat
POST /api/route
POST /api/transcribe
POST /api/tasks
POST /api/tasks/{id}/cancel
GET  /api/tasks/{id}
GET  /api/status
GET  /api/projects
```

Example request:

```json
{
  "message": "Fix the authentication bug in my current project.",
  "project": "/home/user/projects/my-app"
}
```

Example response:

```json
{
  "task_id": "abc123",
  "status": "completed",
  "worker": "cursor",
  "result": "..."
}
```

---

# 11. Router schema

Use strict Pydantic models.

```python
class RouteDecision(BaseModel):
    intent: str
    worker: str
    action: str
    instruction: str
    working_directory: str | None = None
    arguments: dict = {}
    requires_confirmation: bool = False
```

Allowed workers:

```text
lmstudio
claude
cursor
comfyui
```

Reject anything outside the allowlist.

Do not allow the model to return arbitrary shell commands.

---

# 12. Router system prompt

Use a strict prompt similar to:

```text
You are the routing model for a local Fedora AI gateway.

You do not directly answer the user.

Determine which available worker should handle the request.

Available workers:

lmstudio:
General-purpose local AI. Use for questions, brainstorming,
summarization, analysis and privacy-sensitive tasks.

claude:
Advanced coding and repository tasks.

cursor:
Advanced coding and repository tasks.

comfyui:
Image generation and image manipulation.

Rules:
1. Return ONLY valid JSON.
2. Never invent a worker.
3. Never execute commands.
4. Never claim that a task was completed.
5. Choose the minimum number of workers necessary.
6. Respect an explicit user request for Claude or Cursor.
7. Respect local-only mode.
8. Never output arbitrary shell commands.
9. Use the supplied project directory rather than inventing one.
10. Set requires_confirmation=true for modifications unless gateway policy explicitly permits them.

Return:
{
  "intent": "...",
  "worker": "...",
  "action": "...",
  "instruction": "...",
  "working_directory": "...",
  "arguments": {},
  "requires_confirmation": false
}
```

The gateway independently validates every response.

---

# 13. LM Studio worker

LM Studio should provide local general-purpose inference through its local OpenAI-compatible API, normally:

```text
http://127.0.0.1:1234/v1
```

Use `httpx` or an OpenAI-compatible client.

LM Studio is the preferred worker for:

- general questions
- brainstorming
- summarization
- local analysis
- privacy-sensitive reasoning

The gateway should detect whether LM Studio is available before routing to it.

---

# 14. Claude Code worker

Invoke Claude Code as a controlled subprocess, for example:

```bash
claude -p "Fix the authentication bug" --output-format json
```

The exact CLI arguments should be verified against the installed Claude Code version.

Always set:

```python
cwd=working_directory
```

Capture:

- stdout
- stderr
- exit code
- changed files/git diff
- tests
- build output

Do not default to the entire home directory.

Do not use dangerous permission-bypass options by default.

---

# 15. Cursor CLI worker

Invoke Cursor CLI as a controlled subprocess, for example:

```bash
cursor-agent -p "Fix the authentication bug"
```

Verify the exact invocation supported by the installed Cursor CLI version.

Set the project directory with `cwd` and capture:

- stdout
- stderr
- exit code
- Git diff
- test results
- build results

Do not permit access outside configured project roots.

---

# 16. ComfyUI worker

ComfyUI normally runs locally at:

```text
http://127.0.0.1:8188
```

Maintain predefined workflows:

```text
workflows/
├── flux_txt2img.json
├── flux_img2img.json
├── flux_inpaint.json
└── upscale.json
```

Example route:

```json
{
  "worker": "comfyui",
  "action": "generate_image",
  "workflow": "flux_txt2img",
  "arguments": {
    "prompt": "A cyberpunk city at night",
    "width": 1536,
    "height": 1024
  }
}
```

The gateway loads the approved workflow and changes only whitelisted fields. Do not allow the router to create arbitrary ComfyUI graphs.

---

# 17. Multi-step planning

Simple request:

```text
User -> NPU router -> one worker
```

Complex request:

```text
User -> NPU planner -> step 1 -> step 2 -> verifier
```

Example:

```text
Build a landing page and generate the hero artwork.
```

Plan:

```text
1. ComfyUI -> generate hero image
2. Cursor -> integrate image into project
3. Run build/tests
4. NPU verifier
```

The gateway owns dependencies and passes outputs between workers.

---

# 18. NPU verifier

The verifier is a separate NPU inference call.

It receives:

- original user request
- planned task
- worker selected
- worker output
- command results
- build/test results
- files changed
- retry count

It must not execute anything.

Its sole question is:

```text
Did the worker actually fulfill the request?
```

Schema:

```python
class VerificationResult(BaseModel):
    complete: bool
    confidence: float
    reason: str
    retry: bool
    next_action: str | None
```

Success:

```json
{
  "complete": true,
  "confidence": 0.96,
  "reason": "The requested change is present and the project builds successfully.",
  "retry": false,
  "next_action": null
}
```

Failure:

```json
{
  "complete": false,
  "confidence": 0.98,
  "reason": "The TypeScript build still fails.",
  "retry": true,
  "next_action": "Fix TS2345 in src/auth/session.ts and rerun the build."
}
```

---

# 19. Verification evidence

For coding tasks collect:

```text
git status
git diff
git diff --stat
worker stdout
worker stderr
worker exit code
test exit code
build exit code
lint exit code
```

Never expose secrets such as `.env` contents to the verifier.

For ComfyUI collect:

- workflow status
- output path
- output dimensions
- requested parameters
- worker result

For LM Studio collect:

- response
- required fields
- output format compliance

---

# 20. Verification loop

Set:

```python
MAX_RETRIES = 3
```

Conceptual implementation:

```python
for attempt in range(MAX_RETRIES + 1):
    result = execute_worker(task)
    evidence = collect_evidence(result)

    verification = verify_with_npu(
        original_request=original_request,
        task=task,
        evidence=evidence,
    )

    if verification.complete:
        return result

    if not verification.retry:
        return result_with_failure(result, verification.reason)

    if attempt >= MAX_RETRIES:
        return result_with_failure(result, "Maximum retries reached.")

    task.instruction = verification.next_action
```

The NPU makes the judgment; **the Python gateway controls the loop**.

---

# 21. Security model

This application is effectively a local AI agent controller. Treat it as privileged software.

## Never allow the router to execute arbitrary shell commands

Bad:

```json
{"command":"rm -rf /"}
```

Good:

```json
{
  "worker": "cursor",
  "action": "modify_project"
}
```

The gateway and worker policy determine which operations actually occur.

## Project sandbox

Configure allowed project roots, e.g.:

```text
/home/user/projects
/home/user/work
```

Resolve paths before checking them to prevent traversal attacks.

## Confirmation levels

Automatic:

- general questions
- read-only analysis
- image generation
- local inference

Confirmation:

- modifying files
- Git commits
- installing packages
- development commands

Always confirm:

- destructive operations
- deleting files
- system configuration changes
- package removal
- commands outside project roots
- access to sensitive directories

---

# 22. Local-only mode

Add a `Local Only` setting.

Allowed:

```text
NPU
LM Studio
ComfyUI
```

Blocked:

```text
Claude Code
Cursor CLI
```

The gateway must enforce this. Do not merely instruct the router not to use cloud services.

---

# 23. Project selection

`config/projects.toml`:

```toml
[[projects]]
name = "Aurora"
path = "/home/user/projects/aurora"

[[projects]]
name = "Moments"
path = "/home/user/projects/moments"

[[projects]]
name = "Chrono Defender"
path = "/home/user/projects/chrono-defender"
```

If no project is selected:

- LM Studio read-only tasks can run.
- ComfyUI can run.
- Coding workers should ask the user to select a project.

---

# 24. GTK4 launcher

Use GTK4/libadwaita.

Requirements:

- start at login
- stay resident
- hide main window when idle
- appear immediately on activation
- focus the text field
- keyboard navigation
- microphone button
- task progress
- confirmation dialogs
- worker status

Suggested UI:

```text
╭────────────────────────────────────────────────────╮
│ Ask your computer...                           🎙 │
│                                                    │
│                                                    │
│                                                    │
│ Project: Aurora ▼                                  │
│                                                    │
│ Enter Send   Ctrl+Enter Send & Hide   Esc Close    │
╰────────────────────────────────────────────────────╯
```

States:

```text
IDLE
INPUT
DICTATING
ROUTING
WAITING_CONFIRMATION
RUNNING
VERIFYING
RETRYING
SUCCESS
FAILURE
```

---

# 25. Voice input

Recommended first implementation:

```text
Hold/click microphone
        |
        v
Audio capture
        |
        v
Whisper
        |
        v
Text field
        |
        v
FastAPI
```

Start with push-to-talk rather than always-listening.

If NPU Whisper inference is difficult on Fedora, initially run transcription on CPU and add NPU acceleration after the rest of the system works.

---

# 26. Super-key integration

Fedora GNOME normally uses Wayland, so do not build the core design around X11 tools such as `xdotool` or `xbindkeys`.

Use a GNOME Shell extension or another GNOME-compatible global shortcut mechanism.

Desired behavior:

```text
Super
  |
  v
GNOME Shell integration
  |
  v
focus/launch ai-launcher
```

If the current GNOME version makes bare Super interception impractical because GNOME reserves the key, use a fallback such as:

```text
Super + Space
```

The launcher must remain independent from the activation mechanism.

---

# 27. systemd services

`systemd/ai-gateway.service`:

```ini
[Unit]
Description=Local AI Gateway
After=network-online.target

[Service]
WorkingDirectory=%h/.local/share/ai-gateway
ExecStart=%h/.local/share/ai-gateway/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 9000
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
```

Install the user service:

```bash
mkdir -p ~/.config/systemd/user
cp systemd/ai-gateway.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ai-gateway.service
```

Check:

```bash
systemctl --user status ai-gateway.service
journalctl --user -u ai-gateway.service -f
```

Launcher service:

```ini
[Unit]
Description=AI Gateway Launcher
After=graphical-session.target

[Service]
ExecStart=%h/.local/share/ai-gateway/.venv/bin/python %h/.local/share/ai-gateway/launcher/main.py
Restart=on-failure

[Install]
WantedBy=default.target
```

Install similarly with `systemctl --user`.

---

# 28. Worker health checks

`GET /api/status` should return something like:

```json
{
  "npu": true,
  "lmstudio": true,
  "comfyui": true,
  "claude": true,
  "cursor": true
}
```

The launcher can display:

```text
NPU       ●
LM Studio ●
ComfyUI   ●
Claude    ●
Cursor    ●
```

Unavailable workers must be removed from routing.

---

# 29. Explicit worker requests

Support instructions such as:

```text
Use Claude to fix this.
```

```text
Use Cursor to refactor this.
```

```text
Do this locally.
```

```text
Generate an image of...
```

The router should respect explicit worker requests unless the worker is unavailable or prohibited by policy.

---

# 30. Example coding task

User:

```text
Fix the authentication bug in my Aurora project.
```

NPU router:

```json
{
  "intent": "coding",
  "worker": "cursor",
  "action": "modify_project",
  "instruction": "Investigate and fix the authentication bug.",
  "working_directory": "/home/user/projects/aurora",
  "requires_confirmation": true
}
```

Launcher:

```text
Cursor wants to modify the Aurora project.

[Allow] [Cancel]
```

After approval:

```text
Cursor
  |
  v
inspect project
  |
  v
modify files
  |
  v
build/tests
  |
  v
collect evidence
  |
  v
NPU verifier
```

Successful result:

```text
✓ Completed

Cursor modified 3 files.
npm run build ✓
Tests ✓
NPU verification ✓
```

---

# 31. Example failed coding task

Worker reports success, but build fails:

```text
Cursor: changed auth.ts
Build: TS2345
```

Verifier:

```json
{
  "complete": false,
  "confidence": 0.99,
  "reason": "The project still fails TypeScript compilation.",
  "retry": true,
  "next_action": "Fix TS2345 in src/auth/session.ts and rerun the build."
}
```

Gateway:

```text
Retry 1/3
```

The worker receives the actual error, fixes it, reruns checks, and the verifier evaluates the new evidence.

---

# 32. Example image-generation task

User:

```text
Generate a 1536x1024 cyberpunk city background for my game.
```

Router:

```json
{
  "intent": "image_generation",
  "worker": "comfyui",
  "action": "generate_image",
  "workflow": "flux_txt2img",
  "arguments": {
    "prompt": "cyberpunk city background",
    "width": 1536,
    "height": 1024
  },
  "requires_confirmation": false
}
```

Gateway:

```text
Load approved workflow
 -> set prompt
 -> set dimensions
 -> submit to ComfyUI
 -> wait for completion
 -> collect output
 -> NPU verification
```

Verifier checks that the output exists and has the requested dimensions.

---

# 33. Example multi-worker task

User:

```text
Build a landing page for my game and create a cyberpunk hero background.
```

Plan:

```text
1. ComfyUI -> generate hero image
2. Cursor -> integrate image into project
3. Run build/tests
4. NPU verifier
```

Dependency:

```text
ComfyUI output
      |
      v
image path
      |
      v
Cursor
```

The gateway controls the dependency chain.

---

# 34. OpenAI-compatible gateway (optional)

Eventually expose:

```text
http://127.0.0.1:9000/v1/chat/completions
```

Example:

```json
{
  "model": "local-agent",
  "messages": [
    {
      "role": "user",
      "content": "Fix this project."
    }
  ]
}
```

The application does not need to know whether the task is handled by LM Studio, Claude Code, Cursor, or ComfyUI.

---

# 35. Recommended build order

## Phase 1 — NPU validation

```text
Fedora
 -> Intel NPU driver
 -> OpenVINO
 -> small model
 -> NPU inference
```

Success criterion: a model demonstrably executes on the NPU.

## Phase 2 — NPU router

Implement:

```text
request -> NPU -> strict JSON route
```

Test:

```text
coding -> Cursor/Claude
image -> ComfyUI
general question -> LM Studio
```

## Phase 3 — FastAPI

```text
POST /api/chat
 -> NPU router
 -> route decision
```

## Phase 4 — LM Studio

Connect local general-purpose inference.

## Phase 5 — ComfyUI

Add predefined image workflows.

## Phase 6 — Claude Code

Add controlled subprocess execution.

## Phase 7 — Cursor CLI

Add controlled subprocess execution.

## Phase 8 — NPU verifier

Implement:

```text
worker
 -> evidence
 -> NPU verifier
 -> success/retry
```

This should be completed before autonomous multi-step planning.

## Phase 9 — GTK launcher

```text
GTK4 -> FastAPI
```

## Phase 10 — Voice

```text
microphone -> Whisper -> text
```

## Phase 11 — Super key

Add GNOME Shell integration.

## Phase 12 — Multi-step planning

Add:

```text
planner
 -> worker 1
 -> worker 2
 -> verification
 -> retry
```

---

# 36. Minimum viable release

The first useful version should contain only:

```text
Super
 |
v
GTK launcher
 |
v
text input
 |
v
FastAPI
 |
v
NPU router
 |
+-- LM Studio
+-- Claude Code
+-- Cursor CLI
+-- ComfyUI
 |
v
NPU verifier
 |
v
result
```

Do not initially add:

- browser automation
- unrestricted terminal access
- always-on microphone
- complex long-term memory
- multiple simultaneous autonomous agents
- arbitrary ComfyUI graph generation
- remote network access
- unnecessary MCP infrastructure

Add these only after the basic control plane is reliable.

---

# 37. Final target architecture

```text
                               FEDORA
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
             GNOME Shell                  systemd --user
                    |                           |
                    v                  +--------+--------+
                Super Key              |                 |
                    |              AI Gateway       AI Launcher
                    v                 :9000               |
             GTK4 / Libadwaita          |                |
                    |                    |                |
              +-----+-----+              |                |
              |           |              |                |
             Text      Whisper           |                |
              |           |              |                |
              |          NPU             |                |
              +-----+-----+              |                |
                    v                    |                |
               FastAPI Gateway <----------+                |
                    |                                     |
                    v                                     |
               NPU ROUTER                                 |
             OpenVINO / NPU                               |
                    |                                     |
          +---------+---------+------------+              |
          |         |         |            |              |
          v         v         v            v              |
      LM Studio  Claude    Cursor      ComfyUI            |
       Local      Code       CLI         GPU              |
          |         |         |            |              |
          +---------+---------+------------+              |
                            |                             |
                            v                             |
                       Task Evidence                      |
                            |                             |
                            v                             |
                       NPU VERIFIER                       |
                     OpenVINO / NPU                       |
                            |                             |
                    +-------+-------+                     |
                    |               |                     |
                    v               v                     |
                 SUCCESS          RETRY                   |
                    |               |                     |
                    |               +----> Worker -------+
                    |                                      |
                    +--------------------------------------+
```

---

# 38. Completion checklist

- [ ] Fedora recognizes the Intel NPU.
- [ ] OpenVINO sees the NPU.
- [ ] A supported small model successfully runs on NPU.
- [ ] Router produces valid JSON.
- [ ] FastAPI receives instructions.
- [ ] LM Studio worker works.
- [ ] Claude Code worker works in a controlled project directory.
- [ ] Cursor CLI worker works in a controlled project directory.
- [ ] ComfyUI worker works with approved workflows.
- [ ] Gateway collects worker evidence.
- [ ] NPU verifier independently evaluates evidence.
- [ ] Failed tasks retry automatically up to the configured limit.
- [ ] File modifications have appropriate confirmation.
- [ ] Local-only mode blocks cloud workers.
- [ ] Project roots are sandboxed.
- [ ] GTK launcher works.
- [ ] Text input works.
- [ ] Voice input works.
- [ ] Gateway starts automatically.
- [ ] Launcher starts automatically.
- [ ] Super-key activation works, or a documented GNOME-compatible fallback exists.
- [ ] Worker health status is displayed.
- [ ] A coding task completes end-to-end.
- [ ] An image-generation task completes end-to-end.
- [ ] A failed task successfully demonstrates verifier-driven retry.

---

# 39. Operating philosophy

The finished system should feel like one local AI command center rather than four unrelated applications.

The user interacts with one interface:

```text
Super
```

The NPU decides what specialist should handle the request:

```text
NPU Router
```

The appropriate specialist performs the expensive work:

```text
LM Studio
Claude Code
Cursor CLI
ComfyUI
```

The NPU then independently asks:

```text
Was the task actually completed?
```

The gateway decides:

```text
SUCCESS
RETRY
ASK USER
```

The important separation is:

```text
NPU = lightweight control + verification
Workers = actual work
Gateway = permissions + orchestration + state
GTK = user interface
GNOME = global activation
```

This makes the NPU useful every day without wasting its limited compute on workloads that belong on the CPU/GPU or in specialist applications.
