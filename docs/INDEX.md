# Keylane handbook

Keylane is a local-first desktop assistant for Linux. Press
<kbd>Super</kbd>+<kbd>Space</kbd>, say what you want, and a small model on the
Intel NPU either does it — opening an app, searching the web, reading a file,
sending an email — or hands it to a bigger AI tool and checks the result.

Everything runs on your machine. The gateway binds to `127.0.0.1` and nothing
leaves it unless you have enabled a cloud worker.

## Start here

- **[Install and set up](INSTALL.md)** — from a clean Fedora install to a working hotkey.
- **[The popup and the tray](POPUP.md)** — the Spotlight overlay, the hotkey, and the background-work indicator.
- **[How the assistant thinks](ASSISTANT.md)** — try it yourself, delegate what you cannot, then follow up.

## Extend it

- **[Tools](TOOLS.md)** — every capability the assistant can reach, and how to add one.
- **[Skills](SKILLS.md)** — teach Keylane your house rules with a markdown file.
- **[Writing plugins](PLUGINS.md)** — add workers, MCP servers, tools and skills.
- **[Writing themes](THEMES.md)** — restyle the panel and reshape the popup.
- **[HTTP API](API.md)** — every endpoint the gateway exposes.

## How the pieces fit

```text
        Super+Space
             │
             ▼
   ┌───────────────────┐        the theme decides whether this is a
   │  Keylane popup    │        bar, a panel, a window or an orb
   └─────────┬─────────┘
             │  text · microphone
             ▼
   ┌───────────────────┐
   │  FastAPI gateway  │  127.0.0.1:9100     ← the authority
   └─────────┬─────────┘
             │
             ▼
   ┌───────────────────┐
   │  NPU assistant    │  plan → act → observe
   └────┬─────────┬────┘
        │         │
   own tools   delegate
        │         │
        │         ▼
        │   ┌──────────────────────────────┐
        │   │ LM Studio · Claude Code ·    │
        │   │ Cursor · ComfyUI · plugins   │
        │   └──────────────┬───────────────┘
        │                  │  evidence
        │                  ▼
        │        ┌───────────────────┐
        └───────▶│   NPU verifier    │  is this actually done?
                 └─────────┬─────────┘
                           │
                  success  │  retry with feedback
                           ▼
                      answer to you
```

## The design rule

**The gateway is the authority, not the model.** Models emit structured
intents — a worker name, a tool name, JSON arguments. Python validates them
against a schema, checks them against your policy, executes them, and collects
evidence. There is no path from model output to a shell, and no tool the
assistant can reach that you have not enabled.

That is what makes it safe to give a small local model real reach over your
desktop.

## What runs where

| Job | Where |
| --- | --- |
| Routing, planning, verification | NPU (OpenVINO INT4, ~1.5B) |
| Chat, long-form writing | GPU via LM Studio or Lemonade |
| Repository work | Claude Code or Cursor, if you enable them |
| Images | ComfyUI, locally |
| Speech to text | Whisper, locally |
| Desktop actions | Python in the gateway process |

## Getting help

- **Status** in the control panel says what is reachable and what is not; the
  NPU row explains exactly which piece is missing.
- `journalctl --user -u ai-gateway.service -f` for gateway logs.
- `python launcher/main.py -v` to run the popup in the foreground with logging.
