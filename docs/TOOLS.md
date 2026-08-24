# Tools

A tool is one narrow capability the assistant can invoke: launch an app, search
the web, read a file, hand a job to Claude Code. The model picks a name and
supplies JSON arguments; Python validates, gates and executes.

See the live list — including everything your plugins contribute — at
**Control panel → Assistant → Tools**, or:

```bash
curl -s 'http://127.0.0.1:9100/api/tools?available_only=true' | python -m json.tool
```

## Built-in tools

### Desktop

| Tool | Danger | What it does |
| --- | --- | --- |
| `list_applications` | safe | List installed apps, optionally filtered |
| `open_application` | sensitive | Launch an app by name via its `.desktop` entry |
| `open_url` | sensitive | Open a URL, file or folder in the default handler |
| `send_notification` | safe | Show a desktop notification |
| `read_clipboard` | safe | Read the clipboard as text |
| `write_clipboard` | sensitive | Replace the clipboard contents |
| `media_control` | sensitive | play / pause / next / previous / stop via `playerctl` |
| `set_volume` | sensitive | Set output volume or mute via `wpctl` / `pactl` |

App launching scans `~/.local/share/applications`, `/usr/share/applications` and
the Flatpak and Snap export directories, ranking exact matches first. It launches
through `gtk-launch`, then `gio launch`, then the `Exec` line with field codes
stripped.

### Web

| Tool | Danger | What it does |
| --- | --- | --- |
| `web_search` | safe | Search via DuckDuckGo or a self-hosted SearXNG |
| `fetch_url` | safe | Download a page and return its readable text |

Choose the engine under **Assistant → Web search**. Setting it to `none`
disables both the engine and the tool.

### Files

| Tool | Danger | What it does |
| --- | --- | --- |
| `list_files` | safe | List a folder |
| `read_file` | safe | Read a text file |
| `write_file` | sensitive | Create, overwrite or append to a file |
| `search_files` | safe | Grep for text, or find files by glob |

Every path is resolved and checked before any I/O. The sandbox is your
configured project roots plus `~/Documents`, `~/Downloads`, `~/Pictures`,
`~/Desktop`, `~/Music`, `~/Videos` and the ComfyUI output directory. These are
refused even inside the sandbox: `.ssh`, `.gnupg`, `.password-store`, `.aws`,
`.kube`, `.docker`, `.mozilla`, `.thunderbird`, `keyrings`, and any file named
`.env`, `.netrc`, `.pgpass`, `credentials`, `id_rsa`, `id_ed25519`.

### System

| Tool | Danger | What it does |
| --- | --- | --- |
| `system_info` | safe | Host, OS, uptime, CPU, memory, disk, battery, local time |
| `run_command` | **dangerous** | Run one allowlisted program |

### Delegation

| Tool | Danger | What it does |
| --- | --- | --- |
| `list_workers` | safe | Which AI workers are configured, enabled and reachable |
| `delegate_to_worker` | sensitive | Hand a task to Claude Code, Cursor, ComfyUI, LM Studio… |
| `verify_result` | safe | Ask the NPU verifier whether the work satisfies the request |

### Communication

| Tool | Danger | What it does |
| --- | --- | --- |
| `send_email` | sensitive | Send mail through your configured SMTP account |

Unavailable until you fill in **Assistant → Email**. Passwords may be written as
`env:VAR_NAME` to read from the environment instead of `config/assistant.toml`.
For Gmail and similar, use an app password, not your account password.

## `run_command` in detail

This is the one tool with real blast radius, so it is fenced in four ways:

1. **No shell.** The model supplies a program and an argument array; they go
   straight to `execve`. Pipes, redirection, globbing and `;` do nothing.
2. **Allowlist.** The program must appear in `shell.allowlist`. The default list
   is read-oriented: `ls cat head tail grep rg find wc df du free uptime date
   whoami hostname uname ps systemctl journalctl git python3 pip nmcli ip lsblk
   sensors playerctl wpctl pactl notify-send xdg-open gio flatpak`.
3. **Hard deny list.** `rm rmdir shred mkfs dd fdisk parted sudo su doas pkexec
   chown chmod passwd useradd userdel usermod visudo reboot poweroff shutdown
   halt init sh bash zsh fish dash eval exec` are refused even if you add them to
   the allowlist.
4. **Confirmation.** It is `dangerous`, so it is gated at every default setting.

Arguments that would open a nested shell (`-c`, `--command`, `-e`, `--eval`,
`--exec`) are rejected.

> **Warning**: Adding an interpreter to the allowlist (`python3`, `perl`,
> `node`) effectively grants arbitrary code execution, because the interpreter
> can be handed a script path. `python3` ships in the default list for
> convenience; remove it if you want a tighter boundary.

## Plugin tools

Any plugin can contribute tools, and MCP plugins contribute theirs
automatically. Enabling the ComfyUI MCP plugin, for instance, adds around forty
tools named `comfyui.generate_image`, `comfyui.run_workflow`,
`comfyui.system_stats` and so on.

MCP tool danger is inferred from the name: tools starting with `list`, `get`,
`search`, `info`, `status`, `read`, `describe`, `which`, `validate`, `fetch` or
`stats` are treated as `safe`; everything else is `sensitive`. Override it with
`tools.auto_confirm` or `tools.deny` if the guess is wrong for your server.

Press **Rediscover** on the Assistant tab (or `POST /api/tools/refresh`) after
changing an MCP server.

## Writing a tool

Tools live in a plugin. See [Writing plugins](PLUGINS.md) for the full
walkthrough; the tool itself is small:

```python
from app.tools.base import BaseTool, ToolDanger, ToolResult, object_schema, string_prop


class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Current weather for a city. Use when the user asks about weather."
    danger = ToolDanger.SAFE
    category = "web"

    def parameters(self):
        return object_schema(
            {"city": string_prop("City name, e.g. 'Cairo'.")},
            required=["city"],
        )

    def availability(self):
        # Return a reason string to grey the tool out, or None when usable.
        return None

    async def run(self, args):
        import httpx

        city = str(args.get("city", "")).strip()
        if not city:
            return ToolResult.failure("No city given.")
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"https://wttr.in/{city}?format=3")
            r.raise_for_status()
        return ToolResult.success(r.text.strip(), data={"city": city})
```

### Writing good tool descriptions

The description is the only thing the model sees when deciding whether to reach
for your tool. Say **what it does and when to use it**, in one or two sentences.

- Good: *"Current weather for a city. Use when the user asks about weather."*
- Poor: *"Weather tool."*

Parameter descriptions matter as much — they are how the model knows to pass
`"Cairo"` rather than `"the city where the user lives"`.

### Choosing a danger level

| Level | Rule of thumb |
| --- | --- |
| `SAFE` | Reads state, or is trivially reversible. Runs without asking. |
| `SENSITIVE` | Writes, sends, launches, or costs money. Asks first by default. |
| `DANGEROUS` | Arbitrary execution or destructive potential. Always gated. |

When in doubt, pick the higher one. A user can always lower the threshold or add
your tool to `auto_confirm`; they cannot un-send an email.

### Returning results

```python
ToolResult.success("Human-readable text the model will read",
                   data={"structured": "fields"},
                   artifacts=["/path/to/produced/file.png"])

ToolResult.failure("What went wrong, and what would fix it")
```

The `output` string is fed back to the model verbatim, so make it useful: a
failure that says *"'firefox' is not installed; call list_applications to see
what is available"* gets recovered from, one that says *"error"* does not.

## Calling a tool directly

Useful for testing:

```bash
curl -X POST http://127.0.0.1:9100/api/tools/call \
  -H 'Content-Type: application/json' \
  -d '{"tool": "system_info", "arguments": {}}'
```

A gated tool returns `{"requires_confirmation": true, ...}` until you resend with
`"confirmed": true`.
