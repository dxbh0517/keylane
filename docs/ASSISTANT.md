# How the assistant thinks

Keylane's assistant is a small model on the Intel NPU. It is not trying to be
the smartest thing on your machine — it is trying to be the *fastest* thing, and
the thing that knows what everything else is for.

Its whole job is three steps:

1. **Can I do this myself?** It has real tools: launch an application, search the
   web, read a file, check the battery, copy to the clipboard, send an email,
   run an allowlisted command. If the request is one of those, it just does it.
2. **If not, delegate.** Multi-file code changes, deep reasoning, long-form
   writing, image generation — these go to a configured AI tool with a complete,
   self-contained instruction.
3. **Follow up.** The assistant, not the worker, is accountable for the answer.
   When a delegation returns it inspects the evidence: did files actually change,
   does the output file exist, did the command exit cleanly, does the answer
   address what was asked? If not, it delegates again with corrective feedback.

## The loop

```text
       your request
            │
            ▼
   ┌─────────────────┐
   │  NPU assistant  │◀────────────┐
   └────────┬────────┘             │
            │ emits one JSON step  │  observation
            ▼                      │
   ┌─────────────────┐             │
   │  gateway (you)  │             │
   │  validates      │             │
   │  gates          │             │
   │  executes       │─────────────┘
   └────────┬────────┘
            │
    ┌───────┴────────┬──────────────┐
    ▼                ▼              ▼
 own tool      delegate to      verify the
 (open app,    a worker         result
 search…)      (Claude, Comfy)
            │
            ▼
     answer to you
```

The model never executes anything. It emits one small JSON object per turn; the
gateway validates it against the tool schema, applies the confirmation policy,
runs it, and feeds the observation back. Control flow is Python's.

## What the model is told

The system prompt is assembled fresh for every request from four parts:

1. The standing instructions (try yourself → delegate → follow up).
2. The **live tool catalogue** — only tools that are enabled, permitted by
   policy, and actually available on this machine.
3. Any [skills](SKILLS.md) whose trigger words appear in your request.
4. Your house rules from **Assistant → House rules**.

You can read exactly what the model sees at **Assistant → System prompt**, or:

```bash
curl -s http://127.0.0.1:9100/api/assistant | python -c 'import json,sys; print(json.load(sys.stdin)["system_prompt"])'
```

## The reply format

One JSON object per turn, nothing else:

```json
{"thought": "Firefox is installed, I can open it.",
 "action": "tool", "tool": "open_application",
 "arguments": {"application": "Firefox"}}
```

```json
{"thought": "Refactoring across files needs a real coding agent.",
 "action": "tool", "tool": "delegate_to_worker",
 "arguments": {"worker": "claude", "intent": "coding",
               "project": "/home/you/code/app",
               "instruction": "Extract the retry logic in api/client.py into a
                               decorator and update all three call sites."}}
```

```json
{"thought": "The image exists and matches the request.",
 "action": "final",
 "answer": "Generated the hero image at ~/outputs/hero_0001.png (1536×1024)."}
```

```json
{"thought": "I do not know who this should go to.",
 "action": "ask", "question": "What address should I send the summary to?"}
```

Anything malformed falls back to the keyword path rather than looping.

## Delegation

`delegate_to_worker` is a tool like any other, but it is the one that matters.
The assistant calls `list_workers` when unsure, then hands over:

| Argument | Meaning |
| --- | --- |
| `worker` | `claude`, `cursor`, `comfyui`, `lmstudio`, `lemonade`, or any plugin worker |
| `instruction` | The complete task — the worker cannot see your conversation |
| `intent` | `coding`, `image_generation`, `analysis`, `summarization`, … |
| `project` | Absolute path; required for coding workers |
| `reason` | Why it is delegating rather than doing it itself |

The result comes back with the worker's evidence attached: exit code, changed
files, output paths, stderr. That is what the follow-up step reasons over.

Delegation is capped (`max_delegations`, default 2) so a confused model cannot
spend your afternoon in Claude Code.

## Following up

After a delegation the assistant is expected to check the work. It can call
`verify_result`, which runs the same NPU verifier the worker pipeline uses:

```json
{"complete": false, "confidence": 0.9,
 "reason": "No file changes were detected for a modification request.",
 "retry": true,
 "next_action": "Make the required code changes and verify with git diff."}
```

An incomplete verdict with `retry: true` gives the assistant a concrete next
step, which it can feed into a second delegation.

## Confirmation

Every tool carries a danger level, and the gateway — not the model — decides
what needs your approval:

| Level | Examples | Default |
| --- | --- | --- |
| `safe` | `web_search`, `read_file`, `system_info`, `list_workers` | Runs immediately |
| `sensitive` | `open_application`, `write_file`, `send_email`, `delegate_to_worker` | **Asks first** |
| `dangerous` | `run_command` | **Asks first** |

Change the threshold under **Assistant → Ask before running**. When a tool needs
approval the task pauses at `waiting_confirmation`, the popup shows Allow /
Cancel, and the tray icon switches to its attention state.

## When there is no NPU model

If no OpenVINO export is loaded, the assistant falls back to keyword matching for
a handful of unambiguous requests — "open firefox", "search the web for …",
"system status" — and hands everything else to the normal worker router. The
control panel says so plainly on both Status and Assistant.

To get the real loop, download a small instruct model under **Models → Search
Hugging Face** with target **Router**. An INT4 OpenVINO export of a 1.5B–3B
instruct model is the sweet spot for the NPU.

## Settings

All of these live in `config/assistant.toml` and on the **Assistant** tab.

| Setting | Default | Effect |
| --- | --- | --- |
| `tools.enabled` | `true` | Master switch; off sends every request straight to a worker |
| `tools.allow` | `[]` | When non-empty, an exclusive allowlist of tool names |
| `tools.deny` | `[]` | Tool names that can never run |
| `tools.confirm_danger_at` | `sensitive` | Lowest level that needs approval |
| `tools.auto_confirm` | `[]` | Tools that skip approval regardless |
| `tools.max_steps` | `6` | Tool calls per request |
| `delegation.enabled` | `true` | Allow handing work to other AI tools |
| `delegation.follow_up` | `true` | Verify delegated results |
| `delegation.max_delegations` | `2` | Delegations per request |
| `persona` | `""` | Extra sentences appended to the system prompt |

## Safety properties

- The model emits intents; **Python validates and executes**. There is no path
  from model output to a shell.
- File tools are sandboxed to your configured project roots plus the usual home
  folders, with `.ssh`, `.gnupg`, `.env` and friends blocked outright.
- `run_command` has no shell: arguments go straight to `execve`, the program
  must be on an allowlist, and a hard-coded deny list (`rm`, `sudo`, `dd`, `sh`…)
  cannot be overridden.
- Email can be restricted to an allowed-recipients list.
- Tool output is an *observation*, never an instruction. A web page telling the
  assistant to ignore its rules is just text.
- Local-only mode removes every cloud worker from the catalogue, so the
  assistant cannot delegate off-machine.
