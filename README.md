<p align="center">
  <img src="assets/logo.png" alt="Keylane logo" width="128" height="128" />
</p>

<h1 align="center">Keylane</h1>

<p align="center">
  <strong>Your Jarvis — always on, on your NPU</strong><br />
  Press Super+Space. An 8B model on the Intel NPU helps you think, search, plan, and act.
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
- A **curated 1.5–14B model** runs **always-on** on your **Intel NPU**, through either
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
- **Dictation into any window** — hold a key, talk, and the text lands where your caret is,
  not in Keylane's box. **Compose** does the same for intent: say what you want written,
  and Keylane reads the screen and drafts it
- **Keylane draws on your screen** — rings, arrows and step-by-step walkthroughs over your
  live desktop, with targets located by reading the screen rather than guessed at
- **Drop a file on it** — images, PDFs, code and web pages become the subject of the turn
- **MCP servers** over stdio *or* Streamable HTTP — including Mailspring for mail and calendar
- **Audio8 TTS** for spoken answers and optional notify speech
- **Answers stream** into the HUD as the model writes them
- **Updates itself from GitHub**, on your say-so, with a one-command rollback
- **Serves its own model** over the OpenAI API, so other tools can use the NPU

Everything binds to `127.0.0.1`. Nothing leaves your machine unless a tool or MCP server you configured does.

## Quick start

```bash
# Install system deps (Fedora)
sudo dnf install python3-gobject gtk4 gtk4-layer-shell libnotify portaudio ffmpeg
# For the screen layer: typing into other windows, and locating targets by text.
# wtype on wlroots; ydotool on GNOME/Wayland (see "The screen layer").
sudo dnf install wtype tesseract   # or: ydotool tesseract

# Python env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start daemon (first run downloads ~4 GB; the compile itself is about a minute)
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
| Screen | Dictation on/off, Whisper size, language, cleanup; annotation style, OCR targeting; which typing backend was selected |
| Security | Shell allowlist, permitted read directories, permission modes per tool |
| MCP | Servers over stdio (command, arguments, environment) or HTTP (URL, bearer token) |
| About | Version, update channel, release notes, install and restart |
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
| **… on an NVIDIA GPU** | the same, through CUDA | `pip install onnxruntime-genai-cuda "onnxruntime-gpu[cuda,cudnn]"` |

The runtime is a property of the export, not a preference: an `*-int4-ov` repo
holds OpenVINO IR and only OpenVINO GenAI can load it. ONNX Runtime reaches the
same NPU through the **OpenVINO execution provider**, which is what the second
package is for — without it, ONNX models run on CPU only.

> Installing `onnxruntime-openvino` may pin a different OpenVINO version than
> the one `openvino-genai` wants. If OpenVINO GenAI stops loading afterwards,
> install the two runtimes in separate virtualenvs and keep the one you use.

**Device** is chosen per runtime in the same panel (`NPU` / `GPU` / `CUDA` /
`CPU`, plus `AUTO` on ONNX Runtime, which keeps whatever provider the model
shipped with). Changing the device invalidates the compile cache, so the next
load is slow either way.

`GPU` and `CUDA` are not two names for the same thing, and the difference is
the reason a discrete card used to be unreachable. `GPU` means an **Intel** GPU
through the OpenVINO EP. OpenVINO also enumerates an NVIDIA card as `GPU` and
then cannot compile for it, so picking it would start a compile that never
finishes — it is shown greyed with the reason. `CUDA` is that same card
reached through ONNX Runtime's CUDA provider, which is the one path that
actually runs a model on it.

CUDA is a *device* of the ONNX runtime rather than a runtime of its own,
because that is what it is: an execution provider of the same stack, exactly as
OpenVINO is. A separate backend would be a second copy of `runtimes/onnx_rt.py`.

Install the `[cuda,cudnn]` extras, not just the wheel. `onnxruntime-genai-cuda`
pulls `onnxruntime-gpu`, whose CUDA libraries arrive as separate `nvidia-*`
wheels that unpack to `site-packages/nvidia/<pkg>/lib` — a directory no loader
searches. Keylane dlopens them itself before building a CUDA session, so the
common failure

```
Cuda interface not available: Failed to load library:
libcublasLt.so.13: cannot open shared object file
```

does not happen; without that preload, a complete-looking install is unusable.
It is also why `import torch` before `onnxruntime` is a folk remedy for this —
torch solves the same problem the same way.

Being listed is not being loadable, and Keylane checks both.
`onnxruntime.get_available_providers()` reports CUDA because the provider was
*compiled in*; it says nothing about whether its libraries resolve. Trusting it
produced a Settings panel that offered CUDA and a stack trace when it was
chosen.

**Measured** on an RTX 5090 Laptop (Blackwell, sm_120) with driver 610.57 and
CUDA 13, running Qwen2.5-0.5B through Keylane's own backend:

| Device | Load | Decode |
| --- | --- | --- |
| CUDA | 1.2 s | 83 tok/s |
| CPU | 0.7 s | 173 tok/s |

CPU wins at 0.5B, and that is not a defect: at that size per-token launch
overhead dominates and the GPU never gets to work. The 24 GB is for the models
that do not fit anywhere else.

A GPU is shared, so running out of VRAM is the normal failure rather than an
exotic one — and ONNX Runtime reports it by naming the *last small allocation
it tried*, which on a 24 GB card reads as a bug. Keylane says what it means:

```
not enough free VRAM to load this model on CUDA — gpu-int4-awq-block-128 needs
about 2.3 GB of VRAM and 2141 MiB is free. Close whatever else is using the GPU
(nvidia-smi lists it), or pick a smaller model or another device.
```

> Microsoft publishes the same CUDA build under two names: Phi-3-mini has
> `cuda/cuda-int4-rtn-block-32`, while Phi-3.5-mini and Phi-4-mini have
> `gpu/gpu-int4-awq-block-128`. So a folder called `gpu` in an ONNX repo means
> NVIDIA, while `GPU` as a Keylane device means an Intel GPU through the
> OpenVINO EP — opposite vendors, same three letters. Both are recognised.

Each device is probed by the runtime that owns it. Asking OpenVINO whether CUDA
is available — which is what Keylane used to do for every runtime — reports the
only usable path to a discrete GPU as "not present on this machine".

```
ONNX Runtime GenAI:
   ok NPU  — Intel(R) AI Boost
   NO GPU  — NVIDIA GeForce RTX 5090 (dGPU)   not an Intel device; OpenVINO cannot compile for it
   NO CUDA — NVIDIA GeForce RTX 5090          onnxruntime-genai has no CUDA provider (pip install onnxruntime-genai-cuda)
   ok CPU  — Intel(R) Core(TM) Ultra 9 275HX
```

### Models are listed for the device they would land on

Settings shows the models that suit the selected device and holds the rest
behind **Show all (N not suited to …)**, with the reason on each. Two things
make a model unsuited, and neither makes it a bad model:

- an asymmetric INT4 export with the **NPU** selected — it loads and then runs
  far below the hardware, which is exactly what CPU and GPU are for;
- OpenVINO IR with **CUDA** selected — an NVIDIA GPU needs an ONNX export.

Anything already downloaded, or currently active, is always listed whatever the
recommendation: it is already yours.

To see what a model really costs on your hardware rather than guessing:

```bash
PYTHONPATH=. python scripts/npu-bench.py
```

It measures cold compile, warm load and seconds per `generate()` call at three
reply lengths, and writes them to `data/bench.json`.

On an NPU 3720, with the driver and compiler matched to the runtime:

| | |
| --- | --- |
| Cold compile, 7B | 6.8 s |
| Warm load from cache | 5.5 s |
| Time to first token | ~15 s, flat — it does not depend on reply length |
| Decode | ~0.25 s/token |

If your compile is a minute rather than seconds, the driver and its compiler
are out of step with OpenVINO; `scripts/npu-driver-fix.sh` installs a matched
pair. Be clear about what that buys, though: on the machine these numbers come
from it took compile time from 60 s to 6.8 s and changed time to first token by
nothing at all. It also fixes VLM pipelines, which throw
`ZE_RESULT_ERROR_UNINITIALIZED` on every call against a mismatched stack.

The ~15 s before the first token is the cost that actually shapes a turn: the
agent makes one model call per ReAct iteration, so a turn using three tools
pays it four times.

### Curated models

Pick one in **Settings → Model** or via `POST /models/select`. They auto-download
from Hugging Face on first activation.

| ID | Model | Runtime | NPU |
| --- | --- | --- | --- |
| `jan-nano-npu` | Jan-nano 4B (agentic) | OpenVINO | ✅ |
| `qwen3-8b-cw` | Qwen 3 8B (default) | OpenVINO | ✅ |
| `phi-3.5-mini-cw` | Phi 3.5 Mini | OpenVINO | ✅ |
| `phi-3.5-mini-gq` | Phi 3.5 Mini, group-quantized | OpenVINO | ✅ |
| `mistral-7b-cw` | Mistral 7B Instruct v0.3 | OpenVINO | ✅ |
| `deepseek-r1-qwen-1.5b` | DeepSeek R1 Distill Qwen 1.5B | OpenVINO | ✅ |
| `qwen3-4b-npu` | Qwen 3 4B | OpenVINO | ✅ |
| `qwen2.5-coder-7b-npu` | Qwen 2.5 Coder 7B | OpenVINO | ✅ |
| `gemma-3-4b-vlm` | Gemma 3 4B (vision) | OpenVINO | ✅ |
| `minicpm5-1b-ov` | MiniCPM5 1B (small) | OpenVINO | — |
| `qwen2.5-7b-instruct` | Qwen 2.5 7B Instruct | OpenVINO | — |
| `qwen3-8b` | Qwen 3 8B (asymmetric) | OpenVINO | — |
| `qwen2.5-coder-7b` | Qwen 2.5 Coder 7B (asymmetric) | OpenVINO | — |
| `gemma-2-9b-it` | Gemma 2 9B Instruct | OpenVINO | — |
| `mistral-7b-instruct-v03` | Mistral 7B Instruct v0.3 (asymmetric) | OpenVINO | — |
| `deepseek-r1-qwen-7b` | DeepSeek R1 Distill Qwen 7B (asymmetric) | OpenVINO | — |
| `phi-4-mini-instruct` | Phi 4 Mini Instruct (asymmetric) | OpenVINO | — |
| `qwen3.5-9b` | Qwen 3.5 9B (vision, asymmetric) | OpenVINO | — |
| `minicpm5-1b-onnx` | MiniCPM5 1B (agentic) | ONNX Runtime | CPU |
| `phi-4-mini-onnx` | Phi 4 Mini Instruct | ONNX Runtime | CPU |
| `phi-3.5-mini-onnx` | Phi 3.5 Mini Instruct (AWQ) | ONNX Runtime | CPU |
| `phi-4-mini-reasoning-onnx` | Phi 4 Mini Reasoning | ONNX Runtime | CPU |
| `llama-3.2-3b-onnx` | Llama 3.2 3B Instruct | ONNX Runtime | CPU |
| `mistral-7b-onnx` | Mistral 7B Instruct v0.2 | ONNX Runtime | CPU |
| `phi-4-onnx` | Phi 4 (14B) | ONNX Runtime | CPU |

**The NPU column is not decoration.** Intel's NPU guide requires a *symmetric*
INT4 or NF4 export at group size `-1` or `128`. Most `OpenVINO/*-int4-ov` repos
are `INT4_ASYM`, which the NPU does not support — they load, and then run far
below what the hardware can do. The ✅ rows are symmetric channel-wise (`-cw-`),
symmetric group-quantized (`-gq-`), or exported for the NPU by their publisher;
Settings badges each row so you know before a 4 GB download rather than after.
The rest are fine on CPU and GPU.

Most ONNX entries are `cpu_and_mobile` builds — CPU-targeted graphs with
int8 accumulation. None of those repos ships an OpenVINO NPU build, so that
runtime's device default is CPU. `minicpm5-1b-onnx` is the exception in shape
rather than target: a flat repo with one FP16 graph at the root and no
execution provider baked in.

### Small models that are good at tools

The default is an 8B, and an 8B is the right answer when the question is hard.
Most turns are not hard — they are "search this", "remind me", "what is on my
screen" — and there the 8B's ~15 s to first token is the whole cost of the
turn, paid once per ReAct iteration.

Two entries exist for that case, and both were picked for agentic behaviour
rather than for size alone:

| | |
| --- | --- |
| `jan-nano-npu` | Qwen3-4B tuned for MCP tool use and web research. Symmetric INT4, 2.3 GB, runs on the NPU. Half the default's size. |
| `minicpm5-1b-onnx` | 1B, Apache-2.0, built for on-device agents. 131k context, and the strongest tool use in its class. |

**MiniCPM5 does not speak Keylane's tool dialect, and now it does not have
to.** Keylane's prompt asks for `<tool_call>{json}</tool_call>`; MiniCPM5's
chat template trains it to write

```
<function name="web_search"><param name="question">tide times</param></function>
```

and it writes that whatever the prompt says. A parser that knew only the first
dialect did not fail loudly — every branch missed, the call fell through as
prose, and the model looked as though it had simply declined to use its tools.
Both dialects are parsed now, including the CDATA form the template uses for
values containing markup or newlines.

**A declared context is not a budget.** MiniCPM5-1B declares 131072 tokens,
which is true of the model and useless as an instruction: at that length the
prompt alone is ~340,000 characters, and prefilling it on a 1B CPU model costs
minutes per call. The declared context still bounds generation; what Keylane
is willing to *build* is capped separately, at 8192 tokens
(`KEYLANE_ONNX_MAX_PROMPT_TOKENS`).

**Thinking is asked off by default.** Models with a hybrid `<think>` mode
default to thinking *on*: measured on `minicpm5-1b-ov`, the first tokens of a
plain "Say OK." were `<think>\nHmm, the user is saying…`. One Keylane turn is
several model calls and the reasoning is stripped before anyone sees it, so
those tokens buy nothing here.

Both runtimes now render the export's own template with `enable_thinking=false`
— neither could pass a template *variable* before, and the obvious route
(`AutoTokenizer`) does not survive these exports at all: an OpenVINO conversion
of MiniCPM5 names its tokenizer class `TokenizersBackend`, which transformers
refuses outright over a class the render never needed. The template is read as
text and rendered directly instead.

Be clear about what that buys, though. It is the *request* that is now
possible: the rendered prompt ends with the empty `<think></think>` block the
template emits for `false`, which is what Qwen3-family models take as "skip
it". On the `minicpm5-1b-ov` INT4 conversion the model opens a think block
anyway. Nothing breaks when it does — the reasoning is stripped either way —
but the latency win is per-model, not universal. Set
`KEYLANE_ENABLE_THINKING=1` to stop asking.

> Symmetry is checked by reading the IR, not the repo name. OpenVINO writes
> signed `i4` constants for a symmetric export and unsigned `u4` for an
> asymmetric one, and a repo called `int4-g128` can still be full of `u4` —
> two candidates were rejected on exactly that.

> NF4 needs Lunar Lake or newer. Check your NPU generation with
> `python -c "import openvino; print(openvino.Core().get_property('NPU','DEVICE_ARCHITECTURE'))"`
> — `3720` is the Meteor Lake generation and cannot take NF4.

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

ONNX repos ship several builds of the same model — usually `cpu_and_mobile`
and `gpu`, sometimes `cuda`, `directml` or `qnn` — so the import ranks them and
takes the one that runs on Intel hardware. Pass `"subfolder"` to override the choice. Imported
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

## The screen layer

Three things that happen outside Keylane's own window. All of them are local:
Whisper transcribes, tesseract reads, and nothing is sent anywhere.

### Dictation

Hold-to-talk that types into whatever window has the caret. Bind the commands in
your compositor — Keylane deliberately does not read your keyboard globally:

```
# Sway / Hyprland
bindsym $mod+d exec keylane-dictate    # ui/main.py --dictate
bindsym $mod+g exec keylane-compose    # ui/main.py --compose
bindsym $mod+p exec keylane-point      # ui/main.py --point
```

Press once to start, again to send. `--dictate` types what you said; `--compose`
takes what you said as an *instruction*, looks at the screen, and types the draft.
`--point` lets you circle something and then ask about it.

The transcript goes through a second pass that fixes punctuation and casing. That
pass is checked against the original and **discarded if the words moved** — a model
asked to tidy text will otherwise paraphrase it, and watching your own sentence get
rewritten is worse than an uncapitalised one.

Two safety rules are enforced for every backend, in `inject/base.py`:

- **Return is never synthesised.** Trailing newlines are always stripped, and inside
  a terminal *every* newline is folded to a space — the second line of a two-line
  paste executes as surely as the first.
- **A non-QWERTY layout never gets key positions.** `wtype` and `xdotool` press
  places, not letters, so on Dvorak or Colemak Keylane switches to the clipboard
  route, which carries the text itself and puts your clipboard back afterwards.

**Typing into other windows needs a helper, and which one is not a preference:**

| Session | Backend | Install |
| --- | --- | --- |
| wlroots (Sway, Hyprland, river) | `wtype` | `sudo dnf install wtype` |
| **GNOME / Wayland** | `ydotool` | `sudo dnf install ydotool`, plus `ydotoold` running |
| X11 | `xdotool` | `sudo dnf install xdotool` |

GNOME is the awkward one for the same reason it is awkward for the display backend:
**Mutter has never implemented `zwp_virtual_keyboard_v1`**, which is the protocol
`wtype` uses. `ydotool` goes underneath the display server through `/dev/uinput`
instead, so it needs its daemon running and your user in a group that can open that
device. Without one of these three, dictation transcribes and then tells you it
could not type — and puts the text on your clipboard so nothing is lost.

Settings → Screen reports which backend was selected, or that none was.

### Drawing on the screen

`screen_annotate` puts rings, arrows, boxes and labels over the live desktop, and
`walkthrough_show` runs up to fifteen of them in sequence. The surface is a
full-screen layer with an **empty input region**, so the desktop underneath stays
completely clickable while the marks sit on top of it.

Strokes are drawn with a seeded wobble rather than true. That is not whimsy: a
geometrically perfect ring over someone's desktop reads as a compositor glitch,
and an uneven one reads as somebody pointing.

The part that decides whether this is useful is **how a target is located**. The
model is asked to name the control — `target_text: "Export"` — not to estimate its
coordinates, and Keylane finds it with tesseract:

```bash
sudo dnf install tesseract
```

Without tesseract the model's own coordinates are used and land visibly worse; a 4B
vision model does not ground UI positions reliably. When a named target is not on
screen, the tool says so rather than drawing in the wrong place, so the model can
tell you it could not find it.

Walkthrough steps persist until they are passed. Advance one with `keylane-next`
(`ui/main.py --next`), which you can bind to a key, or by asking Keylane to move on.
There is no automatic click detection: Wayland gives no global click events without
an input grab, and Keylane does not take one.

### Dropping files

Drop a file anywhere on the Spotlight bar, the orb or the answer card. Images attach
as images; PDFs, HTML, Markdown, code and plain text are read and carried into the
turn as `<attached_file>` blocks, explicitly framed as data rather than instructions.
Reading PDFs needs `pypdf`; everything else works without it.

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

- **The daemon requires a token.** It listens on loopback, and loopback is not
  a boundary: any web page you visit can `fetch("http://127.0.0.1:9100/…")`,
  and every other user on the machine can reach it too. So a token is generated
  on first run into `data/settings.json` (mode 0600) and required on every route
  but `/health`, as `X-Keylane-Token` or `Authorization: Bearer`. There is no
  CORS policy, and a request carrying an `Origin` header is refused outright —
  a browser attaches that header and a script cannot remove it, so a malicious
  page fails even if it somehow read the token.

  ```bash
  TOKEN=$(python -c 'import json;print(json.load(open("data/settings.json"))["security"]["api_token"])')
  curl -s -H "x-keylane-token: $TOKEN" localhost:9100/memories | jq
  ```

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

Keylane ships six themes, each with a light and a dark scheme:

| Theme | Looks like |
| --- | --- |
| `glass-console` | Dark glass over the desktop, one cyan accent, hairline structure. The default. |
| `paper-terminal` | Ink on warm paper: flat surfaces, hairline rules, serif answers, monospace labels. |
| `aurora` | Translucent material, no borders, large radii, violet and cyan light. |
| `copper-oxide` | Oxidised copper and verdigris over brown-black. Warm, square, hard shadows. |
| `nord-frost` | The Nord palette — Polar Night and Snow Storm — so Keylane matches a Nord desktop. |
| `high-contrast` | Opaque black and white, 2px borders, no translucency. WCAG AAA text. |

`high-contrast` exists because the other five all rely on translucency,
hairlines and mid-grey text, and each of those is what fails for low vision or
a glossy screen in daylight. Every shipped theme is checked against WCAG AA for
text on its own surface; `high-contrast` is checked against AAA.

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

## Keeping current

Keylane reads its own GitHub *releases* — not the source. That distinction is
the whole of how the version check goes wrong: bumping `updater/version.py`
changes what a build calls itself and changes nothing an installed copy can
see. Between 0.5.0 and 0.7.0 the version was bumped three times and tagged
none, so every installed copy went on reporting nothing to install, correctly
and uselessly.

So a release is one command, and it does all three parts:

```bash
scripts/keylane-release 0.7.1              # bump, tag, publish
scripts/keylane-release --dry-run 0.7.1    # show what it would do
```

It refuses to go backwards, refuses a tag that exists, and refuses a dirty
tree. A build that is ahead of every published release now says so —
`keylane-update` and Settings → About both distinguish "you are current" from
"you are newer than anything released", which used to render identically.

It looks once a day, drops a note in the inbox, and stops — it never installs
anything on its own.

```bash
keylane-update                 # is there one?
keylane-update --install       # get it
keylane-update --channel main  # follow the branch instead of releases
keylane-update --rollback      # go back to the previous release
```

Or **Settings → About**, which shows the version, the channel, the release
notes and an Install button that asks first.

`install.sh` lays the install out so an update is safe:

```text
~/.local/share/keylane/
  releases/<tag>/        one version, never modified in place
  current -> releases/…  what the systemd units follow
  data/                  memories, models, settings — outside the releases
  .venv/                 outside too, so a rollback keeps its dependencies
```

Updating unpacks a new directory beside the old one and moves the symlink;
rolling back moves it back. `data/` is never inside the part being replaced. A
git checkout takes the other path — fetch and fast-forward, refused on a dirty
tree. Downloads are HTTPS-only, pinned to GitHub, and checked against the
sha256 published with the release; when a release publishes none, Keylane says
the download was only trusted as far as HTTPS rather than implying otherwise.

Set `permissions.update_apply = "deny"` to turn in-app updating off entirely.

## In your status bar

The first download and compile of a model takes minutes and used to happen
invisibly. `keylane-status` puts it in the bar:

```jsonc
// ~/.config/waybar/config
"custom/keylane": {
  "exec": "~/.local/share/keylane/current/scripts/keylane-status --waybar --watch",
  "return-type": "json",
  "on-click": "keylane-toggle"
}
```

It reports the model, the device, compile progress while a model is loading,
and the unread count from the inbox. Without `--waybar` it prints one plain
line, which is enough for tmux or a shell prompt.

## Using Keylane's model from other tools

Keylane speaks the OpenAI chat-completions API outward, to reach a larger model
on a GPU. It also speaks it *inward*: the resident NPU model is available to
anything else on the machine, at no VRAM cost, because it is already loaded.

```bash
curl -s localhost:9100/v1/chat/completions \
  -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hello"}], "stream": true}'
```

Point any OpenAI client at `http://127.0.0.1:9100/v1` and give it the token as
the API key. This is the raw model — no tools, no memory, no research. Anything
that wants the assistant should use `/chat/stream` and read its events.

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
| `POST /dictation/cleanup` | Punctuate a transcript without rewording it |
| `POST /compose` | Draft text from a spoken instruction plus a screenshot |
| `GET /walkthrough` · `POST /walkthrough/advance` | Step state, and moving through it |
| `POST /permissions/respond` | Approve/deny tool permission prompts |
| `GET /update/status` · `POST /update/check` · `POST /update/apply` | Version, check, install |
| `GET /v1/models` · `POST /v1/chat/completions` | The resident model, OpenAI wire format |

## Architecture

```text
Super+Space → GTK Spotlight → floating orb → answer HUD (click-through)
  --dictate → Whisper → inject/ → the caret in someone else's window
  screen_annotate → control socket (:9101) → OCR → overlay (click-through)
                    ↓
              FastAPI daemon (:9100)
                    ↓
         AIAgent (ReAct + tools + MCP)
                    ↓
              capability seams (seams/)
      llm · web · skills · jobs · subagents · goals · spill
                    ↓
   NPU via runtimes/  ·  optional GPU model over the OpenAI API
   inject/ (wtype · ydotool · xdotool · clipboard) · vision/ (tesseract)
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
