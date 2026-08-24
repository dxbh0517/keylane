# What Keylane still needs

Written after building v0.2-beta, from what actually broke and what the code
makes hard. Ordered by how much each would change the experience, not by how
interesting it is to build.

Every item says *why* — a list of features without reasons is a wish list.

---

## 1. The NPU path does not work on this hardware

**The problem.** OpenVINO GenAI's NPU pipeline sets `NPU_MAX_TILES`, which the
installed Level Zero driver does not accept. Disabling NPUW gets past that and
then compilation fails with `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE`. The NPU is
detected and healthy; the LLM path is the mismatch. Keylane currently falls back
to GPU, which works but is not the product's premise.

**What to do.**

- Ship a **compatibility probe** — try to compile a tiny model on the NPU at
  first run, record the result, and say plainly in the panel whether the NPU is
  usable for LLMs on this machine rather than making the user read a stack
  trace.
- Test against `openvino-genai` versions pinned to the Level Zero driver, and
  publish a known-good matrix.
- Support **NPU-targeted exports** explicitly (static shapes, channel-wise
  int4). A model tagged `-npu` should be preferred for the router.
- Consider **OpenVINO plain `Core.compile_model`** for the router instead of
  GenAI's LLM pipeline. The router emits one small JSON object; it does not need
  a full chat pipeline, and a plain IR compile is far more likely to load.

## 2. Model downloads are opaque and fragile

**The problem.** A download reports 5% and 0 bytes for its entire life, so a
stall is indistinguishable from progress — that is exactly what made a
half-finished model look like a broken NPU for an hour. Nothing resumes
automatically, and a job whose thread dies stays "running" forever.

**What to do.**

- Real progress: poll the destination size against the expected total from
  `HfApi.model_info(files_metadata=True)`.
- A watchdog that marks a job failed when its worker is gone, and a **Resume**
  button — `huggingface_hub` already resumes from the `.incomplete` file, so
  this is UI, not plumbing.
- Verify after download: check the `.bin` sizes against the manifest before
  declaring success.
- Free-space check before starting. A 9B model on a full disk fails late and
  confusingly.

## 3. The assistant needs a bigger model to be reliable

**The problem.** A 1.5B model, even with a 960-token prompt and a tolerant
parser, gets the tool right maybe three times in four. It also invents argument
names and occasionally answers the question instead of calling the tool.

**What to do.**

- **Constrained decoding.** OpenVINO GenAI supports structured output; forcing
  the reply to match the JSON schema would remove the whole class of parse
  failures rather than mopping them up.
- **Few-shot examples in the prompt**, chosen by request type. Three concrete
  examples beat a page of rules for a small model.
- **A tool-choice pre-filter**: embed the request, shortlist the five most
  relevant tools, and show only those. Keeps the prompt tiny as the catalogue
  grows.
- Let the user pick a **larger router model** and run it on the GPU knowingly —
  a 7B on a 5090 would be both fast and reliable, and the "NPU control plane"
  idea should be a preference, not a constraint.

## 4. Multi-turn conversation

**The problem.** Every request starts from nothing. "Open Firefox" then "now
make it fullscreen" cannot work — the second request has no idea what "it" is.

**What to do.** Keep a short rolling history per session (last N turns plus the
tool observations), expose it in the popup as a thread, and let the user clear
it. This is the single biggest gap between Keylane and something people would
use all day.

## 5. Streaming

**The problem.** The result orb spins with no detail until everything finishes.
A delegated Claude Code run can take minutes with nothing to show.

**What to do.** Stream assistant steps to the orb over the existing SSE channel
— "searching the web", "asking Claude Code", "checking the result" — and stream
worker stdout for long jobs. The activity bus already carries the events; the
orb just needs to subscribe.

## 6. Approvals are all-or-nothing

**The problem.** Danger levels are per-tool and global. Approving
`run_command` once approves the concept, not that command. There is no
"always allow `git status`" and no audit trail.

**What to do.**

- Remember approvals per `(tool, argument-shape)` with an expiry.
- An **audit log** of every tool call with arguments and result, visible in the
  panel. Right now a tool runs and leaves no durable record.
- Per-project policy: stricter rules outside a project root than inside it.

## 7. The tray is the only sign of background work

**What to do.** Desktop notifications on completion (opt-in per request),
a small history of recent answers reachable from the tray, and the ability to
re-open a past canvas. Answers currently vanish when the orb closes.

## 8. Themes cannot restyle the result panel independently

**The problem.** The canvas inherits the popup's palette but has no dedicated
knobs — a theme cannot say "stat tiles are wide" or "tables are compact".

**What to do.** Add a `[canvas]` section to the theme manifest mirroring
`[popup]`, and let a theme ship `canvas.css`.

## 9. Plugins have no isolation and no versioning

**The problem.** A community Python plugin runs in-process with the user's full
permissions, and nothing checks it. There is no version constraint, no
signature, no update path — install is a one-way door.

**What to do.**

- A `keylane_version` constraint in `plugin.toml`, checked on install.
- Update-in-place from the source the plugin came from.
- Run community Python plugins in a subprocess with a restricted environment,
  the way MCP plugins already are. MCP is the safer model and should be the
  documented default for anything third-party.
- Sign the catalog so a shipped plugin can be verified.

## 10. No test coverage for the GTK layer

**The problem.** 133 tests cover the gateway thoroughly and the launcher not at
all. Every popup bug this cycle — the 230px logo, the uninitialised
`_layer_shell_active`, the dead click zone — reached the user because nothing
exercises the widget tree.

**What to do.** Headless GTK tests that build the popup and the orb, assert
allocations and input regions, and drive the mic toggle. The measurement
scripts written while debugging are most of the way there; they should be
tests.

## 11. Operational gaps

- **Backup and restore** of `config/`, `skills/` and installed plugins as one
  archive. Reinstalling currently means reconfiguring.
- **A real log view** in the panel. `journalctl` should not be the only way to
  see why something failed.
- **Health checks are serial** — six plugins each with a timeout makes
  `/api/status` slow. Run them concurrently with a short deadline.
- **Config file watching**, so hand edits apply without a restart.

## 12. Reach

- **Wayland global hotkey** via the XDG GlobalShortcuts portal, so the hotkey
  works without GNOME custom keybindings.
- **KDE and Xfce** verification. Only GNOME and wlroots are tested.
- **A CLI** (`keylane ask "..."`) for scripting and for people who live in a
  terminal.
- **Mobile or remote access** over an authenticated tunnel — the gateway is
  already an HTTP API, and being able to ask your desktop something from your
  phone is the natural extension.

---

## Deliberately not doing

- **Cloud sync of settings.** Keylane is local-first; syncing config would mean
  an account and a server.
- **A plugin marketplace.** A git repository and the catalog format are enough;
  a marketplace is a moderation problem, not a feature.
- **Replacing the verifier with the router model.** Separate models for doing
  and for judging is the point — one model marking its own work is how you get
  confident wrong answers.
