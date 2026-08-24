# Install, update and uninstall

Keylane targets Fedora with an Intel Core Ultra NPU, but it runs anywhere GTK 4
and Python 3.11+ do — without an NPU it falls back to CPU and to keyword
routing.

Everything lives in two places:

| Path | Holds |
| --- | --- |
| `~/.local/share/ai-gateway` | The installed program, its virtualenv, your config, models, themes and skills |
| `~/.config/systemd/user` | The two user services |

The git checkout you clone is only the source; the installer copies from it.

## Install

```bash
git clone https://github.com/dxbh0517/keylane.git
cd keylane
./scripts/install.sh
```

The script:

1. Installs the Fedora GTK, PyGObject, AppIndicator and audio packages.
2. Copies the tree to `~/.local/share/ai-gateway`.
3. Builds a virtualenv with `--system-site-packages` so PyGObject is importable.
4. Builds the handbook into `web/docs`.
5. Installs the user services, the desktop entry and the icons.
6. Enables the AppIndicator shell extension so the tray icon has somewhere to live.
7. Offers to bind <kbd>Super</kbd>+<kbd>Space</kbd>, clearing whatever held it before.

Then start it:

```bash
systemctl --user enable --now ai-gateway.service ai-launcher.service
xdg-open http://127.0.0.1:9100/
```

| | |
| --- | --- |
| Control panel | <http://127.0.0.1:9100/> |
| Handbook | <http://127.0.0.1:9100/docs/> |
| API explorer | <http://127.0.0.1:9100/api-docs> |

### Verify it worked

```bash
systemctl --user is-active ai-gateway.service ai-launcher.service   # active, active
curl -s http://127.0.0.1:9100/api/status | python3 -m json.tool
```

Press <kbd>Super</kbd>+<kbd>Space</kbd> — the popup should appear. The tray icon
should be in your top bar.

## Update

Pull the new source and re-run the installer. **It never touches your data:**
`models/`, `config/`, `themes/`, `skills/` and `plugins/community/` are excluded
from the sync and only seeded with files that are missing.

```bash
cd keylane
git pull
./scripts/install.sh --update
```

`--update` reuses the existing virtualenv and skips the system packages, which
takes seconds instead of minutes. Drop the flag after a release that changes
`requirements.txt`, or if a dependency looks broken.

The installer stops the running services first, so nothing is reading files
while they move. Restart afterwards:

```bash
systemctl --user restart ai-gateway.service ai-launcher.service
```

> **Note**: Updating never overwrites `config/*.toml`. If a release adds a new
> setting, the file is left alone and the new setting takes its default — the
> control panel will show it. Delete a config file and re-run the installer to
> get the shipped version back.

### Updating just the docs or the logo

Both are generated, and both have a script:

```bash
python scripts/build_docs.py    # rebuild web/docs from docs/*.md
python scripts/make_logo.py     # regenerate every icon from one definition
```

## Uninstall

```bash
cd keylane
./scripts/uninstall.sh
```

That stops and removes the services, the desktop entries, the icons and the
keyboard shortcut, hands <kbd>Super</kbd>+<kbd>Space</kbd> back to GNOME, and
removes the program — **keeping** `models/`, `config/`, `themes/`, `skills/`,
`outputs/` and `plugins/` so a reinstall picks up where you left off.

To remove everything, downloaded weights included:

```bash
./scripts/uninstall.sh --purge
```

Add `-y` to skip the confirmation prompt.

### By hand

If you would rather not run the script:

```bash
systemctl --user disable --now ai-gateway.service ai-launcher.service
rm -f  ~/.config/systemd/user/ai-{gateway,launcher}.service
rm -f  ~/.config/systemd/user/*.target.wants/ai-{gateway,launcher}.service
systemctl --user daemon-reload

rm -f  ~/.local/share/applications/{app.keylane.Launcher,keylane}.desktop
rm -f  ~/.config/autostart/keylane-panel.desktop
find   ~/.local/share/icons/hicolor -name 'keylane*' -delete
gtk-update-icon-cache -f ~/.local/share/icons/hicolor

rm -rf ~/.local/share/ai-gateway
```

Then remove the shortcut in **Settings → Keyboard → View and Customise
Shortcuts → Custom Shortcuts**.

## Enable the NPU

Fedora ships the `intel_vpu` kernel driver, but OpenVINO only sees an NPU once
the Level Zero userspace is installed and you are in the `render` group:

```bash
sudo dnf install intel-npu-driver oneapi-level-zero
sudo usermod -aG render "$USER"
# log out and back in
```

Check it:

```bash
python scripts/check_npu.py
curl -s http://127.0.0.1:9100/api/status | python -m json.tool
```

`"npu": true` with `"openvino_devices": ["CPU", "GPU", "NPU"]` means you are set.
Anything else, the `npu_detail` field says exactly what is missing.

## Download a router model

Until an OpenVINO export is loaded, the assistant recognises only a handful of
obvious requests. To get the real plan–act–verify loop:

**Control panel → Models**. The *Recommended for your hardware* list has a
**Download** button on anything you do not already have — it fetches straight
into the right folder and shows live progress. Then choose it under **Router
model**, press **Save**, and **Reload models now**.

Some repositories are marked **Get access** instead: those are gated on Hugging
Face and need their licence accepted on the model page first. A download button
there would only return a 401.

For anything not on the list, use **Search Hugging Face** with target
**Router** — something like `qwen2.5 1.5b instruct int4 openvino`.

An INT4 OpenVINO export of a 1.5B–3B instruct model is the sweet spot: small
enough for the NPU to run in well under a second, capable enough to plan.

## Connect your AI tools

Everything is a [plugin](PLUGINS.md), and each one is optional.

| Tool | How to make it work |
| --- | --- |
| **LM Studio** | Start the local server; Keylane finds it on `127.0.0.1:1234`. Chat models downloaded from the Models tab land in LM Studio's own folder, so they appear there without extra setup. |
| **Lemonade** | Start Lemonade Server; default `127.0.0.1:13305` |
| **Claude Code** | `npm i -g @anthropic-ai/claude-code`, then `claude` once to sign in |
| **Cursor** | Install the Cursor Agent CLI and sign in |
| **ComfyUI** | `pip install "comfy-cli>=1.14.0" comfy-mcp`, `comfy install`, `comfy launch` |

Check **Status** — each one turns green when Keylane can reach it. Anything you
do not use, disable on the **Plugins** tab.

## Set the project sandbox

Coding workers and the file tools can only touch directories you list. Under
**Gateway → Allowed project roots**, one path per line:

```text
~/Documents/Code
~/work
```

Nothing outside these — nor `.ssh`, `.gnupg`, `.env` and similar inside them —
is reachable. This is enforced in Python, not requested of the model.

## Configure the assistant

On the **Assistant** tab, decide what the on-device model may do:

- **Ask before running** — `sensitive` is a good default: reads happen freely,
  anything that changes something asks first.
- **Web search** — DuckDuckGo works out of the box; point it at a self-hosted
  SearXNG if you would rather.
- **Shell commands** — on by default with a read-oriented allowlist. Trim it, or
  switch the tool off entirely.
- **Email** — off until you add SMTP details. Use an app password, or write
  `env:KEYLANE_SMTP_PASSWORD` and put the secret in the service environment.

## Local-only mode

**Gateway → Local only** blocks every cloud worker. Claude Code and Cursor
disappear from routing and from the assistant's catalogue, so nothing leaves the
machine.

## Autostart

**Models → Startup** toggles the user services for the gateway, the popup and
the control panel at login.

## Where downloaded models live

Downloads go into the **install**, not the source checkout:

```text
~/.local/share/ai-gateway/models/router/    OpenVINO exports Keylane runs
~/.lmstudio/models/<publisher>/<repo>/      GGUF chat models, for LM Studio
```

Config stores these as relative paths (`./models/router/…`) so a config file
stays portable, but `./` means the install directory. If you have the source
checked out, its `models/` folder stays empty — that is expected. The Models
tab shows the absolute path so there is no guessing.

Upgrading never touches them; `models/` is excluded from the sync.

## Layout

```text
app/            gateway: routing, tools, assistant, plugins, themes
  tools/        the assistant's capabilities
  plugins/      plugin contracts, registry, built-ins
  npu/          OpenVINO pipelines, router, verifier
  workers/      worker implementations
launcher/       GTK 4 popup, GTK 3 tray, entry point
web/            control panel, and the built handbook under web/docs
docs/           handbook source (markdown — edit these)
themes/         installed themes; built-ins regenerate on start
skills/         your markdown instruction packs
plugins/community/   installed community plugins
config/         workers.toml, models.toml, plugins.toml, assistant.toml, themes.toml
models/         router/ chat/ comfyui/ — downloaded weights
```

## Development

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 9100
python launcher/main.py -v
pytest -q
python scripts/build_docs.py          # rebuild web/docs from docs/*.md
```

To run a checkout beside an installed copy, give it its own port and app id:

```bash
uvicorn app.main:app --port 9199
KEYLANE_GATEWAY=http://127.0.0.1:9199 \
KEYLANE_APP_ID=app.keylane.LauncherDev \
python launcher/main.py
```

## Getting the router onto the NPU

Keylane falls back to the GPU when the NPU cannot compile a model, and says so
in the panel. On Fedora the usual cause is a version mismatch, because the NPU
compiler ships **inside the NPU driver**, not with the OpenVINO package, and
Fedora versions the two separately:

| Component | Fedora ships | Built for |
| --- | --- | --- |
| `intel-npu-driver` | 1.32.0 | OpenVINO 2026.0 |
| `intel-npu-compiler` | 2025.1.0 | OpenVINO 2025.1 |
| `openvino` (pip) | 2026.3.0 | — |

Three components, three generations. Nothing compiles, and the error names a
configuration key (`NPU_MAX_TILES`) rather than the cause. An eight-element
addition fails just as surely as a language model, which is the quickest way to
confirm this is what you are looking at:

```bash
python -c "
import openvino as ov, openvino.opset13 as op, numpy as np
x = op.parameter([1,8], ov.Type.f32)
m = ov.Model([op.add(x, op.constant(np.ones((1,8), np.float32)))], [x])
ov.Core().compile_model(m, 'NPU'); print('NPU compiles fine')"
```

To fix it:

```bash
scripts/npu-driver-fix.sh --dry-run   # see exactly what it will do
scripts/npu-driver-fix.sh
```

It installs Intel's matched compiler and Level Zero backend into
`/usr/local/lib`, ahead of Fedora's in the loader path — Fedora's files are
left alone, and deleting the three installed libraries plus `ldconfig` reverts
it. It then pins the gateway's OpenVINO to the release the driver was built
for.

**Model size matters.** On a Meteor Lake NPU a 1.5B int4 export compiles in
about ten seconds. A 3.8B export may not finish at all. Prefer a small router
model for the NPU and leave larger models on the GPU.
