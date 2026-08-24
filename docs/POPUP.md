# The popup and the tray

Keylane has two pieces of desktop presence: an overlay you summon with a
keystroke, and a taskbar icon that tells you when it is working.

## The popup

Press <kbd>Super</kbd>+<kbd>Space</kbd>. A chromeless bar appears above the
centre of the screen, focused and ready. Type, press <kbd>Enter</kbd>, and it
grows downward to show the answer.

| Key | Action |
| --- | --- |
| <kbd>Super</kbd>+<kbd>Space</kbd> | Show the popup, or hide it if it is already up |
| <kbd>Enter</kbd> | Send |
| <kbd>Ctrl</kbd>+<kbd>Enter</kbd> | Send and hide — the work continues in the background |
| <kbd>Esc</kbd> | Close |
| Microphone | Click to start dictating, click again to stop |
| `/` | Show the skill list; arrows to move, Tab or Enter to pick |
| Click away | Close (unless the theme disables it) |

The bar **closes** when it loses focus — it does not linger in the background.
Every Super+Space opens a fresh bar that has re-read the active theme.

The microphone button is a toggle: click it to start dictating, click again to
stop. Whisper transcribes locally and appends the text to whatever is already in
the field. A recording you forget about stops itself after two minutes.

## Calling a skill

Type `/` and the enabled skills appear below the bar, filtered as you keep
typing. Arrow keys move, Tab or Enter picks one, Escape closes the list
without closing the popup.

Only enabled skills are listed — install and switch them on under
**Control panel → Skills**.

## The device chip

A quiet chip in the bar shows which device the control plane is actually
running on. Click it to choose: Auto, NPU, GPU or CPU — whatever this machine
exposes, named as the hardware reports itself.

It turns amber when the device in use is not the one you asked for, which
happens when a model will not compile on your first choice. Hovering says why.

## What happens after you press Enter

The bar gets out of the way. It closes, and a small **orb** appears in a screen
corner and spins while the work runs — so a Claude Code job that takes two
minutes does not pin an input field to your screen.

While it works, three arcs orbit a pulsing core — drawn with Cairo rather
than a stock spinner, and centred in the circle. **The colour says what it is
doing**: indigo while the model is deciding, teal running a local tool, violet
once a worker has it, sky while checking the result, amber waiting for your
approval, cyan reading aloud, then green or red. Colours ease between states
rather than cutting, so a fast tool call does not strobe. They run at unequal rates so
the figure never visibly repeats. With animations switched off in your desktop
settings it becomes a slow opacity breath instead.

When the answer arrives the orb **expands into a squircle** showing the result.
It grows from the orb rather than appearing centre-screen, so it is obvious
where it came from, and the growth is a critically damped spring: no bounce,
because nothing here was thrown. Clicking away or pressing <kbd>Esc</kbd>
collapses it, interruptibly, from wherever the animation currently is.

The panel is sized to its answer: a one-line reply gets a narrow squircle, a
table gets the full width. It **closes itself** after about nine seconds —
twenty for a longer answer — and the countdown **pauses while your pointer is
over it**, so reading is never interrupted. A request waiting for your approval
never times out.

If the assistant needs approval for something, the orb expands into an
Allow / Cancel choice instead — you never have to reopen the bar to approve.

## Read aloud

With speech enabled, answers carry a speaker button. Tables and command output
are described rather than spelled out — reading a `df` listing cell by cell is
worse than useless — and the auto-dismiss waits until it has finished speaking.

Turn it on under **Assistant → Read aloud**, where you also pick the engine and
voice. See [Speech](speech.html).

Choose the corner in **Control panel → Gateway → Result panel corner**:
`top-right` (the default), `top-left`, `bottom-right`, `bottom-left` or
`center`.

### Staying on top

The orb and the answer stay above other windows, whatever you switch to.

How that is achieved depends on the compositor, because Wayland deliberately
gives clients no control over stacking:

| Compositor | Mechanism |
| --- | --- |
| sway, Hyprland, other wlroots | `gtk4-layer-shell` overlay layer |
| GNOME (Mutter) | XWayland plus `_NET_WM_STATE_ABOVE` |

Mutter implements neither layer-shell nor any Wayland stacking protocol, so on
GNOME the launcher runs through XWayland — that is the only route that works
there. Set `KEYLANE_BACKEND=wayland` to override, at the cost of the panel no
longer staying on top.

`wmctrl` makes the X11 path more reliable; without it Keylane falls back to
`xprop`.

> **Note**: Exact corner placement needs `gtk4-layer-shell`. Without it Keylane
> uses a screen-sized transparent window with its input region clipped to the
> orb, which reaches the corner and still lets every click outside pass
> through to whatever is underneath.

## Answers are canvases, not walls of text

The result panel renders a **canvas** — a small structured document of stat
tiles, tables, callouts, lists and code blocks — rather than raw JSON or
markdown. The assistant is asked to produce one; anything it returns as prose
is wrapped in a minimal canvas so the layout is consistent either way.

See [Canvas answers](canvas.html) for the block types and how to produce one
from a plugin.

It is not a window in the usual sense: no title bar, no taskbar entry, no
maximise. It is a Spotlight-style overlay, and its exact shape belongs to the
active theme — see [Writing themes](THEMES.md). A theme can make it a bar, a
panel, a full window or a corner orb, and switching themes changes it
immediately with nothing to restart.

### Binding the hotkey

The launcher is a single-instance application, so "run it again" means "toggle
it". Bind that to whatever key you like.

**GNOME** — Settings → Keyboard → View and Customise Shortcuts → Custom
Shortcuts → **+**

| Field | Value |
| --- | --- |
| Name | `Keylane` |
| Command | `~/.local/share/ai-gateway/.venv/bin/python ~/.local/share/ai-gateway/launcher/main.py --toggle` |
| Shortcut | <kbd>Super</kbd>+<kbd>Space</kbd> |

`scripts/install.sh` offers to set this up for you.

> **Note**: On GNOME, <kbd>Super</kbd>+<kbd>Space</kbd> is bound by default to
> "Switch input source". The installer clears that binding when you accept;
> otherwise remove it yourself, or pick another key such as
> <kbd>Super</kbd>+<kbd>K</kbd>.

**sway / Hyprland** — bind it in your config:

```text
# sway
bindsym $mod+space exec ~/.local/share/ai-gateway/.venv/bin/python \
    ~/.local/share/ai-gateway/launcher/main.py --toggle
```

### Positioning

On wlroots compositors (sway, Hyprland, river) Keylane takes a
`gtk4-layer-shell` overlay surface and honours the theme's `position`,
`offset_x` and `offset_y` exactly, including corner anchoring for the orb.

GNOME's Mutter has no layer-shell protocol. There the popup is an undecorated
window that Mutter centres, and Keylane reproduces a vertical offset by padding
the opposite side with transparent space — so the Spotlight bar still floats
above centre. Corner positions fall back to centred.

Install `gtk4-layer-shell` if your compositor supports it:

```bash
sudo dnf install gtk4-layer-shell
```

## The tray indicator

The taskbar icon is how you know Keylane is doing something when the popup is
closed — a delegated Claude Code run can take minutes, and you should not have
to keep the popup open to find out how it is going.

| State | Icon | Meaning |
| --- | --- | --- |
| Idle | Outline keyhole | Gateway up, nothing running |
| Busy | Filled, pulsing | One or more tasks in flight |
| Attention | Exclamation | A task is waiting for your approval |
| Offline | Struck through | The gateway is not answering |

Clicking the icon opens a menu with the current task, plus shortcuts to the
popup, the control panel and this handbook. Hovering the status line shows what
is running and which worker has it.

When a task starts waiting for approval you also get a desktop notification, so
a confirmation prompt cannot sit unnoticed behind other windows.

### How it works

The tray subscribes to `/api/events`, a Server-Sent Events stream carrying a
fresh activity snapshot on every state change, and falls back to polling
`/api/activity` if the stream cannot connect. Either way it reconnects on its
own when the gateway restarts.

```bash
# See what the tray sees
curl -s http://127.0.0.1:9100/api/activity | python -m json.tool
curl -N  http://127.0.0.1:9100/api/events
```

### Requirements

The indicator uses the Ayatana/AppIndicator protocol, which KDE, Xfce, Budgie
and most panels speak natively. **GNOME needs an extension:**

```bash
sudo dnf install libayatana-appindicator-gtk3 gnome-shell-extension-appindicator
```

Then log out and back in, and enable **AppIndicator and KStatusNotifierItem
Support** in the Extensions app.

Without a tray host the popup still works normally — you just lose the icon, and
Keylane says so in the launcher log rather than failing.

> **Note**: The tray runs as a separate process from the popup. AppIndicator is
> a GTK 3 library and the popup is GTK 4, and the two cannot share a process.
> `launcher/main.py` starts both; `--no-tray` skips the indicator and
> `--tray` runs only it.

## Running the launcher

```bash
python launcher/main.py             # popup plus tray (what the service runs)
python launcher/main.py --toggle    # show/hide — bind this to your hotkey
python launcher/main.py --no-tray   # popup only
python launcher/main.py --tray      # tray only
python launcher/main.py -v          # verbose logging
```

As a user service:

```bash
systemctl --user enable --now ai-launcher.service
systemctl --user status ai-launcher.service
journalctl --user -u ai-launcher.service -f
```

### Environment variables

| Variable | Effect |
| --- | --- |
| `KEYLANE_GATEWAY` | Gateway base URL (default `http://127.0.0.1:9100`) |
| `KEYLANE_APP_ID` | Override the single-instance app id — lets a development checkout run beside an installed copy |
| `KEYLANE_NO_TRAY` | Set to skip the tray even without `--no-tray` |

## Troubleshooting

**The popup does not appear.** Check the gateway is up
(`curl http://127.0.0.1:9100/healthz`) and run the launcher in a terminal with
`-v` to see the error.

**The hotkey does nothing.** Something else owns it. On GNOME check Settings →
Keyboard for a conflicting binding, especially "Switch input source".

**It appears in the wrong place.** Expected on GNOME for anything but a centred
position — see *Positioning* above.

**No tray icon.** Install the AppIndicator extension (above) and log out and in.

**The popup opens but shows the wrong shape.** The theme decides that. Check
`curl -s http://127.0.0.1:9100/api/themes/active/popup.json`.
