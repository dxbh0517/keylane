# Writing themes

A Keylane theme controls three surfaces:

| File / section | Controls |
| --- | --- |
| `[popup]` in `theme.toml` | **The shape of the popup** — bar, panel, window or orb, plus size, position and what it shows |
| `launcher.css` | GTK styling for the popup |
| `web.css` | The control panel |

The `[popup]` section is the important one. It is what lets a theme turn the
same launcher into a macOS-Spotlight bar, a full assistant window, or a small
orb parked in the corner of the screen — without touching any code.

## Package layout

```text
my-theme/
  theme.toml       required
  launcher.css     optional — generated from [colors] if absent
  web.css          optional — generated from [colors] if absent
  popup.css        optional — extra GTK CSS layered on top of launcher.css
```

Zip the folder (with `theme.toml` inside the archive) and install it from
**Control panel → Themes**, or drop the folder into `themes/<id>/`.

## theme.toml

```toml
id = "aurora"
name = "Aurora"
author = "you"
version = "1.0.0"
description = "Cool slate with an emerald accent, as a floating bar."

[colors]
bg      = "#0b1220"
surface = "#121a2b"
text    = "#e8eefc"
muted   = "#9aa8c7"
accent  = "#34d399"
border  = "#243047"
danger  = "#fb7185"

[popup]
preset  = "spotlight"
width   = 720
offset_y = -160
font_size = 20
```

### Colours

Seven roles, all required if you want generated CSS:

| Key | Used for |
| --- | --- |
| `bg` | The popup shell and the panel background |
| `surface` | Cards, the entry field, results |
| `text` | Primary text |
| `muted` | Secondary text, hints, inactive chips |
| `accent` | Focus rings, active states, the orb, primary buttons |
| `border` | Hairlines and outlines |
| `danger` | Errors and destructive actions |

Keep text on `bg` and on `surface` above 4.5:1 contrast. One theme is one mode —
ship a light theme and a dark theme as two ids rather than trying to do both.

## The `[popup]` section

### Presets

Start from a preset, then override anything:

| Preset | Shape |
| --- | --- |
| `spotlight` | One chromeless search bar, centred, floating above the middle of the screen. Grows downward as results arrive. |
| `panel` | The bar plus status chips, a project picker and hints. |
| `window` | A conventional decorated window with a title bar and scrollback. Does not dismiss on focus loss. |
| `orb` | A small circle in a screen corner that expands into a panel when clicked or when the hotkey fires. |

```toml
[popup]
preset = "window"
width = 900
```

### Every key

| Key | Type | Default | What it does |
| --- | --- | --- | --- |
| `mode` | `bar` \| `panel` \| `window` \| `orb` | `panel` | The overall shape |
| `preset` | string | — | Load a preset's values first, then apply your overrides |
| `width` | int | `720` | Popup width in pixels |
| `height` | int | `0` | Fixed height; `0` sizes to content |
| `max_height` | int | `560` | Ceiling once results appear |
| `position` | see below | `center` | Anchor on screen |
| `offset_x` | int | `0` | Horizontal nudge from the anchor |
| `offset_y` | int | `0` | Vertical nudge; negative lifts the popup above centre |
| `corner_radius` | int | `16` | Corner rounding |
| `padding` | int | `14` | Padding inside the shell |
| `opacity` | float | `1.0` | `0.3`–`1.0` |
| `blur_background` | bool | `true` | Ask the compositor for a translucent backdrop |
| `shadow` | bool | `true` | Drop shadow |
| `decorated` | bool | `false` | `true` gives a real title bar; `false` is the chromeless Spotlight look |
| `show_logo` | bool | `true` | The Keylane mark |
| `show_title` | bool | `false` | "Keylane / Ask your computer" heading |
| `show_status_chips` | bool | `true` | NPU / worker health chips |
| `show_project_picker` | bool | `true` | Project dropdown and local-only toggle |
| `show_hints` | bool | `true` | Keyboard hint line |
| `show_results` | bool | `true` | The result area |
| `dismiss_on_focus_loss` | bool | `true` | Close when you click away |
| `animation` | `none` \| `fade` \| `scale` \| `slide` | `scale` | Entry animation |
| `animation_ms` | int | `140` | Its duration |
| `input_placeholder` | string | `Ask anything…` | Entry placeholder |
| `orb_size` | int | `72` | Diameter of the collapsed orb |
| `font_family` | string | `""` | Font for the popup; empty uses the system font |
| `font_size` | int | `15` | Base size for the prompt entry. The entry's height is derived from it (`font_size × 2.2`, floor 40px), so this is what sets how tall the bar is. |

`position` accepts `center`, `top`, `bottom`, `left`, `right`, `top-left`,
`top-right`, `bottom-left`, `bottom-right`.

> **Note**: Precise positioning needs `gtk4-layer-shell`, which wlroots
> compositors (sway, Hyprland) provide. GNOME's Mutter has no layer-shell
> protocol, so there Keylane centres an undecorated window and reproduces
> `offset_y` with transparent padding — the bar still floats above centre, but
> corner anchoring falls back to centred.

### Four shapes, four recipes

**A Spotlight bar** — the default:

```toml
[popup]
mode = "bar"
width = 720
position = "center"
offset_y = -150
corner_radius = 18
padding = 10
decorated = false
show_status_chips = false
show_project_picker = false
show_hints = false
font_size = 20
```

That gives a 720 x 64 bar — a 44px entry with 10px of padding either side.
Width to height of roughly 11:1 is what makes it read as a search field;
much below 6:1 and it starts to look like a dialog. In `bar` mode the entry
is drawn frameless and transparent so the bar itself is the only surface —
a bordered box inside a bordered bar is the usual reason a Spotlight clone
looks like a form.

**A full assistant window:**

```toml
[popup]
mode = "window"
width = 880
height = 640
decorated = true
show_title = true
show_status_chips = true
show_project_picker = true
dismiss_on_focus_loss = false
```

**A corner orb:**

```toml
[popup]
mode = "orb"
position = "bottom-right"
offset_x = -32
offset_y = -32
orb_size = 64
width = 420
show_hints = false
```

**A top-edge command strip:**

```toml
[popup]
mode = "bar"
position = "top"
offset_y = 12
width = 900
corner_radius = 0
opacity = 0.96
```

## launcher.css

Plain GTK 4 CSS. If you omit it, Keylane generates one from `[colors]` and
`[popup]`. Ship your own when you want more than colours.

Use `@define-color` aliases so the rest of the generated styling still resolves:

```css
@define-color ag_bg #0b1220;
@define-color ag_surface #121a2b;
@define-color ag_text #e8eefc;
@define-color ag_muted #9aa8c7;
@define-color ag_accent #34d399;
@define-color ag_border #243047;
@define-color ag_danger #fb7185;

/* The window itself must stay transparent — the shell paints the surface. */
window.keylane-popup { background-color: transparent; }

.keylane-shell {
  background-color: alpha(@ag_bg, 0.94);
  color: @ag_text;
  border: 1px solid @ag_border;
  border-radius: 18px;
  padding: 10px;
}

.keylane-prompt {
  background-color: transparent;
  border: none;
  font-size: 20px;
  caret-color: @ag_accent;
}

.keylane-orb {
  background-image: linear-gradient(135deg, @ag_accent, shade(@ag_accent, 0.7));
  border-radius: 999px;
}
```

### Classes you can style

| Class | Element |
| --- | --- |
| `window.keylane-popup` | The window; keep it transparent |
| `.keylane-shell` | The visible container — this is your "popup" |
| `.keylane-card` | Inner grouping in panel/window modes |
| `.keylane-prompt` | The text entry |
| `.keylane-icon-btn` | Microphone and other icon buttons |
| `.keylane-icon-btn.recording` | Microphone while dictating |
| `.keylane-send` | The send button |
| `.keylane-title`, `.keylane-subtitle` | Header text |
| `.keylane-chip`, `.keylane-chip.on`, `.keylane-chip.warn` | Status chips |
| `.keylane-result`, `.keylane-result-view` | The result area |
| `.keylane-progress` | Progress and result text |
| `.keylane-hint` | The keyboard hint line |
| `.keylane-orb` | The collapsed orb |

`popup.css`, if present, is appended after `launcher.css` — handy for keeping
your overrides separate from generated colours.

## web.css

CSS custom properties consumed by the control panel. The panel derives every
other colour it needs from these seven:

```css
:root {
  --ag-bg: #0b1220;
  --ag-surface: #121a2b;
  --ag-text: #e8eefc;
  --ag-muted: #9aa8c7;
  --ag-accent: #34d399;
  --ag-border: #243047;
  --ag-danger: #fb7185;
}
```

## Testing your theme

```bash
# Install and activate
curl -F file=@my-theme.zip http://127.0.0.1:9100/api/themes/install
curl -X PUT http://127.0.0.1:9100/api/themes/active \
     -H 'Content-Type: application/json' -d '{"id":"aurora"}'

# Check the shape the gateway resolved
curl -s http://127.0.0.1:9100/api/themes/active/popup.json
```

Then press <kbd>Super</kbd>+<kbd>Space</kbd>. The popup re-reads the active
theme every time it opens, so there is nothing to restart.

## API

```http
GET    /api/themes                       list, with each theme's popup spec
GET    /api/themes/active                active id + popup spec + colours
GET    /api/themes/active/popup.json     just the popup spec
GET    /api/themes/active/launcher.css   GTK CSS, launcher.css + popup.css
GET    /api/themes/presets               the built-in popup presets
GET    /theme.css                        control-panel CSS
PUT    /api/themes/active                {"id": "aurora"}
POST   /api/themes/install               multipart zip
DELETE /api/themes/{id}
```

The active id lives in `config/themes.toml`.

## Built-in themes

| ID | Mode | Look |
| --- | --- | --- |
| `default` | bar | Light Spotlight bar, blue accent |
| `midnight` | bar | Dark Spotlight bar, cyan accent |
| `panel` | panel | Centred panel with chips and a project picker |
| `paper` | panel | Warm paper light, brick accent |
| `studio` | window | Full dark assistant window, violet accent |
| `orb` | orb | Corner orb, sky accent |

> **Warning**: Built-in themes regenerate their `theme.toml` and CSS on every
> gateway start, so edits to them are overwritten. To customise one, copy its
> folder to a new id and edit that.

## Guidelines

- Keep text contrast at WCAG AA or better on both `bg` and `surface`.
- Do not reference remote fonts — Keylane is local-first and may be offline.
  Use a system stack, or a font the user has installed.
- Respect `animation_ms` at 120–200 ms; the popup should feel instant.
- Test at least `bar` and `panel`; those are what most people use.
