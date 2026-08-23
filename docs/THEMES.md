# Theme system

**Keylane** themes style:

1. The **web control panel** (`web.css`)
2. The **GTK launcher popup** (`launcher.css`)

Built-ins: `default`, `midnight`, `paper`. Community themes install as zip archives.

## Theme package layout

```text
my-theme/
  theme.toml
  web.css
  launcher.css
```

### theme.toml

```toml
id = "aurora"
name = "Aurora"
author = "community"
version = "1.0.0"
description = "Cool slate with emerald accent."

[colors]
bg = "#0b1220"
surface = "#121a2b"
text = "#e8eefc"
muted = "#9aa8c7"
accent = "#34d399"
border = "#243047"
```

If `web.css` / `launcher.css` are missing, the gateway generates them from `[colors]`.

### web.css

Define CSS variables consumed by the control panel:

```css
:root {
  --ag-bg: #0b1220;
  --ag-surface: #121a2b;
  --ag-text: #e8eefc;
  --ag-muted: #9aa8c7;
  --ag-accent: #34d399;
  --ag-border: #243047;
}
```

### launcher.css

GTK4 CSS. Prefer `@define-color` aliases:

```css
@define-color ag_bg #0b1220;
@define-color ag_surface #121a2b;
@define-color ag_text #e8eefc;
@define-color ag_accent #34d399;
@define-color ag_border #243047;

window { background-color: @ag_bg; color: @ag_text; }
```

## Install / select

- UI: **Themes** tab → upload zip, or click **Use theme**
- API:
  - `GET /api/themes`
  - `PUT /api/themes/active` `{"id":"midnight"}`
  - `POST /api/themes/install` (multipart zip)
  - `DELETE /api/themes/{id}`
  - `GET /api/themes/active/launcher.css`

Active theme id is stored in `config/themes.toml`.

## Community guidelines

- Ship both light and dark only if you intend two separate theme ids (one theme = one mode).
- Keep contrast WCAG AA for text on `bg` / `surface`.
- Do not embed remote fonts that phone home; prefer system stacks.
- Zip the theme folder so `theme.toml` is inside the archive.
