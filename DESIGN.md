# Design System: Keylane

## 1. Visual Theme & Atmosphere

A restrained, gallery-airy control panel and handbook with confident monochrome
layout and quiet spring-like motion. Density sits in the daily-app band: enough
structure for a local AI gateway, never a packed cockpit. The atmosphere is
clinical yet warm, like a well-lit architecture studio on warm bone paper
(`#F7F6F3`) with charcoal ink.

**Dials:** Variance 5 · Motion 4 · Density 5

## 2. Color Palette & Roles

- **Warm Bone** (`#F7F6F3`) — Primary canvas / page background
- **Pure Surface** (`#FFFFFF`) — Cards, forms, elevated panels
- **Charcoal Ink** (`#18181B`) — Primary text and primary CTAs (Zinc-950 depth)
- **Muted Steel** (`#71717A`) — Secondary text, descriptions, metadata
- **Whisper Border** (`#EAEAEA`) — 1px structural lines on all cards and inputs
- **Pale Red** (`#FDEBEC` / text `#9F2F2D`) — Danger / error soft fill
- **Pale Green** (`#EDF3EC` / text `#346538`) — Success / online soft fill
- **Pale Yellow** (`#FBF3DB` / text `#956400`) — Warning soft fill

Single accent lock: charcoal for interactive emphasis. Themes may override
`--ag-accent`; the default stays monochrome. No purple, neon, or saturated
brand fills as the product default.

Dark mode mirrors the same roles on Zinc-950 surfaces (`#111113` / `#18181B`).

## 3. Typography Rules

- **Display / UI:** SF Pro Display, Geist Sans, Helvetica Neue, system-ui
- **Body:** Same sans stack, 15–16px, line-height ~1.55–1.65, max ~65ch on docs
- **Mono:** Geist Mono, SF Mono, JetBrains Mono for code, badges, metadata, counts
- **Banned:** Inter, Roboto, Open Sans as defaults. No generic serif in the
  control panel or software docs chrome.

Headings use tight tracking (`-0.02em` to `-0.03em`) and weight-driven hierarchy.

## 4. Component Stylings

- **Buttons:** Flat charcoal primary, 6px radius, no outer glow. Active state
  uses `scale(0.98)`. Ghost buttons stay transparent with sunken hover.
- **Cards:** `1px solid #EAEAEA`, 8–12px radius max, ultra-diffuse shadow
  (`0 2px 8px rgba(0,0,0,0.04)`). Used when grouping interactive settings.
- **Inputs:** Label above, helper optional, error below. 44px min tap height.
  Focus ring is a soft charcoal soft-fill halo.
- **Nav:** Sticky translucent rail (`backdrop-filter`) with solid fallback under
  `prefers-reduced-transparency`. Active item uses soft fill, not loud color.
- **Badges:** Small mono uppercase pills with muted pastel fills for status only.
- **Loaders:** Compact border spinner or skeletal empty dashed regions. No
  decorative circular theater.
- **Empty states:** Dashed border region with short functional copy.

## 5. Layout Principles

- Control panel: sticky left rail + measured main column (`max-width ~1120px`)
- Docs: sidebar + article (`~68ch`) + on-this-page TOC; collapses cleanly below
  1180px / 860px
- Prefer CSS Grid for status and tool grids; single-column under 768px
- Hairline dividers over heavy card stacks when density rises
- Full-height regions use `min-height: 100dvh`, never `h-screen`

## 6. Motion & Interaction

- Default easing: `cubic-bezier(0.16, 1, 0.3, 1)` at 150–280ms
- Panel enter: `translateY(10px)` + opacity
- Press feedback on pointer-down feel via `:active { scale(0.98) }`
- Animate only `transform` and `opacity`
- Honor `prefers-reduced-motion` (collapse to near-instant) and
  `prefers-reduced-transparency` (opaque chrome)

## 7. Anti-Patterns (Banned)

- No emojis as icons (SVG sprite / stroke icons only)
- No Inter / Roboto / Open Sans as the product default
- No pure black (`#000000`) backgrounds or body text
- No neon outer glows, AI-purple gradients, or glassmorphism as the hero look
- No `rounded-full` on large cards or primary button containers
- No 3-column equal marketing feature cards in product chrome
- No AI copywriting clichés ("Elevate", "Seamless", "Unleash")
- No em-dashes in UI copy
- No scroll-cue labels ("Scroll to explore")
- No layout-thrashing animations (`width` / `height` / `top` / `left`)
