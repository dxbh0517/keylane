# Canvas answers

An assistant that replies with raw JSON, or a wall of markdown, makes you do the
layout work. Keylane answers with a **canvas** instead: a small structured
document that the popup renders as native widgets and the control panel renders
as HTML.

The same canvas produces stat tiles in the result panel and a styled block in
the browser, and both follow the active theme.

## The shape

```json
{
  "title": "Disk usage",
  "summary": "254 GB free of 952 GB.",
  "blocks": [
    {"type": "stats", "items": [
      {"label": "Free", "value": "254 GB"},
      {"label": "Used", "value": "695 GB", "detail": "74% of the disk"}
    ]},
    {"type": "table",
     "columns": ["Mount", "Size", "Use%"],
     "rows": [["/", "952G", "74%"], ["/boot", "1.0G", "21%"]]},
    {"type": "note", "style": "warning", "text": "Root is 74% full."}
  ],
  "source": "via run_command"
}
```

Only `type` is required on a block. `title`, `summary` and `source` are all
optional.

## Block types

| Type | Fields | Use it for |
| --- | --- | --- |
| `text` | `text` | A paragraph |
| `heading` | `text`, `level` (2–4) | A section break |
| `stats` | `items: [{label, value, detail?}]` | Numbers worth seeing at a glance |
| `table` | `columns: []`, `rows: [[]]` | Anything columnar |
| `list` | `entries: []`, `ordered?` | Steps or bullets |
| `code` | `text`, `language?` | Commands, output, snippets |
| `note` | `text`, `style` | A callout: `info`, `success`, `warning`, `danger` |
| `links` | `links: [{label, href}]` | Files or URLs the answer produced |

The schema is deliberately small — eight types, one level of nesting, every
field optional. A 1.5B model has to be able to emit it correctly first time.

## Rules the renderer enforces

- **Empty blocks are dropped.** A canvas exists to show real content, so a
  table with no rows or a note with no text never reaches the screen. There are
  no "No data" placeholders.
- **Everything is escaped.** Canvas content is model output and tool output —
  it is data, never markup.
- **A malformed canvas is never fatal.** If the JSON does not parse, or the
  shape is wrong, the answer falls back to plain text wrapped in a minimal
  canvas. You always get an answer.
- **Loose types are coerced.** Numbers in table cells become strings; an unknown
  note style becomes `info`.

## Canvases are built, not just requested

Asking a 1.5B model to emit a canvas as JSON does not work reliably — it tends
to loop trying more tools, or answer in prose. So the canvas is **derived from
what the tools actually produced**, in Python, where the result is
deterministic. Four steps, in order:

1. The model emitted a canvas outright. Best case, and rare.
2. A canvas hiding inside the answer string.
3. The answer is prose or markdown: headings, tables, lists, fences and
   callouts are parsed into real blocks, so you see layout rather than literal
   `##` and `|---|`.
4. No usable answer at all: build one from the tool results. Columnar command
   output — `df`, `free`, `ps`, `ls -l` — becomes a table; anything else
   becomes a code block or paragraphs.

Step 4 is what makes the popup useful with a small model. `df -h` renders as a
six-column table whether or not the model ever managed to say so.

## When the model answers in prose

Not every answer deserves structure, and a small model will sometimes just
write a sentence. That is wrapped automatically:

- Columnar output — `df`, `ls`, `ps` — becomes a `code` block.
- Anything else becomes one `text` block per paragraph.

So the popup renders consistently whether or not the model produced a real
canvas.

## Producing one from a plugin

A worker's output is wrapped for you: changed files become a `list`, an output
image becomes `links`, and the text becomes `text` or `code`. To take control,
return a canvas from your tool in `ToolResult.data`:

```python
from app.canvas import Block, Canvas, Stat
from app.tools.base import BaseTool, ToolResult


class DiskTool(BaseTool):
    name = "disk_usage"
    description = "Report free space on the main filesystem."

    async def run(self, args):
        canvas = Canvas(
            title="Disk usage",
            blocks=[
                Block(type="stats", items=[Stat(label="Free", value="254 GB")]),
                Block(type="note", style="warning", text="Root is 74% full."),
            ],
        )
        return ToolResult.success(canvas.to_text(), data={"canvas": canvas.model_dump()})
```

`to_text()` keeps the plain-text fallback honest for logs and the OpenAI-
compatible endpoint.

## Where it appears

| Surface | Rendered by |
| --- | --- |
| The result panel | `launcher/canvas_view.py` — GTK widgets |
| The control panel | `app.canvas.render_html` — classed HTML the theme styles |
| `/api/chat` | The `canvas` field on the response, next to `result` |

## Why not the Cursor canvas format

Cursor's canvas is a `.canvas.tsx` React file that imports from `cursor/canvas`
and is compiled by the Cursor IDE. It is an excellent format inside Cursor and
cannot run anywhere else — not in a GTK popup, and not in a page served from
`127.0.0.1` with no build step.

Keylane's canvas takes the same idea (structured, self-describing output over
a wall of text) in a form that renders natively, offline, in both of Keylane's
surfaces, and that a small local model can actually emit.
