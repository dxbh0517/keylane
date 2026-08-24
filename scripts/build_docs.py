#!/usr/bin/env python3
"""Build the Keylane handbook at web/docs/ from the markdown in docs/.

The markdown files are the single source of truth: they read well on GitHub and
they render into the designed site the gateway serves at ``/docs``. Run this
after editing anything under ``docs/``::

    python scripts/build_docs.py

Only the Markdown this project actually uses is supported — headings,
paragraphs, lists, tables, fenced code, blockquote callouts, and inline
code/bold/italic/links. That keeps the renderer small enough to read.
"""

from __future__ import annotations

import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs"
OUT = ROOT / "web" / "docs"


@dataclass
class Page:
    slug: str
    source: str
    title: str
    eyebrow: str
    summary: str


@dataclass
class Group:
    title: str
    pages: list[Page]


NAV: list[Group] = [
    Group(
        "Start here",
        [
            Page("index", "INDEX.md", "Keylane handbook", "Documentation",
                 "What Keylane is and where to go next."),
            Page("install", "INSTALL.md", "Install, update, uninstall", "Guide",
                 "Getting Keylane onto a machine, upgrading it, and taking it off again."),
            Page("popup", "POPUP.md", "The popup and the tray", "Guide",
                 "The Spotlight overlay, the hotkey, and the background-work indicator."),
        ],
    ),
    Group(
        "The assistant",
        [
            Page("assistant", "ASSISTANT.md", "How the assistant thinks", "Concept",
                 "Try it yourself, delegate what you cannot, then follow up."),
            Page("tools", "TOOLS.md", "Tools", "Reference",
                 "Every capability the assistant can reach, and how to add one."),
            Page("skills", "SKILLS.md", "Skills", "Guide",
                 "Teach Keylane your house rules, or import skills from GitHub."),
            Page("speech", "SPEECH.md", "Speech", "Guide",
                 "Read answers aloud, locally, with the engine and voice you pick."),
            Page("canvas", "CANVAS.md", "Canvas answers", "Reference",
                 "The structured document an answer is rendered from."),
        ],
    ),
    Group(
        "Extending",
        [
            Page("plugins", "PLUGINS.md", "Writing plugins", "Guide",
                 "Add workers, MCP servers, tools and skills to Keylane."),
            Page("themes", "THEMES.md", "Writing themes", "Guide",
                 "Restyle the panel and reshape the popup — bar, panel, window or orb."),
            Page("api", "API.md", "HTTP API", "Reference",
                 "Every endpoint the gateway exposes on 127.0.0.1."),
            Page("roadmap", "ROADMAP.md", "What it still needs", "Notes",
                 "Known gaps and where Keylane should go next."),
        ],
    ),
]

ALL_PAGES = [page for group in NAV for page in group.pages]


# ------------------------------------------------------------------ markdown


INLINE_CODE = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*([^*]+)\*\*")
ITALIC = re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# Matched after escaping, so the pattern is the escaped form.
KBD = re.compile(r"&lt;kbd&gt;(.+?)&lt;/kbd&gt;")


def inline(text: str) -> str:
    """Render inline markdown. Escapes first, so source HTML cannot leak in."""
    placeholders: list[str] = []

    def stash(markup: str) -> str:
        placeholders.append(markup)
        return f"\x00{len(placeholders) - 1}\x00"

    # Pull code spans out before escaping so their contents stay literal.
    text = INLINE_CODE.sub(lambda m: stash(f"<code>{html.escape(m.group(1))}</code>"), text)
    text = html.escape(text)

    def link(match: re.Match[str]) -> str:
        label, href = match.group(1), match.group(2)
        # Cross-references between docs point at the built pages.
        if href.endswith(".md") and "/" not in href:
            href = f"{Path(href).stem.lower()}.html"
        external = href.startswith("http")
        rel = ' target="_blank" rel="noopener"' if external else ""
        return f'<a href="{html.escape(href, quote=True)}"{rel}>{label}</a>'

    text = LINK.sub(link, text)
    text = BOLD.sub(r"<strong>\1</strong>", text)
    text = ITALIC.sub(r"<em>\1</em>", text)
    text = KBD.sub(r"<kbd>\1</kbd>", text)

    for index, markup in enumerate(placeholders):
        text = text.replace(f"\x00{index}\x00", markup)
    return text


CALLOUT_KINDS = {"note": "", "tip": "ok", "warning": "warn", "danger": "danger"}


def render(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    index = 0
    total = len(lines)

    def close_list(stack: list[str]) -> None:
        while stack:
            out.append(f"</{stack.pop()}>")

    list_stack: list[str] = []

    while index < total:
        line = lines[index]
        stripped = line.strip()

        # Fenced code
        if stripped.startswith("```"):
            close_list(list_stack)
            language = stripped[3:].strip()
            index += 1
            body: list[str] = []
            while index < total and not lines[index].strip().startswith("```"):
                body.append(lines[index])
                index += 1
            index += 1
            label = f'<span class="code-label">{html.escape(language)}</span>' if language else ""
            code = html.escape("\n".join(body))
            out.append(f'{label}<pre><code>{code}</code></pre>')
            continue

        # Blank line
        if not stripped:
            close_list(list_stack)
            index += 1
            continue

        # Headings
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            close_list(list_stack)
            level = len(heading.group(1))
            # The page title comes from the nav, so H1 in source is skipped.
            if level > 1:
                out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            index += 1
            continue

        # Horizontal rule
        if re.fullmatch(r"-{3,}|\*{3,}", stripped):
            close_list(list_stack)
            out.append("<hr />")
            index += 1
            continue

        # Tables
        if "|" in stripped and index + 1 < total and re.match(
            r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[index + 1]
        ):
            close_list(list_stack)
            header = [c.strip() for c in stripped.strip("|").split("|")]
            index += 2
            rows: list[list[str]] = []
            while index < total and "|" in lines[index] and lines[index].strip():
                rows.append([c.strip() for c in lines[index].strip().strip("|").split("|")])
                index += 1
            head = "".join(f"<th>{inline(c)}</th>" for c in header)
            body = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>" for row in rows
            )
            out.append(
                f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
                f"<tbody>{body}</tbody></table></div>"
            )
            continue

        # Blockquote callouts: "> **Note** body" or plain "> body"
        if stripped.startswith(">"):
            close_list(list_stack)
            block: list[str] = []
            while index < total and lines[index].strip().startswith(">"):
                block.append(lines[index].strip().lstrip(">").strip())
                index += 1
            text = " ".join(block).strip()
            kind = ""
            title = ""
            marker = re.match(r"^\*\*(Note|Tip|Warning|Danger)\*\*:?\s*(.*)$", text, re.I)
            if marker:
                title = marker.group(1).capitalize()
                kind = CALLOUT_KINDS.get(title.lower(), "")
                text = marker.group(2)
            heading_html = f'<span class="note-title">{title}</span>' if title else ""
            out.append(f'<div class="note {kind}">{heading_html}<p>{inline(text)}</p></div>')
            continue

        # Lists
        bullet = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if bullet:
            indent = len(bullet.group(1))
            ordered = bullet.group(2)[0].isdigit()
            tag = "ol" if ordered else "ul"
            depth = indent // 2 + 1
            while len(list_stack) > depth:
                out.append(f"</{list_stack.pop()}>")
            while len(list_stack) < depth:
                out.append(f"<{tag}>")
                list_stack.append(tag)
            out.append(f"<li>{inline(bullet.group(3))}</li>")
            index += 1
            continue

        # Paragraph — join until a blank line or a block starter.
        close_list(list_stack)
        chunk = [stripped]
        index += 1
        while index < total:
            nxt = lines[index].strip()
            if not nxt or nxt.startswith(("#", "```", ">", "-", "*", "|")) or re.match(r"^\d+\.\s", nxt):
                break
            chunk.append(nxt)
            index += 1
        out.append(f"<p>{inline(' '.join(chunk))}</p>")

    close_list(list_stack)
    return "\n".join(out)


# --------------------------------------------------------------------- shell


def sidebar(current: Page) -> str:
    groups = []
    for group in NAV:
        links = "".join(
            f'<a href="{p.slug}.html"'
            + (' aria-current="page"' if p.slug == current.slug else "")
            + f">{html.escape(p.title)}</a>"
            for p in group.pages
        )
        groups.append(
            f'<div><p class="side-group-title">{html.escape(group.title)}</p>'
            f'<div class="side-links">{links}</div></div>'
        )
    return "".join(groups)


def page_nav(current: Page) -> str:
    position = ALL_PAGES.index(current)
    previous = ALL_PAGES[position - 1] if position > 0 else None
    following = ALL_PAGES[position + 1] if position + 1 < len(ALL_PAGES) else None
    parts = []
    if previous:
        parts.append(
            f'<a href="{previous.slug}.html"><span class="dir">Previous</span>'
            f'<span class="name">{html.escape(previous.title)}</span></a>'
        )
    if following:
        parts.append(
            f'<a class="next" href="{following.slug}.html"><span class="dir">Next</span>'
            f'<span class="name">{html.escape(following.title)}</span></a>'
        )
    return f'<nav class="page-nav">{"".join(parts)}</nav>' if parts else ""


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title} · Keylane docs</title>
  <meta name="description" content="{summary}" />
  <link rel="icon" type="image/png" href="/assets/favicon.png" />
  <link rel="stylesheet" href="assets/docs.css" />
</head>
<body>
  <div class="docs">
    <aside class="sidebar">
      <a class="docs-brand" href="index.html">
        <img src="/assets/logo.svg" width="30" height="30" alt="" />
        <span>
          <strong>Keylane</strong>
          <span>Handbook</span>
        </span>
      </a>
      <nav class="side-nav" aria-label="Documentation">{sidebar}</nav>
      <div class="side-foot">
        <a href="/">← Control panel</a>
        <span>Local-first · 127.0.0.1</span>
      </div>
    </aside>

    <main class="content">
      <article class="article">
        <p class="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
{body}
        {page_nav}
      </article>
    </main>

    <aside class="toc" aria-label="On this page"></aside>
  </div>
  <script src="assets/docs.js"></script>
</body>
</html>
"""


def build() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    written = 0

    for page in ALL_PAGES:
        source = SRC / page.source
        if not source.exists():
            missing.append(page.source)
            continue
        body = render(source.read_text(encoding="utf-8"))
        markup = TEMPLATE.format(
            title=html.escape(page.title),
            summary=html.escape(page.summary, quote=True),
            eyebrow=html.escape(page.eyebrow),
            sidebar=sidebar(page),
            body=body,
            page_nav=page_nav(page),
        )
        (OUT / f"{page.slug}.html").write_text(markup, encoding="utf-8")
        written += 1
        print(f"  {page.source:16} → web/docs/{page.slug}.html")

    print(f"\nBuilt {written} page(s) into {OUT}")
    if missing:
        print(f"Missing markdown sources: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
