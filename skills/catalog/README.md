# Skill catalog

A short, curated list of skills worth installing, so the Skills tab is not
an empty box asking you to already know what you want.

Each entry is one JSON file naming a GitHub repository and the path to the
skill inside it. Installing goes through the same importer as any other
GitHub skill (`POST /api/skills/import`), so a catalog entry is a
**shortcut, not a special case** — nothing here can do anything you could
not do by pasting the repo yourself.

Entries were chosen for three things: a reputable source, a real install
count on skills.sh, and usefulness to a *desktop assistant* rather than to
a coding agent. Every one is checked against the live repository by
`GET /api/skills/catalog`, which marks anything that has moved or been
renamed rather than offering a broken install.

Imported skills arrive **disabled**, as with any import.

## Adding an entry

```json
{
  "id": "pdf",
  "name": "PDF handling",
  "repo": "anthropics/skills",
  "path": "skills/pdf/SKILL.md",
  "desc": "What it does, in one sentence.",
  "source": "Anthropic",
  "installs": "184K",
  "tags": ["documents", "files"]
}
```
