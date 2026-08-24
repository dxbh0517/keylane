# Skills

A skill is a markdown file that teaches Keylane something about *your* setup.
When its trigger words appear in a request, its content is appended to the
assistant's system prompt for that turn.

Skills are how you encode house rules without writing code: which branch to
deploy from, what your ComfyUI checkpoints are called, that "the API" means one
particular repository.

## Writing one

Drop a `.md` file into `skills/` at the repository root:

```markdown
---
name: deploy
description: How this machine deploys things.
triggers: deploy, ship, release, publish
---

Deployments always go through `make release`, never `git push` to production.
Run the test suite first. If anything fails, stop and report — do not retry.

The staging URL is https://staging.internal; production is https://app.internal.
```

Then press **Reload** on the Skills tab, or `POST /api/skills/reload`.

## Front matter

| Key | Meaning |
| --- | --- |
| `name` | Display name; defaults to the filename |
| `description` | One line, shown in the control panel |
| `triggers` | Comma-separated substrings; matched case-insensitively against the request |
| `always` | `true` to include this skill in every request |
| `enabled` | `false` to keep the file but switch it off |

A skill with no triggers and `always: false` never activates — the control panel
flags that so you notice the typo.

At most four matching skills are included per request, in name order, to keep
the prompt small enough for a 1.5B model to follow.

## Examples

**Project shorthand:**

```markdown
---
name: projects
description: What the user's shorthand names mean.
always: true
---

- "the API" means /home/you/code/api-server
- "the site" means /home/you/code/marketing
- "the app" means /home/you/code/mobile

When delegating a coding task, always pass the matching absolute path as the
project argument.
```

**Image style:**

```markdown
---
name: image-style
description: Default look for generated images.
triggers: image, render, artwork, illustration, hero
---

Unless told otherwise, generate images at 1536x1024 with the flux checkpoint.
Prefer natural light and muted colour. Never add text to an image — it comes out
garbled.
```

**Delegation preference:**

```markdown
---
name: worker-choice
description: Which AI tool to hand which job to.
triggers: refactor, implement, fix, bug, test
---

Send anything touching more than two files to Claude Code, not Cursor.
Single-file edits can go to either. Always pass the project path, and always
check `changed_files` in the result before reporting success.
```

## Suggested skills

The Skills tab opens with a short curated list — PDF, Word, spreadsheet and
presentation handling, research, writing structure, and design skills — from
Anthropic and other reputable sources.

Each entry is checked against its source repository before being offered, so a
skill that has moved or been renamed upstream is marked rather than handed to
you as a broken install. Installing one goes through the ordinary GitHub
importer, so a catalog entry is a **shortcut, not a special case**: nothing it
can do is anything you could not do by pasting the repository yourself.

Entries live in `skills/catalog/*.json`, one file each — add your own the same
way. See [the catalog README](https://github.com/dxbh0517/keylane/tree/main/skills/catalog).

## Importing from GitHub

**Control panel → Skills → Import skills from GitHub**, then give it a
repository: `anthropics/skills`, a full URL, or `owner/repo@branch`.

Keylane reads the repository *tree* through the GitHub API rather than cloning
it, works out which files are actually skills, and shows you the list. You pick
the ones you want; nothing else is downloaded.

Recognised layouts:

| Path | Typical source |
| --- | --- |
| `skills/<name>/SKILL.md` | Claude and Cursor plugin repos |
| `<name>/SKILL.md` | one skill per folder |
| `skills/<name>.md` | a flat folder of markdown skills |
| `.cursor/rules/<name>.md` | Cursor rules used as skills |
| `agents/<name>.md` | agent instruction packs |

A file only counts as a skill if it carries front matter with a `name` or
`description`, or is called `SKILL.md` — otherwise every README would match.
`node_modules`, `tests`, `dist` and similar are never walked.

> **Note**: Imported skills arrive **disabled**. A repository should not be able
> to change how your assistant behaves the moment you import it — read them, then
> switch on the ones you want.

### Rate limits and private repos

Unauthenticated GitHub allows 60 requests an hour, which one scan can use up.
Keylane looks for a token in `KEYLANE_GITHUB_TOKEN`, `GITHUB_TOKEN` or
`GH_TOKEN`, and falls back to `gh auth token` if the GitHub CLI is signed in.
A token is also what makes private repositories readable.

```bash
systemctl --user edit ai-gateway.service
# [Service]
# Environment=KEYLANE_GITHUB_TOKEN=ghp_...
```

### API

```http
POST /api/skills/discover   {"repo": "anthropics/skills"}
POST /api/skills/import     {"repo": "...", "paths": ["skills/a/SKILL.md"]}
```

## Skills from plugins

A plugin can ship skills alongside its tools, so installing it also teaches the
assistant how to use it. See [Writing plugins](PLUGINS.md#contributing-skills).

Plugin skills appear in the control panel with the plugin's id as their source,
and are refreshed by **Plugins → Reload**.

## API

```http
GET  /api/skills          list every skill with its content
POST /api/skills/reload   re-read skills/ and re-collect plugin skills
```

## Notes

- Skills are prompt text, not policy. A skill cannot grant the assistant a tool
  it does not have, or bypass a confirmation gate — those are decided in Python.
- Keep each skill short. A 1.5B NPU model follows three clear sentences far
  better than three paragraphs.
- Skills are matched by plain substring, so `triggers: cat` will also fire on
  "concatenate". Prefer distinctive words.
