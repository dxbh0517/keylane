---
name: example-projects
description: Example skill — rename or delete this once you write your own.
triggers: example-trigger-that-never-fires
enabled: false
---

This file shows the shape of a Keylane skill. Front matter decides when it
applies; the body is appended to the assistant's system prompt for that request.

Replace it with something true about your machine, for example:

- "the API" means /home/you/code/api-server
- deployments always go through `make release`, never a direct push
- when generating images, default to 1536x1024 with the flux checkpoint

Set `enabled: true` and give it real `triggers` to switch it on. See
docs/SKILLS.md for the full reference.
