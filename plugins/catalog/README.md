# Plugin catalog

Every plugin Keylane ships with lives here, and **none of them are installed
by default**. A fresh install talks to nothing until you choose what it may
talk to.

Install from the control panel (**Plugins -> Catalog**) or the API:

```bash
curl -s http://127.0.0.1:9100/api/plugins/catalog
curl -X POST http://127.0.0.1:9100/api/plugins/catalog/lmstudio/install
```

Installing records the plugin in `config/plugins.toml` and registers it.
Removing it takes the entry back out; your settings for it are kept unless
you pass `?purge=true`.

Each folder holds a `plugin.toml`. `entry = "builtin:<id>"` points at an
implementation that ships inside Keylane. Community plugins use the same
manifest format but carry their own `plugin.py` or an MCP `command` — see
[docs/PLUGINS.md](../../docs/PLUGINS.md).
