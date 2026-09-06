"""Render a model's own chat template, including the variables it accepts.

Both runtimes had a way to apply a template and neither could pass it a
*variable*, which is how modern templates carry switches. The one that matters
here is `enable_thinking`: MiniCPM5's template ends with

    {%- if enable_thinking is defined %}
        {%- if enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}
        {%- elif enable_thinking is true %}{{- '<think>\\n' }}

so passing `false` pre-closes the reasoning block and the model goes straight
to the answer. Undefined means the model decides, and it decides to think.

The obvious route — `AutoTokenizer.from_pretrained` — does not survive contact
with these exports. An OpenVINO conversion of MiniCPM5 names its tokenizer
class `TokenizersBackend`, which transformers refuses with "does not exist or
is not currently imported", and the whole render fails over a class we never
needed: only the template text is wanted, not a tokenizer. So the template is
read as text and rendered with Jinja directly, which works for any export that
ships one and cannot be broken by a tokenizer class from the future.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Newer exports ship the template as its own file; older ones inline it in
# tokenizer_config.json. Both are still in circulation, often side by side.
TEMPLATE_FILE = "chat_template.jinja"
TOKENIZER_CONFIG = "tokenizer_config.json"


def load_template(model_dir: Path) -> str | None:
    """The model's chat template as text, or None if it ships none."""
    path = model_dir / TEMPLATE_FILE
    if path.is_file():
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError:
            logger.debug("could not read %s", path, exc_info=True)

    config = model_dir / TOKENIZER_CONFIG
    if not config.is_file():
        return None
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    template = data.get("chat_template")
    # Some exports carry a list of named templates rather than one string.
    if isinstance(template, list):
        for entry in template:
            if isinstance(entry, dict) and entry.get("name") == "default":
                template = entry.get("template")
                break
        else:
            first = template[0] if template else None
            template = first.get("template") if isinstance(first, dict) else None
    return template if isinstance(template, str) and template.strip() else None


def _environment() -> Any:
    """A Jinja environment with what chat templates actually use.

    `trim_blocks`/`lstrip_blocks` off, matching transformers: templates are
    written with explicit `{%-` whitespace control and turning these on shifts
    every token boundary.
    """
    from jinja2 import Environment
    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def _raise_exception(message: str) -> None:
        raise TemplateError(message)

    # Sandboxed for the same reason transformers sandboxes it: a chat template
    # is code that arrives with a downloaded model.
    env: Environment = ImmutableSandboxedEnvironment(
        trim_blocks=False, lstrip_blocks=False, extensions=["jinja2.ext.loopcontrols"]
    )
    env.filters["tojson"] = lambda value, **kw: json.dumps(value, ensure_ascii=False, **kw)
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env


def render(
    template: str,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = False,
    **extra: Any,
) -> str | None:
    """Render *template*. Returns None if it cannot be rendered at all.

    `enable_thinking=None` leaves the variable undefined, which is how a caller
    asks for the model's own default.
    """
    try:
        compiled = _environment().from_string(template)
    except Exception:  # noqa: BLE001
        logger.debug("chat template will not compile", exc_info=True)
        return None

    context: dict[str, Any] = {
        "messages": list(messages),
        "add_generation_prompt": add_generation_prompt,
        **extra,
    }
    if enable_thinking is not None:
        context["enable_thinking"] = enable_thinking

    try:
        return str(compiled.render(**context))
    except Exception:  # noqa: BLE001
        # A template that needs a variable we did not supply — `tools`, a BOS
        # token — is not an error worth failing a turn over; the caller falls
        # back to the runtime's own renderer.
        logger.debug("chat template render failed", exc_info=True)
        return None


def render_for(
    model_dir: Path,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool | None = False,
) -> str | None:
    """Load and render this export's template in one step."""
    template = load_template(model_dir)
    if not template:
        return None
    return render(template, messages, enable_thinking=enable_thinking)
