"""Prompt-length limits for the NPU pipelines.

The NPU compiles a fixed maximum prompt length into the pipeline, so a prompt
that exceeds it is not slow — it throws. Both the value compiled in and the
budget the prompt is trimmed to have to come from here, because when they drift
apart the trimming stops protecting anything, which is exactly what happened:
the VLM budget said 10000 characters while the compiled limit was 1024 tokens.
"""

from __future__ import annotations

from npu.kind import PipelineKind

# Tokens compiled into an NPU pipeline. Raising it costs compile time and NPU
# memory, so it buys headroom for the system prompt and a few turns of history,
# not a large context.
NPU_MAX_PROMPT_TOKENS = 4096

# Characters per token, deliberately pessimistic. Prose runs nearer 4, but a
# prompt full of punctuation, JSON and tool names runs far lower — the system
# prompt that triggered this measured 2.8 — and being wrong here throws.
CHARS_PER_TOKEN = 2.6

# Room left for the generated reply and the chat scaffolding.
RESERVE_TOKENS = 512


def npu_prompt_budget_chars() -> int:
    """How many characters of prompt an NPU pipeline can actually take."""
    usable = max(NPU_MAX_PROMPT_TOKENS - RESERVE_TOKENS, 256)
    return int(usable * CHARS_PER_TOKEN)


def prompt_budget_chars(device: str, kind: PipelineKind) -> int:
    """The character budget for one pipeline on one device."""
    if device.upper() != "NPU":
        # CPU and GPU pipelines are bounded by patience, not by a compiled limit.
        return 24000
    return npu_prompt_budget_chars()
