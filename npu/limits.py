"""Prompt-length limits for the NPU pipelines.

The NPU compiles a fixed maximum prompt length into the pipeline, so a prompt
that exceeds it is not slow — it throws. Both the value compiled in and the
budget the prompt is trimmed to have to come from here, because when they drift
apart the trimming stops protecting anything, which is exactly what happened:
the VLM budget said 10000 characters while the compiled limit was 1024 tokens.
"""

from __future__ import annotations

import os

from npu.kind import PipelineKind

# Tokens compiled into an NPU pipeline. This is not a soft limit: it is built
# into the compiled model, so exceeding it throws rather than truncates.
#
# It used to be believed that raising this was expensive — that 1024 to 4096
# took a load from thirteen seconds to half an hour. That was true against an
# older OpenVINO. Measured on 2026-09-03 against 2026.3 on an NPU 3720, a cold
# 7B compile is 64 s at 1024 and 60 s at 4096: indistinguishable. Since 2025.3
# the pipeline defaults to PREFILL_HINT: DYNAMIC and chunks the prefill
# (NPUW_LLM_PREFILL_CHUNK_SIZE, default 1024) rather than compiling a static
# graph the full width.
#
# What it does still cost is time to first token, and that cost is flat: the
# same measurement put TTFT at ~15 s for a 38-token prompt whatever the reply
# length, with decode at ~0.25 s/token afterwards. Prefill runs a whole chunk
# even for a short prompt, so a smaller MAX_PROMPT_LEN is worth trying if your
# turns are short.
#
# The prompt's required sections need about 1700 tokens, so anything below
# ~2048 cannot serve a turn at all. Override with KEYLANE_NPU_MAX_PROMPT_TOKENS;
# changing it invalidates the compile cache, so the next load is slow either way.
NPU_MAX_PROMPT_TOKENS = int(os.environ.get("KEYLANE_NPU_MAX_PROMPT_TOKENS", "4096"))

# Characters per token, deliberately pessimistic. Prose runs nearer 4, but a
# prompt full of punctuation, JSON and tool names runs far lower — the system
# prompt that triggered this measured 2.8 — and being wrong here throws.
CHARS_PER_TOKEN = 2.6

# Room left for the generated reply and the chat scaffolding.
RESERVE_TOKENS = 512


def npu_prompt_budget_tokens() -> int:
    """How many tokens of prompt an NPU pipeline can actually take."""
    return max(NPU_MAX_PROMPT_TOKENS - RESERVE_TOKENS, 256)


def npu_prompt_budget_chars() -> int:
    """How many characters of prompt an NPU pipeline can actually take."""
    return int(npu_prompt_budget_tokens() * CHARS_PER_TOKEN)


def prompt_budget_tokens(device: str, kind: PipelineKind) -> int:
    """The token budget for one pipeline on one device.

    This is the honest unit: the compiled limit is a token count, and the
    tokenizer that would answer exactly is loaded in the same process. The
    character budget below exists only for callers that have no tokenizer to
    ask — prompt assembly sheds optional sections long before a pipeline is
    chosen — and it stays deliberately pessimistic for that reason.
    """
    if device.upper() != "NPU":
        # CPU and GPU pipelines are bounded by patience, not by a compiled limit.
        return int(24000 / CHARS_PER_TOKEN)
    return npu_prompt_budget_tokens()


def prompt_budget_chars(device: str, kind: PipelineKind) -> int:
    """The character budget for one pipeline on one device."""
    if device.upper() != "NPU":
        return 24000
    return npu_prompt_budget_chars()
