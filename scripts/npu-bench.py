#!/usr/bin/env python3
"""Measure what a model actually costs on this machine, and write it down.

Every timeout and every "this takes minutes" comment in the tree was once
someone's measurement on someone's hardware, and they aged badly: the deadlines
in npu/probe.py were set when a 4096-token compile really did take half an
hour, and on OpenVINO 2026.3 the same compile is a minute. A comment cannot be
re-run. This can.

It reports three things, because they fail differently:

**Cold compile** — the first load with an empty cache. This is what the probe's
deadline has to cover.

**Warm load** — the second load, from the cache. If this is not seconds, the
cache is not being hit and every restart pays full price.

**Seconds per generate() call, at several reply lengths.** The important one,
and the one nothing measured before. On the NPU there is a large fixed cost per
call that does not depend on how many tokens you ask for, and the agent loop
makes one call per ReAct iteration — so this number, times the iteration count,
is the floor under every answer.

    PYTHONPATH=. python scripts/npu-bench.py                  # the active model
    PYTHONPATH=. python scripts/npu-bench.py --model qwen3-8b --device NPU
    PYTHONPATH=. python scripts/npu-bench.py --quick          # skip cold compile
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daemon.paths import DATA  # noqa: E402
from models.catalog import default_model_id, get_model, load_catalog  # noqa: E402

REPORT_PATH = DATA / "bench.json"

# Short, medium and long, because the interesting result is that the first two
# often cost the same — the cost is per call, not per token.
REPLY_LENGTHS = (8, 64, 256)

PROMPT = "Write a short paragraph about the sea."


def _machine() -> dict[str, Any]:
    """Everything that would make these numbers not apply somewhere else."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "unknown",
    }
    try:
        import openvino as ov

        info["openvino"] = ov.__version__
        core = ov.Core()
        info["devices"] = {}
        for device in core.available_devices:
            try:
                info["devices"][device] = str(core.get_property(device, "FULL_DEVICE_NAME"))
            except Exception:  # noqa: BLE001
                info["devices"][device] = "?"
        if "NPU" in core.available_devices:
            for prop in ("DEVICE_ARCHITECTURE", "NPU_DRIVER_VERSION", "NPU_COMPILER_TYPE"):
                try:
                    info[prop.lower()] = str(core.get_property("NPU", prop))
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        info["openvino"] = "not importable"
    try:
        import openvino_genai as ov_genai

        info["openvino_genai"] = getattr(ov_genai, "__version__", "?")
    except Exception:  # noqa: BLE001
        pass
    return info


def _timed_load(model_dir: Path, device: str, cache: Path, kind: str) -> tuple[Any, float]:
    from npu.pipeline_config import create_pipeline

    start = time.time()
    pipe = create_pipeline(model_dir, device, cache, kind)  # type: ignore[arg-type]
    return pipe, time.time() - start


def run(model_id: str, device: str, *, quick: bool) -> dict[str, Any]:
    entry = get_model(model_id)
    if entry is None:
        raise SystemExit(f"unknown model: {model_id}")
    if not entry.is_downloaded():
        raise SystemExit(
            f"{model_id} is not downloaded. Fetch it first:\n"
            f"  curl -X POST localhost:9100/models/download "
            f'-H "content-type: application/json" -d \'{{"model_id": "{model_id}"}}\''
        )
    if entry.runtime != "openvino":
        raise SystemExit(
            f"{model_id} runs on {entry.runtime}; this bench measures the OpenVINO path"
        )

    kind = entry.backend.model_kind(entry.model_dir)
    result: dict[str, Any] = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model_id,
        "hf_repo": entry.hf_repo,
        "device": device,
        "pipeline": kind,
        "machine": _machine(),
    }

    cold_cache = DATA / "cache" / "bench-cold"
    if not quick:
        # A cold compile means an empty cache. Its own directory, so the real
        # one is never thrown away by running a benchmark.
        shutil.rmtree(cold_cache, ignore_errors=True)
        cold_cache.mkdir(parents=True, exist_ok=True)
        print(f"cold compile of {model_id} on {device} — this is the slow one…", flush=True)
        pipe, seconds = _timed_load(entry.model_dir, device, cold_cache, kind)
        result["cold_compile_s"] = round(seconds, 1)
        print(f"  cold compile: {seconds:.1f}s", flush=True)
        del pipe

        blobs = list(cold_cache.glob("*.blob"))
        result["cache_bytes"] = sum(b.stat().st_size for b in blobs)

        print("warm load from the cache it just wrote…", flush=True)
        pipe, seconds = _timed_load(entry.model_dir, device, cold_cache, kind)
        result["warm_load_s"] = round(seconds, 1)
        print(f"  warm load: {seconds:.1f}s", flush=True)
    else:
        cache = entry.backend.cache_dir()
        print(f"loading {model_id} from {cache}…", flush=True)
        pipe, seconds = _timed_load(entry.model_dir, device, cache, kind)
        result["warm_load_s"] = round(seconds, 1)
        print(f"  load: {seconds:.1f}s", flush=True)

    calls: dict[str, Any] = {}
    for n in REPLY_LENGTHS:
        first: list[float] = []
        start = time.time()

        def _streamer(_piece: str, _seen: list[float] = first, _t0: float = start) -> bool:
            if not _seen:
                _seen.append(time.time() - _t0)
            return False  # False means "keep going"

        pipe.generate(PROMPT, max_new_tokens=n, streamer=_streamer)
        seconds = time.time() - start
        ttft = round(first[0], 1) if first else None
        calls[str(n)] = {"seconds": round(seconds, 1), "time_to_first_token_s": ttft}
        suffix = f"  (first token {ttft:.1f}s)" if ttft is not None else ""
        print(f"  {n:>4} tokens: {seconds:5.1f}s{suffix}", flush=True)
    result["generate"] = calls

    shortest = calls[str(REPLY_LENGTHS[0])]["seconds"]
    longest = calls[str(REPLY_LENGTHS[-1])]["seconds"]
    result["fixed_cost_per_call_s"] = shortest
    if longest > shortest:
        spread = REPLY_LENGTHS[-1] - REPLY_LENGTHS[0]
        result["marginal_tokens_per_s"] = round(spread / (longest - shortest), 2)

    shutil.rmtree(cold_cache, ignore_errors=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="", help="catalog id (default: the startup model)")
    parser.add_argument("--device", default="", help="NPU, GPU or CPU")
    parser.add_argument("--quick", action="store_true", help="skip the cold compile")
    parser.add_argument("--json", action="store_true", help="print the report and write nothing")
    args = parser.parse_args()

    model_id = args.model or default_model_id()
    entry = get_model(model_id)
    device = args.device or (entry.resolve_device(load_catalog()[1]) if entry else "NPU")

    report = run(model_id, device.upper(), quick=args.quick)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if REPORT_PATH.is_file():
        try:
            history = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            history = []
    if not isinstance(history, list):
        history = [history]
    # Newest first, and keep enough history to see a regression across an
    # OpenVINO bump without the file growing without bound.
    history.insert(0, report)
    REPORT_PATH.write_text(json.dumps(history[:20], indent=2) + "\n", encoding="utf-8")

    print(f"\nwritten to {REPORT_PATH}")
    fixed = report.get("fixed_cost_per_call_s")
    if fixed:
        budget = 12
        print(
            f"\nEvery generate() call costs at least {fixed:.1f}s here. The agent makes one "
            f"per ReAct iteration, so a turn using three tools pays about "
            f"{fixed * 4:.0f}s of floor before any useful token."
        )
        print(f"(the iteration budget is {budget}, so the worst case is {fixed * budget:.0f}s)")


if __name__ == "__main__":
    main()
