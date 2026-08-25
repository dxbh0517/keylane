#!/usr/bin/env python3
"""Build reference clips for Audio8 voice cloning.

Audio8 is zero-shot: it has no voices of its own, it copies one from a
recording. So a "voice pack" is a set of short clips plus the exact words
spoken in each — which is what this script assembles.

The clips come from two openly licensed corpora of *real* speech, read through
Hugging Face's dataset row API so only the handful of utterances we actually
want get downloaded rather than the whole corpus:

``ylacombe/english_dialects``  CC BY-SA 4.0 — Google/CSTR crowdsourced UK and
                               Ireland English, labelled by region and gender.
``mythicinfinity/libritts_r``  CC BY 4.0 — LibriTTS-R, restored from public
                               domain LibriVox recordings.

One utterance is a single sentence, which is too little to clone from well, so
several from the *same* speaker are concatenated into one clip of roughly
twenty seconds and their transcripts joined to match.

Usage:  scripts/fetch_voices.py [--out DIR] [--only NAME] [--force]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROWS_API = "https://datasets-server.huggingface.co/rows"

# Around twenty seconds is the sweet spot: enough for the model to pick up a
# voice, short enough that a clip stays quick to fetch and to condition on.
TARGET_SECONDS = 20.0
MAX_SECONDS = 28.0
# Sentences shorter than this are usually a fragment, and clipped breaths make
# a poor reference.
MIN_UTTERANCE_SECONDS = 2.0


@dataclass
class VoiceSpec:
    name: str
    label: str
    dataset: str
    config: str
    split: str
    text_field: str = "text"
    # Where to look in the corpus. Scanning costs one API call per page, so
    # keep it modest and let pitch selection do the rest.
    scan_rows: int = 220
    # Pitch band in Hz used to choose between candidate speakers, for corpora
    # that label neither gender nor age. None means "take the first speaker
    # with enough audio".
    pitch_range: tuple[float, float] | None = None
    attribution: str = ""
    notes: str = ""
    skip_speakers: set[str] = field(default_factory=set)


VOICES: list[VoiceSpec] = [
    VoiceSpec(
        name="british-man",
        label="British man (Southern English)",
        dataset="ylacombe/english_dialects",
        config="southern_male",
        split="train",
        attribution="Google/CSTR UK & Ireland English, CC BY-SA 4.0",
    ),
    VoiceSpec(
        name="british-woman",
        label="British woman (Southern English)",
        dataset="ylacombe/english_dialects",
        config="southern_female",
        split="train",
        attribution="Google/CSTR UK & Ireland English, CC BY-SA 4.0",
    ),
    VoiceSpec(
        name="american-woman",
        label="American woman (young adult)",
        dataset="mythicinfinity/libritts_r",
        config="clean",
        split="dev.clean",
        text_field="text_normalized",
        # LibriTTS labels neither gender nor age, so pick on measured pitch:
        # this band sits in the typical young-adult female range and well
        # clear of male speakers.
        pitch_range=(185.0, 260.0),
        attribution="LibriTTS-R (public domain LibriVox), CC BY 4.0",
        notes="Chosen by measured pitch; age is approximate.",
    ),
    VoiceSpec(
        name="american-man",
        label="American man",
        dataset="mythicinfinity/libritts_r",
        config="clean",
        split="dev.clean",
        text_field="text_normalized",
        pitch_range=(85.0, 145.0),
        attribution="LibriTTS-R (public domain LibriVox), CC BY 4.0",
    ),
    VoiceSpec(
        name="scottish-man",
        label="Scottish man",
        dataset="ylacombe/english_dialects",
        config="scottish_male",
        split="train",
        attribution="Google/CSTR UK & Ireland English, CC BY-SA 4.0",
    ),
    VoiceSpec(
        name="scottish-woman",
        label="Scottish woman",
        dataset="ylacombe/english_dialects",
        config="scottish_female",
        split="train",
        attribution="Google/CSTR UK & Ireland English, CC BY-SA 4.0",
    ),
    VoiceSpec(
        name="irish-man",
        label="Irish man",
        dataset="ylacombe/english_dialects",
        config="irish_male",
        split="train",
        attribution="Google/CSTR UK & Ireland English, CC BY-SA 4.0",
    ),
    VoiceSpec(
        name="welsh-woman",
        label="Welsh woman",
        dataset="ylacombe/english_dialects",
        config="welsh_female",
        split="train",
        attribution="Google/CSTR UK & Ireland English, CC BY-SA 4.0",
    ),
    VoiceSpec(
        name="northern-english-man",
        label="Northern English man",
        dataset="ylacombe/english_dialects",
        config="northern_male",
        split="train",
        attribution="Google/CSTR UK & Ireland English, CC BY-SA 4.0",
    ),
]


def _get_json(url: str, *, timeout: float = 90.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "keylane-voices/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _get_bytes(url: str, *, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "keylane-voices/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _audio_src(value) -> str:
    """The URL out of a row's audio cell, whichever shape it arrives in."""
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        return value.get("src") or value.get("url") or ""
    return ""


def mean_pitch(samples, rate: float) -> float:
    """Rough mean F0 in Hz, by autocorrelation over voiced frames.

    Good enough to tell a speaker apart by register, which is all it is used
    for. Deliberately dependency-free — no need for a pitch tracker to pick
    between a handful of candidates.
    """
    import numpy as np

    frame = int(rate * 0.04)
    if frame < 64 or len(samples) < frame * 4:
        return 0.0
    lo, hi = int(rate / 400.0), int(rate / 70.0)  # 70-400 Hz search band
    if hi <= lo:
        return 0.0

    estimates: list[float] = []
    for start in range(0, len(samples) - frame, frame):
        window = samples[start : start + frame]
        window = window - window.mean()
        energy = float(np.sqrt((window**2).mean()))
        if energy < 0.02:  # silence or breath, not a voiced frame
            continue
        correlation = np.correlate(window, window, mode="full")[frame - 1 :]
        segment = correlation[lo:hi]
        if segment.size == 0:
            continue
        peak = int(np.argmax(segment)) + lo
        if correlation[0] <= 0 or correlation[peak] / correlation[0] < 0.3:
            continue  # too weak a periodicity to trust
        estimates.append(rate / peak)
    if not estimates:
        return 0.0
    return float(np.median(estimates))


def _load_mono(raw: bytes):
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(io.BytesIO(raw), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), int(rate)


def _fetch_page(spec: VoiceSpec, offset: int, length: int) -> tuple[list[dict], int]:
    url = (
        f"{ROWS_API}?dataset={spec.dataset}&config={spec.config}"
        f"&split={spec.split}&offset={offset}&length={length}"
    )
    payload = _get_json(url)
    rows = [r["row"] for r in payload.get("rows", [])]
    return rows, int(payload.get("num_rows_total") or 0)


def _scan_offsets(total: int, pages: int, page: int) -> list[int]:
    """Offsets spread across the split rather than packed at the front.

    These corpora are ordered by speaker, so reading the first few hundred
    rows shows one voice and no more. Sampling across the whole split is what
    makes a choice between speakers possible at all.
    """
    if total <= page:
        return [0]
    stride = max(page, (total - page) // max(pages - 1, 1))
    offsets = [min(i * stride, total - page) for i in range(pages)]
    return sorted(set(offsets))


def collect(spec: VoiceSpec) -> tuple[list, int, str, str]:
    """Gather clips for one voice. Returns (chunks, rate, transcript, speaker)."""
    import numpy as np

    by_speaker: dict[str, list[dict]] = {}
    page = 100
    first, total = _fetch_page(spec, 0, page)
    pages = max(1, spec.scan_rows // page)
    offsets = _scan_offsets(total, pages, page) if total else [0]

    for index, offset in enumerate(offsets):
        try:
            rows = first if offset == 0 and index == 0 else _fetch_page(spec, offset, page)[0]
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"    row fetch failed at offset {offset}: {exc}", file=sys.stderr)
            continue
        if not rows:
            continue
        for row in rows:
            speaker = str(row.get("speaker_id") or "")
            if not speaker or speaker in spec.skip_speakers:
                continue
            by_speaker.setdefault(speaker, []).append(row)

    if not by_speaker:
        raise RuntimeError("no rows returned")

    # Most utterances first: a speaker we have plenty of is one we can build a
    # long enough reference from without another round of paging.
    candidates = sorted(by_speaker.items(), key=lambda kv: -len(kv[1]))

    for speaker, rows in candidates[:6]:
        chunks: list = []
        texts: list[str] = []
        rate = 0
        total = 0.0
        for row in rows:
            if total >= TARGET_SECONDS:
                break
            src = _audio_src(row.get("audio"))
            text = (row.get(spec.text_field) or "").strip()
            if not src or not text:
                continue
            try:
                samples, sample_rate = _load_mono(_get_bytes(src))
            except Exception as exc:  # noqa: BLE001
                print(f"    skipped an utterance: {exc}", file=sys.stderr)
                continue
            seconds = len(samples) / sample_rate
            if seconds < MIN_UTTERANCE_SECONDS or total + seconds > MAX_SECONDS:
                continue
            if rate and sample_rate != rate:
                continue
            rate = sample_rate
            chunks.append(samples)
            texts.append(text)
            total += seconds

        if total < MIN_UTTERANCE_SECONDS * 2 or not chunks:
            continue

        if spec.pitch_range is not None:
            pitch = mean_pitch(np.concatenate(chunks), rate)
            low, high = spec.pitch_range
            if not (low <= pitch <= high):
                print(f"    speaker {speaker}: {pitch:.0f} Hz, outside band — skipping")
                continue
            print(f"    speaker {speaker}: {pitch:.0f} Hz, in band")

        return chunks, rate, " ".join(texts), speaker

    raise RuntimeError("no speaker matched")


def build(spec: VoiceSpec, out_dir: Path, *, force: bool) -> bool:
    import numpy as np
    import soundfile as sf

    wav_path = out_dir / f"{spec.name}.wav"
    txt_path = out_dir / f"{spec.name}.txt"
    if wav_path.exists() and txt_path.exists() and not force:
        print(f"  {spec.name}: already present")
        return True

    print(f"  {spec.name}: {spec.label}")
    try:
        chunks, rate, transcript, speaker = collect(spec)
    except Exception as exc:  # noqa: BLE001
        print(f"    FAILED: {exc}", file=sys.stderr)
        return False

    # A short gap between utterances, so the join does not sound like a splice.
    gap = np.zeros(int(rate * 0.25), dtype=np.float32)
    audio = np.concatenate([part for c in chunks for part in (c, gap)][:-1])

    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio * (0.95 / peak)  # normalise; a quiet reference clones badly

    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, audio, rate, subtype="PCM_16")
    txt_path.write_text(transcript + "\n", encoding="utf-8")
    print(
        f"    {len(audio) / rate:.1f}s at {rate} Hz, speaker {speaker} "
        f"({len(transcript)} chars)"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "voices",
        help="where to write the clips (default: the repo's voices/)",
    )
    parser.add_argument("--only", action="append", help="build just this voice, repeatable")
    parser.add_argument("--force", action="store_true", help="rebuild clips that exist")
    parser.add_argument("--list", action="store_true", help="list the voices and exit")
    args = parser.parse_args()

    if args.list:
        for spec in VOICES:
            print(f"{spec.name:24s} {spec.label}  [{spec.attribution}]")
        return 0

    wanted = [v for v in VOICES if not args.only or v.name in args.only]
    if not wanted:
        print("Nothing matched --only", file=sys.stderr)
        return 2

    print(f"Building {len(wanted)} voice(s) into {args.out}")
    ok = sum(build(spec, args.out, force=args.force) for spec in wanted)
    print(f"\n{ok}/{len(wanted)} voices ready")

    credits = args.out / "CREDITS.md"
    if ok:
        lines = [
            "# Voice reference clips",
            "",
            "Each `.wav` is a reference Audio8 clones from; the matching `.txt`",
            "is exactly what it says. Built by `scripts/fetch_voices.py`.",
            "",
        ]
        for spec in VOICES:
            if (args.out / f"{spec.name}.wav").exists():
                note = f" {spec.notes}" if spec.notes else ""
                lines.append(f"- **{spec.name}** — {spec.label}. {spec.attribution}.{note}")
        credits.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {credits}")
    return 0 if ok == len(wanted) else 1


if __name__ == "__main__":
    raise SystemExit(main())
