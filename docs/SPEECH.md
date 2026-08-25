# Speech

Keylane can read answers aloud, locally. No cloud service is involved and
nothing leaves the machine — the same rule the rest of Keylane follows.

Turn it on under **Control panel → Assistant → Read aloud**.

## The engine

Speech is **Audio8 TTS** (`Audio8/Audio8-TTS-Preview-0.1b`), a 0.1B neural
model that runs in the gateway process on torch. It replaces the shell-out
synthesisers Keylane used before — there is one engine now, not three.

It is **zero-shot**: rather than shipping a fixed set of voices, it clones one
from a short recording you provide. Out of the box it speaks in its own voice,
which needs no setup.

### Getting the model

The weights are about 1.7 GB and are not bundled. Download them into
`models/tts/Audio8-TTS-Preview-0.1b` under the gateway's directory:

```bash
cd ~/.local/share/ai-gateway
./.venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Audio8/Audio8-TTS-Preview-0.1b', \
  local_dir='models/tts/Audio8-TTS-Preview-0.1b')"
```

Until it is there the Assistant tab says so, and read-aloud stays off rather
than failing at the moment you press the button.

The Python side needs `torch`, `transformers`, `soundfile` and `safetensors`
(see `requirements.txt`). If one is missing the panel names it.

### Cloning a voice

Put a clean recording in the `voices/` directory beside a `.txt` file holding
**exactly** what the clip says:

```
voices/alba.wav      a few seconds of clear speech
voices/alba.txt      the words spoken in alba.wav
```

Press **Rediscover** and "Alba" appears in the voice list. A clip without a
matching transcript is skipped rather than half-used — the model conditions on
the transcript, and a wrong one degrades the clone badly.

Ten to thirty seconds of clean, single-speaker audio works well. Background
noise is cloned along with the voice.

## Settings

| Setting | Meaning |
| --- | --- |
| Show a read-aloud button | Adds the speaker button to answers |
| Read every answer automatically | Speaks each answer as it arrives |
| Engine | Audio8 TTS — the only engine |
| Voice | The built-in voice, or one cloned from `voices/` |
| Rate | **Ignored.** Audio8 has no speed control; the panel greys it out |
| Pitch | **Ignored.** Audio8 has no pitch control |

**Test** speaks a sample line so you can hear a voice before committing to it.

## What actually gets read

Answers are [canvases](canvas.html), not prose, so they are flattened for
speech rather than read literally:

- Titles, summaries, paragraphs and callouts are read as written.
- Stat tiles are read as "Free: 237 GB".
- **Tables are described** — "a table of 4 rows" — not read cell by cell.
- **Code and command output are not read**; you are told it is on screen.
- Markdown, URLs and bullet characters are stripped, so the synthesiser reads
  words rather than punctuation.

Long answers are cut at a sentence boundary rather than mid-word, and what
survives is split into sentence-sized chunks before synthesis — one generation
pass cannot cover an arbitrarily long answer, so the pieces are synthesised in
turn and stitched with a short gap.

## API

```http
GET  /api/speech            engines, voices and current settings
POST /api/speech/refresh    re-probe (after installing an engine or a voice)
POST /api/speech/speak      {"text": "...", "engine": "...", "voice": "..."}
POST /api/speech/stop       silence whatever is speaking
```

`speak` blocks until the speech finishes, and its timeout scales with the
length of the text.

## Notes

- Playback goes through PipeWire (`pw-play`), falling back to PulseAudio or
  ALSA. If none is present Keylane says so rather than appearing to work.
- Synthesis runs on a worker thread, so the gateway keeps answering while it
  speaks. The model loads on first use and stays resident — the first line
  spoken after a restart is slower than the rest.
- **Speed depends entirely on the device.** Measured on a CPU-only torch
  build, synthesis runs about **10x slower than realtime** — 43 s of compute
  for 4 s of speech. The read-aloud cap follows suit: 320 characters on CPU,
  2400 on CUDA. The Assistant tab says which device it found.
  Installing a CUDA build of torch in the gateway venv is the single biggest
  improvement available:
  `.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu130 torch`
- `stop` silences playback *and* cancels a synthesis still running. The
  generation thread cannot be interrupted, but its audio is thrown away rather
  than spoken a minute after you gave up.
- The model is under the Audio8 Community License: free for non-commercial use
  and for companies under $2M revenue, separate licence above that.
