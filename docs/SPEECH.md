# Speech

Keylane can read answers aloud, locally. No cloud service is involved and
nothing leaves the machine — the same rule the rest of Keylane follows.

Turn it on under **Control panel → Assistant → Read aloud**.

## Engines

Three are supported. Keylane probes for them at startup and lists whatever this
machine actually has, so the panel never offers something that is not installed.

| Engine | Sounds like | Install |
| --- | --- | --- |
| **Piper** | Neural, natural. The best of the three. | `sudo dnf install piper` plus a voice |
| **eSpeak NG** | Robotic, instant, 140+ languages | `sudo dnf install espeak-ng` |
| **Flite** | Small and fast, a few English voices | `sudo dnf install flite` |

If the engine you picked is missing, Keylane falls back to the best one present
rather than failing silently.

### Piper voices

Piper needs a voice model — the binary alone cannot speak. Download an `.onnx`
and its `.onnx.json` into `~/.local/share/piper/voices`:

```bash
mkdir -p ~/.local/share/piper/voices && cd ~/.local/share/piper/voices
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium
curl -LO $BASE/en_GB-alba-medium.onnx
curl -LO $BASE/en_GB-alba-medium.onnx.json
```

Press **Rediscover** on the Assistant tab and it appears in the voice list.
Voices for other languages are in the same repository under their language code.

## Settings

| Setting | Meaning |
| --- | --- |
| Show a read-aloud button | Adds the speaker button to answers |
| Read every answer automatically | Speaks each answer as it arrives |
| Engine | Which synthesiser to use |
| Voice | Engine-specific; Piper voices are files, eSpeak's are language codes |
| Rate | Percent of the engine's normal speed |
| Pitch | 0–99, eSpeak only |

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

Long answers are cut at a sentence boundary rather than mid-word.

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

- Playback goes through PipeWire (`pw-play` / `pw-cat`), falling back to
  PulseAudio or ALSA. If none is present Keylane says so rather than appearing
  to work.
- Flite takes its text as a command-line argument rather than on stdin —
  worth knowing if you add another engine, since piping to it silently
  produces an empty audio file.
