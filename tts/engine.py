"""Audio8 TTS — simplified wrapper for Keylane."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from daemon.paths import TTS_MODEL_DIR, VOICES_DIR

logger = logging.getLogger(__name__)
MODEL_REPO = "Audio8/Audio8-TTS-Preview-0.1b"


def model_ready() -> bool:
    return TTS_MODEL_DIR.exists() and any(TTS_MODEL_DIR.iterdir())


def speak(text: str, voice: str = "british-man") -> None:
    if not text.strip():
        return
    if not model_ready():
        logger.warning("Audio8 model not downloaded")
        return

    try:
        import numpy as np
        import soundfile as sf
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(str(TTS_MODEL_DIR), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(TTS_MODEL_DIR), trust_remote_code=True, torch_dtype=torch.float32
        ).to(device)

        ref_wav = VOICES_DIR / f"{voice}.wav"
        ref_text_path = VOICES_DIR / f"{voice}.txt"
        ref_text = ref_text_path.read_text(encoding="utf-8").strip() if ref_text_path.exists() else ""

        inputs = tokenizer(text[:320], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512)

        audio = out.cpu().numpy().astype(np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, 24000)
            wav_path = tmp.name

        player = shutil.which("pw-play") or shutil.which("aplay") or shutil.which("ffplay")
        if player:
            if "ffplay" in player:
                subprocess.run([player, "-nodisp", "-autoexit", wav_path], check=False)
            else:
                subprocess.run([player, wav_path], check=False)
    except Exception:  # noqa: BLE001
        logger.exception("Audio8 speak failed")
