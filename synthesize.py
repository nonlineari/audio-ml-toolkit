#!/usr/bin/env python3
"""Synthesize a text script in the target speaker's voice using Coqui XTTS-v2.

Inputs:
  transcription.txt  - the script to render (path given on CLI)
  audio.wav          - the speaker reference (path given on CLI, 6-60 s of clean
                       speech recommended; longer is fine, XTTS uses an embedding).

Output:
  synthesized.wav    - 24 kHz mono PCM, concatenated per-sentence synthesis.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # Replace paragraph / line separators and any weird whitespace with spaces
    text = re.sub(r"[\u2028\u2029\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str, max_len: int = 220) -> list[str]:
    """Split text on sentence boundaries, keeping chunks <= max_len chars."""
    # Split on ., !, ?, ; followed by whitespace.
    pieces = re.split(r"(?<=[\.\!\?\;])\s+", text)
    chunks: list[str] = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if len(p) <= max_len:
            chunks.append(p)
            continue
        # Further split long pieces on commas
        sub = re.split(r",\s+", p)
        buf = ""
        for s in sub:
            candidate = (buf + ", " + s).strip(", ") if buf else s
            if len(candidate) > max_len and buf:
                chunks.append(buf.strip(", "))
                buf = s
            else:
                buf = candidate
        if buf:
            chunks.append(buf.strip(", "))
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Path to text file to synthesize")
    ap.add_argument("--speaker-wav", required=True, help="Path to speaker reference wav")
    ap.add_argument("--out", required=True, help="Output wav path")
    ap.add_argument("--language", default="en")
    ap.add_argument("--temperature", type=float, default=0.55,
                    help="XTTS sampling temperature (lower=more stable)")
    ap.add_argument("--repetition-penalty", type=float, default=5.0,
                    help="Penalize repeated tokens (higher=less stutter)")
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.85)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--single-shot", action="store_true",
                    help="Feed the whole text to XTTS with enable_text_splitting=True")
    args = ap.parse_args()

    text_raw = Path(args.text).read_text(encoding="utf-8")
    text = normalize(text_raw)
    chunks = split_sentences(text)
    if not chunks:
        print("No text to synthesize", file=sys.stderr)
        return 1
    print(f"Input: {len(text)} chars -> {len(chunks)} chunks")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] ({len(c)}) {c[:80]}{'...' if len(c) > 80 else ''}")

    # Import heavy deps only after arg parsing
    import numpy as np
    from scipy.io.wavfile import write as wav_write

    # Workaround: PyTorch 2.6+ changed torch.load default to weights_only=True,
    # which rejects Coqui's pickled checkpoints. Force weights_only=False BEFORE
    # importing TTS (the checkpoints come from the official Coqui CDN).
    import torch as _torch
    _orig_torch_load = _torch.load
    def _patched_load(*a, **kw):
        kw["weights_only"] = False
        return _orig_torch_load(*a, **kw)
    _torch.load = _patched_load

    from TTS.api import TTS

    print("Loading XTTS-v2 (first run will download ~1.8 GB)...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
              progress_bar=False)

    # Keep on CPU by default to avoid MPS edge cases; user can flip to mps.
    # XTTS has conv ops that are slow on MPS; CPU is reliable.
    device = "cpu"
    try:
        tts.to(device)
    except Exception:
        pass

    sr = getattr(tts.synthesizer, "output_sample_rate", 24000) or 24000

    xtts_kwargs = dict(
        speaker_wav=args.speaker_wav,
        language=args.language,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        top_k=args.top_k,
        top_p=args.top_p,
        speed=args.speed,
    )

    all_audio: list[np.ndarray] = []
    silence = np.zeros(int(0.15 * sr), dtype=np.float32)  # 150 ms gap
    if args.single_shot:
        print("Synthesizing in single-shot mode (XTTS splits internally)")
        wav = tts.tts(text=text, enable_text_splitting=True, **xtts_kwargs)
        all_audio.append(np.asarray(wav, dtype=np.float32))
    else:
        for i, chunk in enumerate(chunks, 1):
            print(f"Synthesizing chunk {i}/{len(chunks)}")
            wav = tts.tts(text=chunk, **xtts_kwargs)
            wav = np.asarray(wav, dtype=np.float32)
            all_audio.append(wav)
            all_audio.append(silence)

    out_arr = np.concatenate(all_audio) if all_audio else np.zeros(sr, dtype=np.float32)
    # Peak-normalize to avoid clipping, then convert to int16
    peak = float(np.max(np.abs(out_arr))) or 1.0
    out_arr = out_arr / peak * 0.97
    out_int16 = (out_arr * 32767.0).astype(np.int16)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wav_write(args.out, sr, out_int16)
    print(f"Wrote {args.out} ({len(out_int16)/sr:.2f}s @ {sr} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
