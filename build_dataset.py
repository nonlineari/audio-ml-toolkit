#!/usr/bin/env python3
"""Build a TTS training dataset + MFA-compatible corpus from a WhisperX run.

Reads:  whisperx_out/audio.json (from whisperx CLI, --output_format all)
        audio.wav               (16 kHz mono PCM, from ffmpeg step)

Writes: dataset/
          wavs/clip_0001.wav, clip_0002.wav, ...
          txts/clip_0001.txt, clip_0002.txt, ...
          metadata.csv          Coqui-style "wav|text" (pipe-delimited)
          corpus/kobe/clip_0001.wav + clip_0001.lab, ...   MFA layout

Each segment in the WhisperX transcript becomes one clip. Segments shorter
than MIN_DUR or longer than MAX_DUR are skipped so the resulting clips are
usable for TTS training / forced alignment.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "audio.wav"
TRANSCRIPT = ROOT / "whisperx_out" / "audio.json"
OUT = ROOT / "dataset"
SPEAKER = "kobe"

MIN_DUR = 1.0   # seconds
MAX_DUR = 30.0  # seconds
PAD = 0.05      # seconds of padding at each end when slicing


def clean_text(t: str) -> str:
    t = t.strip()
    # Collapse internal whitespace
    t = re.sub(r"\s+", " ", t)
    return t


def main() -> int:
    if not AUDIO.exists():
        print(f"Missing {AUDIO}", file=sys.stderr)
        return 1
    if not TRANSCRIPT.exists():
        print(f"Missing {TRANSCRIPT}", file=sys.stderr)
        return 1

    data = json.loads(TRANSCRIPT.read_text())
    segments = data.get("segments", [])
    if not segments:
        print("No segments in transcript", file=sys.stderr)
        return 1

    # Reset output dirs
    if OUT.exists():
        shutil.rmtree(OUT)
    wav_dir = OUT / "wavs"
    txt_dir = OUT / "txts"
    corpus_dir = OUT / "corpus" / SPEAKER
    for d in (wav_dir, txt_dir, corpus_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = OUT / "metadata.csv"
    kept = 0
    total_dur = 0.0

    with manifest_path.open("w") as manifest:
        for i, seg in enumerate(segments, start=1):
            start = max(0.0, float(seg["start"]) - PAD)
            end = float(seg["end"]) + PAD
            dur = end - start
            text = clean_text(seg.get("text", ""))
            if not text:
                continue
            if dur < MIN_DUR or dur > MAX_DUR:
                print(f"skip #{i} dur={dur:.2f}s text={text!r}")
                continue

            stem = f"clip_{i:04d}"
            wav_out = wav_dir / f"{stem}.wav"
            txt_out = txt_dir / f"{stem}.txt"
            corpus_wav = corpus_dir / f"{stem}.wav"
            corpus_lab = corpus_dir / f"{stem}.lab"

            # Slice with ffmpeg (copy-safe for PCM)
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-i", str(AUDIO),
                "-ac", "1", "-ar", "16000",
                "-acodec", "pcm_s16le",
                str(wav_out),
            ]
            subprocess.run(cmd, check=True)

            txt_out.write_text(text + "\n")
            # Duplicate into MFA corpus layout
            shutil.copy2(wav_out, corpus_wav)
            corpus_lab.write_text(text + "\n")

            # Coqui-style manifest: relative wav path | text
            rel = wav_out.relative_to(OUT)
            manifest.write(f"{rel}|{text}\n")
            kept += 1
            total_dur += dur

    print(
        f"Wrote {kept}/{len(segments)} clips, total {total_dur:.1f}s "
        f"({total_dur/60:.2f} min)"
    )
    print(f"  manifest: {manifest_path}")
    print(f"  wavs:     {wav_dir}")
    print(f"  txts:     {txt_dir}")
    print(f"  corpus:   {corpus_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
