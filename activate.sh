#!/usr/bin/env zsh
# audio-ml-toolkit/activate.sh
# Shell helper functions for the audio ML toolkit.
#
# Usage:
#   source ~/audio-ml-toolkit/activate.sh
#
# Then call:
#   use-whisper   → activates ~/whisperx-env  (WhisperX + OpenAI Whisper)
#   use-mfa       → activates mfa conda env   (Montreal Forced Aligner)
#   use-tts       → activates ~/tts-env       (Coqui TTS)
#   ml-status     → shows which env is active
#   ml-help       → prints this summary

# ── WhisperX ──────────────────────────────────────────────────────────────────
use-whisper() {
  if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠  Deactivating conda env '$CONDA_DEFAULT_ENV' first..."
    conda deactivate
  fi
  if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "⚠  Deactivating venv '$(basename $VIRTUAL_ENV)' first..."
    deactivate
  fi
  source "$HOME/whisperx-env/bin/activate"
  echo "✓  WhisperX env active  (Python $(python --version 2>&1 | cut -d' ' -f2))"
  echo "   whisperx, openai-whisper, faster-whisper, pyannote.audio, torch, torchaudio"
}

# ── MFA ───────────────────────────────────────────────────────────────────────
use-mfa() {
  if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "⚠  Deactivating venv '$(basename $VIRTUAL_ENV)' first..."
    deactivate
  fi
  conda activate mfa
  echo "✓  MFA conda env active  (Python $(python --version 2>&1 | cut -d' ' -f2))"
  echo "   montreal-forced-aligner 3.3.9 | models: english_us_arpa"
}

# ── Coqui TTS ─────────────────────────────────────────────────────────────────
use-tts() {
  if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠  Deactivating conda env '$CONDA_DEFAULT_ENV' first..."
    conda deactivate
  fi
  if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "⚠  Deactivating venv '$(basename $VIRTUAL_ENV)' first..."
    deactivate
  fi
  source "$HOME/tts-env/bin/activate"
  echo "✓  Coqui TTS env active  (Python $(python --version 2>&1 | cut -d' ' -f2))"
  echo "   TTS 0.22.0 | torch 2.11.0 | MPS available"
}

# ── Status ────────────────────────────────────────────────────────────────────
ml-status() {
  echo "── Audio ML Environment Status ──────────────────────"
  if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "  Active venv : $(basename $VIRTUAL_ENV)  ($VIRTUAL_ENV)"
  elif [[ -n "$CONDA_DEFAULT_ENV" && "$CONDA_DEFAULT_ENV" != "base" ]]; then
    echo "  Active conda: $CONDA_DEFAULT_ENV"
  else
    echo "  No ML environment active"
  fi
  echo ""
  echo "  Available environments:"
  echo "    use-whisper  →  ~/whisperx-env  (Python 3.13)"
  echo "    use-mfa      →  conda:mfa       (Python 3.11)"
  echo "    use-tts      →  ~/tts-env       (Python 3.10)"
  echo "─────────────────────────────────────────────────────"
}

# ── Help ──────────────────────────────────────────────────────────────────────
ml-help() {
  cat <<'EOF'
Audio ML Toolkit — Quick Reference
════════════════════════════════════════════════════════════════

ACTIVATION SHORTCUTS
  use-whisper      Activate WhisperX + OpenAI Whisper env
  use-mfa          Activate Montreal Forced Aligner conda env
  use-tts          Activate Coqui TTS env
  ml-status        Show current active environment
  ml-help          Show this help

WHISPER / WHISPERX  (after use-whisper)
  whisper audio.mp3 --model large-v3 --language en --output_format srt
  whisperx audio.mp3 --model large-v2 --align_model WAV2VEC2_ASR_LARGE_LV60K_960H

MFA  (after use-mfa)
  mfa align ./corpus english_us_arpa english_us_arpa ./output
  mfa validate ./corpus english_us_arpa english_us_arpa
  mfa model download acoustic <name>
  mfa model download dictionary <name>

COQUI TTS  (after use-tts)
  tts --list_models
  tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
      --text "Hello." --speaker_wav ref.wav --language en --out_path out.wav

MPS FALLBACK (if TTS crashes on Metal)
  PYTORCH_ENABLE_MPS_FALLBACK=1 tts ...

FULL PIPELINE: Transcribe → Align → Synthesize
  See ~/audio-ml-toolkit/README.md

════════════════════════════════════════════════════════════════
EOF
}

# Print a brief welcome on source
echo "Audio ML Toolkit loaded. Run 'ml-help' for usage, 'ml-status' for env info."
