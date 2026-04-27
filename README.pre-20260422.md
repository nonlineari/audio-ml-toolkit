# Audio ML Toolkit — Environment Reference

> MacOS (Apple Silicon) · April 2026
> All tools installed locally. MPS (Metal) available for PyTorch inference.

---

## Environment Map

| Tool | Type | Location | Python |
|------|------|----------|--------|
| WhisperX + OpenAI Whisper | venv | `~/whisperx-env` | 3.13 |
| Montreal Forced Aligner (MFA) | conda env | `mfa` | 3.11 |
| Coqui TTS | venv | `~/tts-env` | 3.10 |

> **Rule:** Only one environment should be active at a time. Always `deactivate` or open a new shell before switching.

---

## 1. WhisperX / OpenAI Whisper

### Activate
```bash
source ~/whisperx-env/bin/activate
```

### Deactivate
```bash
deactivate
```

### Versions installed
- `whisperx 3.8.5`
- `openai-whisper 20250625`
- `torch`, `torchaudio`, `faster-whisper`, `pyannote.audio`, `transformers`, `soundfile`, `numpy`, `tqdm`

### Usage — OpenAI Whisper (CLI)
```bash
# Transcribe a file (auto-detects language)
whisper audio.mp3 --model medium

# Force English, output SRT subtitle
whisper audio.mp3 --model large-v3 --language en --output_format srt

# Available models: tiny, base, small, medium, large, large-v2, large-v3
```

### Usage — WhisperX (Python, word-level alignment)
```python
import whisperx

device = "mps"   # Use "cpu" if MPS causes issues
audio = whisperx.load_audio("audio.mp3")

# 1. Transcribe
model = whisperx.load_model("large-v2", device, compute_type="float32")
result = model.transcribe(audio, batch_size=8)

# 2. Align to get word-level timestamps
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device)

# 3. (Optional) Diarize speakers
diarize_model = whisperx.DiarizationPipeline(use_auth_token="YOUR_HF_TOKEN", device=device)
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)

print(result["segments"])  # word-level timestamps + speaker labels
```

> **Note:** Speaker diarization requires a Hugging Face token with access to `pyannote/speaker-diarization-3.1`.

---

## 2. Montreal Forced Aligner (MFA)

### Activate
```bash
conda activate mfa
```

### Deactivate
```bash
conda deactivate
```

### Version installed
- `montreal-forced-aligner 3.3.9`

### Resources downloaded
- Acoustic model: `english_us_arpa`
- Dictionary: `english_us_arpa`

### Usage — Align audio to transcript
```bash
# Directory structure expected:
# corpus/
#   speaker1/
#     utterance1.wav
#     utterance1.lab   ← plain text transcript, same stem as wav

mfa align \
  /path/to/corpus \
  english_us_arpa \
  english_us_arpa \
  /path/to/output_textgrids

# Output: TextGrid files with phone/word-level alignments
```

### Usage — Download more models
```bash
# List all available pretrained models
mfa model list acoustic
mfa model list dictionary

# Download additional language resources
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
```

### Usage — Validate corpus before aligning
```bash
mfa validate /path/to/corpus english_us_arpa english_us_arpa
```

### Usage — Train a new acoustic model from scratch
```bash
mfa train \
  /path/to/corpus \
  english_us_arpa \
  /path/to/output_model.zip \
  --output_directory /path/to/textgrids
```

---

## 3. Coqui TTS (Voice Cloning)

### Activate
```bash
source ~/tts-env/bin/activate
```

### Deactivate
```bash
deactivate
```

### Versions installed
- `TTS 0.22.0`
- `torch 2.11.0`, `torchaudio 2.11.0`
- MPS available: **Yes** (Apple Silicon GPU inference works)

### Usage — List all pretrained models
```bash
tts --list_models
```

### Usage — Basic TTS (no voice cloning)
```bash
tts --text "Hello, this is a test." \
    --model_name tts_models/en/ljspeech/tacotron2-DDC \
    --out_path output.wav
```

### Usage — Voice Cloning with YourTTS (multilingual)
```bash
tts --model_name tts_models/multilingual/multi-dataset/your_tts \
    --text "Hello, this is my cloned voice." \
    --speaker_wav /path/to/reference_clip.wav \
    --language_idx en \
    --out_path cloned_output.wav
```
> `reference_clip.wav` should be 6–30 seconds of clean speech from the target speaker.

### Usage — Voice Cloning with XTTS v2 (best quality)
```bash
tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --text "Hello, this is my cloned voice." \
    --speaker_wav /path/to/reference_clip.wav \
    --language en \
    --out_path xtts_output.wav
```

### Usage — Python API (XTTS v2)
```python
from TTS.api import TTS

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text="Hello from Coqui TTS on Apple Silicon.",
    speaker_wav="/path/to/reference_clip.wav",
    language="en",
    file_path="output.wav"
)
```

### Apple Silicon notes
- Inference runs on MPS (Metal GPU) — no Linux machine needed for synthesis.
- For **fine-tuning/training** a custom model on your own voice dataset, a Linux + CUDA GPU is significantly faster. Train there, copy the `.pth` checkpoint to Mac for inference.
- If you hit MPS errors, fall back to CPU: set `PYTORCH_ENABLE_MPS_FALLBACK=1` before running.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 tts --model_name ... --text "..." --out_path out.wav
```

---

## Quick-reference helper script

Source `~/audio-ml-toolkit/activate.sh` to get shorthand shell functions:

```bash
source ~/audio-ml-toolkit/activate.sh

use-whisper    # activate ~/whisperx-env
use-mfa        # conda activate mfa
use-tts        # activate ~/tts-env
```

---

## Installed package locations

| Environment | Path |
|-------------|------|
| WhisperX venv | `~/whisperx-env/` |
| Coqui TTS venv | `~/tts-env/` |
| MFA conda env | `/opt/homebrew/Caskroom/miniforge/base/envs/mfa/` |
| MFA models/dicts | `~/Documents/MFA/` |

---

## Common Workflow: Transcribe → Align → Synthesize

```bash
# Step 1: Transcribe audio with WhisperX (get word timestamps)
source ~/whisperx-env/bin/activate
whisper input.mp3 --model large-v3 --output_format txt --output_dir ./transcripts
deactivate

# Step 2: Force-align transcript to audio with MFA
conda activate mfa
mfa align ./corpus english_us_arpa english_us_arpa ./aligned_textgrids
conda deactivate

# Step 3: Clone voice / synthesize with Coqui TTS
source ~/tts-env/bin/activate
tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --text "New synthesized speech." \
    --speaker_wav input.mp3 \
    --language en \
    --out_path synthesized.wav
deactivate
```
