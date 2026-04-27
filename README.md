# Audio ML Toolkit
A self-contained workbench for transcribing audio, aligning transcripts to
waveforms, and synthesising new speech in the source speaker's voice — all
running locally on Apple Silicon (MPS/CPU).

Pipeline in one sentence:

> **`audio.wav` → WhisperX transcript → clip corpus → MFA TextGrids → XTTS-v2 voice clone → `synthesized.wav`**

Every stage lives in its own Python environment; a small set of scripts in
this directory glues them together. The toolkit is easy to retarget to any
speaker with a few minutes of clean reference audio.

## Repository layout
Only source files are tracked; audio, models, environments, and pipeline
outputs are produced locally and listed in `.gitignore`.

```
audio-ml-toolkit/
├── README.md            this file
├── .gitignore           excludes venvs, audio, and pipeline outputs
├── activate.sh          shell helpers: use-whisper / use-mfa / use-tts / ml-status / ml-help
├── build_dataset.py     slice audio.wav into a TTS+MFA-ready corpus from a WhisperX transcript
└── synthesize.py        render a text file in the speaker's voice via Coqui XTTS-v2
```

Locally produced (gitignored) artefacts:

```
audio.wav                source/reference audio
synthesized*.wav         outputs of synthesize.py
.venv/                   demucs / utility venv (this repo)
whisperx-env/            WhisperX venv (created in $HOME)
tts-env/                 Coqui TTS venv  (created in $HOME)
dataset/                 built by build_dataset.py
├── metadata.csv         Coqui-style `wav|text` manifest
├── wavs/                16 kHz mono PCM clips (clip_0001.wav…)
├── txts/                matching plain-text transcripts
├── corpus/<speaker>/    MFA layout: clip.wav + clip.lab pairs
└── textgrids/           MFA alignment output + alignment_analysis.csv
separated/               demucs stems (bass / drums / other / vocals)
whisperx_out/            WhisperX run on audio.wav     (json / tsv / srt / vtt / txt)
whisperx_synth*/         WhisperX runs on synthesised outputs (sanity-check transcripts)
```

## Environment map
This toolkit deliberately uses three isolated Python environments because
the three toolchains pin incompatible versions of PyTorch, NumPy, and Numba.

| Tool                         | Env type | Path                                                  | Python |
|------------------------------|----------|-------------------------------------------------------|--------|
| WhisperX + OpenAI Whisper    | venv     | `~/whisperx-env`                                      | 3.13   |
| Montreal Forced Aligner      | conda    | `mfa` (under your local miniforge/conda install)      | 3.11   |
| Coqui TTS (XTTS-v2, YourTTS) | venv     | `~/tts-env`                                           | 3.10   |

> **Hard rule:** only one env active at a time. `deactivate` /
> `conda deactivate` between stages, or open a fresh shell.
> `use-whisper` / `use-mfa` / `use-tts` (from `activate.sh`) handle the
> switch automatically.

### Load the shell helpers
```bash
source ~/audio-ml-toolkit/activate.sh
ml-help           # prints cheatsheet
ml-status         # shows the currently active env
use-whisper       # activate WhisperX env
use-mfa           # activate MFA conda env
use-tts           # activate Coqui TTS env
```

## Stage 1 · Transcribe (WhisperX)
### Activate
```bash
use-whisper
# or: source ~/whisperx-env/bin/activate
```

### Packages used by this stage
- `whisperx` · `openai-whisper`
- `torch` · `torchaudio` · `faster-whisper` · `pyannote.audio`
- `transformers` · `soundfile` · `numpy` · `tqdm`

### CLI — OpenAI Whisper (simple, single model call)
```bash
whisper audio.wav --model medium                                # auto-detect language
whisper audio.wav --model large-v3 --language en --output_format srt
# Models: tiny, base, small, medium, large, large-v2, large-v3
```

### CLI — WhisperX (adds word-level alignment; what build_dataset.py expects)
```bash
whisperx audio.wav \
  --model large-v2 \
  --output_format all \
  --output_dir whisperx_out
```

This writes `whisperx_out/audio.{json,tsv,srt,vtt,txt}`. The `.json` is the
input for `build_dataset.py`; it contains per-segment `start`, `end`, and
`text` plus the word-level alignment table.

### Python API — word-level timestamps + optional diarization
```python
import os, whisperx
device = "mps"                                              # or "cpu"
audio = whisperx.load_audio("audio.wav")
model = whisperx.load_model("large-v2", device, compute_type="float32")
result = model.transcribe(audio, batch_size=8)
model_a, meta = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, meta, audio, device)
# Optional speaker diarization (needs HF token with access to pyannote/speaker-diarization-3.1):
diarize = whisperx.DiarizationPipeline(use_auth_token=os.environ["HF_TOKEN"], device=device)
result = whisperx.assign_word_speakers(diarize(audio), result)
print(result["segments"])
```

## Stage 2 · Build training corpus (`build_dataset.py`)
Uses the WhisperX JSON + the source WAV to emit one clip per segment in
three parallel layouts so the same data feeds both MFA and Coqui TTS.

```bash
use-whisper                 # needs ffmpeg + python; venv is fine
python build_dataset.py
```

Defaults (edit at the top of the script):

- `MIN_DUR = 1.0 s`, `MAX_DUR = 30.0 s` — drops too-short / too-long segments
- `PAD = 0.05 s` — padding added to each boundary before slicing
- `SPEAKER = "kobe"` — change to any speaker label
- Resamples every clip to **16 kHz mono PCM** via `ffmpeg -ac 1 -ar 16000 -acodec pcm_s16le`.

Outputs into `dataset/`:

```
dataset/
├── wavs/        clip_0001.wav … clip_00NN.wav      (TTS training audio)
├── txts/        clip_0001.txt … clip_00NN.txt      (matching text)
├── corpus/
│   └── <speaker>/  clip_0001.wav + clip_0001.lab …  (MFA layout)
└── metadata.csv                                    (Coqui "wav|text" manifest)
```

Re-running the script wipes `dataset/` and rebuilds from scratch.

## Stage 3 · Force-align (MFA)
MFA produces phoneme- and word-level `.TextGrid` files from the corpus/lab
pairs emitted above.

### Activate and run
```bash
use-mfa
mfa validate ./dataset/corpus english_us_arpa english_us_arpa       # dry-run sanity check
mfa align    ./dataset/corpus english_us_arpa english_us_arpa \
             ./dataset/textgrids
```

### Maintenance commands
```bash
mfa model list acoustic
mfa model list dictionary
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
mfa train ./dataset/corpus english_us_arpa ./dataset/<speaker>_acoustic.zip \
          --output_directory ./dataset/textgrids
```

### Required assets
- Acoustic model: `english_us_arpa`
- Dictionary: `english_us_arpa`
- Global cache: `~/Documents/MFA/`

## Stage 4 · Synthesize (`synthesize.py`, Coqui XTTS-v2)
Clones the source speaker and renders any text script in their voice.

### Activate and run
```bash
use-tts
python synthesize.py \
  --text        script.txt \
  --speaker-wav audio.wav \
  --out         synthesized.wav \
  --language    en
```

### What the script does
- Normalises the script (`unicodedata.NFKC`, collapses whitespace).
- Splits into ≤ 220-char chunks on sentence boundaries, then on commas for
  anything still too long. Keeps prosody stable across long inputs.
- Monkey-patches `torch.load(weights_only=False)` **before** importing
  `TTS.api` — PyTorch 2.6+ made `weights_only=True` the default, which
  rejects Coqui's pickled XTTS checkpoints.
- Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` and `COQUI_TOS_AGREED=1`.
- Runs XTTS-v2 on **CPU by default** (MPS conv ops can be slow/flaky);
  change `device = "cpu"` in `main()` to try MPS.
- Concatenates chunks with 150 ms silence between them and peak-normalises
  to 0.97 before writing int16 PCM at the model's native 24 kHz.

### CLI flags
| Flag                      | Default     | Notes                                                    |
|---------------------------|-------------|----------------------------------------------------------|
| `--text`                  | *required*  | Path to a UTF-8 text file (the script to speak)          |
| `--speaker-wav`           | *required*  | Reference clip (6–60 s of clean speech recommended)      |
| `--out`                   | *required*  | Output `.wav` path                                       |
| `--language`              | `en`        | ISO 639-1 code                                           |
| `--temperature`           | `0.55`      | Lower = more stable, higher = more expressive            |
| `--repetition-penalty`    | `5.0`       | Higher reduces stutters                                  |
| `--length-penalty`        | `1.0`       |                                                          |
| `--top-k` / `--top-p`     | `50 / 0.85` | Standard sampling knobs                                  |
| `--speed`                 | `1.0`       | 0.8–1.2 usable range                                     |
| `--single-shot`           | off         | Send the whole text in one call (XTTS internal splitting)|

### Alternative: bare `tts` CLI (no chunking)
```bash
# YourTTS (multilingual, lighter)
tts --model_name tts_models/multilingual/multi-dataset/your_tts \
    --text "Hello, this is my cloned voice." \
    --speaker_wav audio.wav --language_idx en --out_path out.wav

# XTTS-v2 (best quality; downloads ~1.8 GB on first run)
tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --text "Hello, this is my cloned voice." \
    --speaker_wav audio.wav --language en --out_path out.wav
```

## Optional · Source separation (Demucs)
A local `.venv/` in this repo can host [Demucs](https://github.com/facebookresearch/demucs)
for splitting `audio.wav` into stems before transcription/cloning.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip demucs torchcodec
python -m demucs -o separated audio.wav
# → separated/htdemucs/audio/{bass,drums,other,vocals}.wav
deactivate
```

## End-to-end: re-run the whole pipeline
Starting from a fresh `audio.wav`:

```bash
# 0. Convert/ensure 16 kHz mono PCM source (if your input isn't already)
ffmpeg -y -i source.mp4 -ac 1 -ar 16000 -acodec pcm_s16le audio.wav

# 1. Transcribe with WhisperX (word timestamps, all formats)
use-whisper
whisperx audio.wav --model large-v2 --output_format all --output_dir whisperx_out
deactivate

# 2. Slice into TTS+MFA corpus
use-whisper
python build_dataset.py
deactivate

# 3. Force-align with MFA
use-mfa
mfa align ./dataset/corpus english_us_arpa english_us_arpa ./dataset/textgrids
conda deactivate

# 4. Synthesize new text in the cloned voice
use-tts
python synthesize.py \
  --text my_script.txt --speaker-wav audio.wav \
  --out synthesized.wav --language en
deactivate

# 5. (Optional) Transcribe the synthesis to QA drift
use-whisper
whisperx synthesized.wav --model large-v2 --output_format all --output_dir whisperx_synth
deactivate
```

## Troubleshooting
**`torch.load` refuses Coqui checkpoints (`UnpicklingError: Weights only load failed`)**
Handled in `synthesize.py`. If you hit it elsewhere, patch before import:
```python
import torch
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
```

**MPS kernel failures with Coqui TTS**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 tts ...
# or just leave synthesize.py on its CPU default; XTTS is CPU-fine.
```

**WhisperX diarization 401**
Generate a Hugging Face token, accept terms on
`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`, then
export it before running:
```bash
export HF_TOKEN=...
```

**MFA validate fails with OOV words**
Add pronunciations to a user dictionary or regenerate with:
```bash
mfa g2p ./dataset/corpus english_us_arpa ./dataset/g2p.dict
mfa align ./dataset/corpus ./dataset/g2p.dict english_us_arpa ./dataset/textgrids
```

**`ffmpeg: command not found` inside `build_dataset.py`**
Ensure `ffmpeg` is on `PATH` in whichever env runs the script (the WhisperX
venv is fine; `brew install ffmpeg` if missing system-wide).

**Switching envs warns about "Deactivating conda/venv first…"**
That's the `use-*` helpers being polite — they refuse to stack environments
because it produces undefined `PYTHONPATH` behaviour.

## License
Scripts (`build_dataset.py`, `synthesize.py`, `activate.sh`): MIT.

Model weights (WhisperX, pyannote, Coqui XTTS-v2, MFA acoustic/dictionary)
retain their upstream licenses — check each project before redistributing.
