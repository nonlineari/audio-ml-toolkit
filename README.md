# Audio ML Toolkit
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-black)]()
[![Python: 3.10/3.11/3.13](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.13-blue)]()
A self-contained workbench for transcribing audio, aligning transcripts to
waveforms, and synthesising new speech in the source speaker's voice — all
running locally on Apple Silicon (MPS/CPU).

Pipeline in one sentence:

> **`audio.wav` → WhisperX transcript → clip corpus → MFA TextGrids → XTTS-v2 voice clone → `synthesized.wav`**

Every stage lives in its own Python environment; a small set of scripts in
this directory glues them together. The toolkit is easy to retarget to any
speaker with a few minutes of clean reference audio.

## Quick start
For users who already have the three Python environments configured (see
[Prerequisites](#prerequisites) and [Environment map](#environment-map)).
Given any source audio file you want to clone:

```bash
# 0. Clone and enter the repo
git clone https://github.com/nonlineari/audio-ml-toolkit.git
cd audio-ml-toolkit

# 1. Make sure your reference audio is 16 kHz mono PCM (ffmpeg one-liner)
ffmpeg -y -i source.mp3 -ac 1 -ar 16000 -acodec pcm_s16le audio.wav

# 2. Load the helper functions for the rest of the session
source ./activate.sh                           # use-whisper / use-mfa / use-tts / ml-status

# 3. Transcribe → corpus → align → synthesize
use-whisper && whisperx audio.wav --model large-v2 --output_format all --output_dir whisperx_out
python build_dataset.py                        # → dataset/{wavs,txts,corpus,metadata.csv}
use-mfa     && mfa align ./dataset/corpus english_us_arpa english_us_arpa ./dataset/textgrids
use-tts     && python synthesize.py \
               --text my_script.txt --speaker-wav audio.wav \
               --out synthesized.wav --language en
```

Nothing in the repo writes outside the project directory; everything lands
in `dataset/`, `whisperx_out/`, or `synthesized*.wav` (all gitignored).

## Prerequisites
Mandatory:
- **macOS on Apple Silicon** (Intel will work for inference but is slower).
- **`ffmpeg`** on `PATH`: `brew install ffmpeg`
- **`python3`** ≥ 3.10
- **conda** (Miniforge or Anaconda) for the MFA env: `brew install --cask miniforge`

Optional:
- **`gh`** GitHub CLI if you plan to fork/PR: `brew install gh`
- **HuggingFace token** (`HF_TOKEN`) only if you want WhisperX speaker
  diarization — accept the gated terms on `pyannote/speaker-diarization-3.1`
  and `pyannote/segmentation-3.0`.

Disk budget for the three environments + cached weights: **~12 GB**
(WhisperX ≈ 4 GB, MFA + models ≈ 1 GB, Coqui TTS + XTTS-v2 ≈ 4 GB,
build caches ≈ 3 GB).

## Repository layout
Only source files are tracked; audio, models, environments, and pipeline
outputs are produced locally and listed in `.gitignore`.

```
audio-ml-toolkit/
├── README.md            this file
├── LICENSE              MIT (scripts only — model weights keep upstream licenses)
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

## Detailed usage guide
This section walks through every flag you'd realistically tweak. The four
stages are independent: you can stop after WhisperX (transcript only), stop
after MFA (transcript + alignment), or run end-to-end.

### Inputs you provide
| File           | Format                                | Used by              |
|----------------|---------------------------------------|----------------------|
| `audio.wav`    | 16 kHz mono PCM (use the ffmpeg step) | WhisperX, build_dataset, MFA, synthesize (as `--speaker-wav`) |
| `script.txt`   | UTF-8 plain text                      | `synthesize.py --text` |

### Tunable knobs by stage
**Stage 1 — WhisperX**
- `--model {tiny,base,small,medium,large,large-v2,large-v3}` — accuracy ↔ speed.
  `large-v2` is the sweet spot on Apple Silicon. `large-v3` is more accurate
  but slower and uses more RAM.
- `--language en` — skip auto-detection if you already know the language.
- `--diarize --hf_token $HF_TOKEN` — optional speaker labels (needs gated access).

**Stage 2 — `build_dataset.py`** (edit constants at the top of the file)
- `MIN_DUR = 1.0`, `MAX_DUR = 30.0` — filter outliers; raise `MAX_DUR` for
  audiobooks, lower for clean TTS training data.
- `PAD = 0.05` — boundary padding; raise to 0.10 if MFA reports clipped
  starts/ends in `alignment_analysis.csv`.
- `SPEAKER = "kobe"` — controls `dataset/corpus/<speaker>/`; rename for new speakers.

**Stage 3 — MFA**
- Acoustic / dictionary models: `english_us_arpa` ships by default. List
  alternatives with `mfa model list acoustic` and download with
  `mfa model download acoustic <name>`.
- `mfa validate <corpus> <dict> <acoustic>` — always run this once before
  the first `align` to catch OOV words and malformed `.lab` files.
- `mfa align ... --beam 100 --retry_beam 400` — increase if some clips
  refuse to align (default beam can be tight on noisy or fast speech).

**Stage 4 — `synthesize.py`** (CLI flags)
| Flag                      | Default     | When to change                                              |
|---------------------------|-------------|-------------------------------------------------------------|
| `--temperature`           | `0.55`      | 0.4–0.6 for stable narration; up to 0.85 for expressive vocal |
| `--repetition-penalty`    | `5.0`       | Raise to 6–8 if the output stutters or repeats syllables    |
| `--top-k` / `--top-p`     | `50 / 0.85` | Lower (e.g. 30 / 0.7) makes the voice more deterministic     |
| `--speed`                 | `1.0`       | 0.9 for slower delivery, 1.1 for snappier                    |
| `--single-shot`           | off         | Toggle on for short scripts (≤ 250 chars); avoids chunk seams|

### Worked example: clone yourself in 60 seconds
```bash
# Record ~30 s of clean speech with QuickTime / sox / arecord, then:
ffmpeg -y -i my_recording.m4a -ac 1 -ar 16000 -acodec pcm_s16le audio.wav

echo "Hello world. This is my cloned voice on a laptop." > script.txt

use-tts
python synthesize.py \
  --text script.txt --speaker-wav audio.wav \
  --out  hello_clone.wav --language en
afplay hello_clone.wav             # macOS built-in player
```

### Sanity-checking output quality
Run WhisperX over your synthesised file and `diff` the transcript against
the input script:
```bash
use-whisper
whisperx hello_clone.wav --model large-v2 --output_format txt --output_dir whisperx_synth
diff -u script.txt whisperx_synth/hello_clone.txt
```
A closely-matching transcript means XTTS pronounced the script faithfully;
large diffs usually mean the chunker split a word — try `--single-shot`.

### Reproducibility checklist before you commit your own forks
1. `python3 -m py_compile build_dataset.py synthesize.py` — Python syntax.
2. `zsh -n activate.sh && bash -n activate.sh` — shell syntax.
3. `git status` clean and `git ls-files` shows only the 6 source files.
4. No audio (`*.wav`, `*.mp3`, …) staged: `git check-ignore audio.wav` should print a rule.
5. No tokens in your changes: `git --no-pager grep -nE 'hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}'` returns nothing.

## License
Scripts (`build_dataset.py`, `synthesize.py`, `activate.sh`): **MIT**
— see [`LICENSE`](LICENSE).

Model weights (WhisperX, pyannote, Coqui XTTS-v2, MFA acoustic/dictionary)
retain their upstream licenses — check each project before redistributing.
