# Audio ML Toolkit
A self-contained workbench for transcribing audio, aligning transcripts to
waveforms, and synthesising new speech in the source speaker's voice — all
running locally on Apple Silicon (MPS/CPU).
Pipeline in one sentence:
> **audio.wav → WhisperX transcript → clip corpus → MFA TextGrids → XTTS-v2 voice clone → synthesized.wav**
Every stage lives in its own Python environment; a small set of scripts in this
directory glues them together. This toolkit was bootstrapped on a Kobe Bryant
game-commentary clip (see `dataset/metadata.csv`) and is easy to retarget to
any speaker with ≥ a few minutes of clean reference audio.
## Repository layout
```
audio-ml-toolkit/
├── README.md                  this file
├── README.pre-20260422.md     previous README, kept for diffing
├── activate.sh                shell helpers: use-whisper / use-mfa / use-tts / ml-status / ml-help
├── audio.wav                  original reference/source audio (stereo, 2.3 MB)
├── build_dataset.py           slice audio.wav into a TTS+MFA-ready corpus using the WhisperX transcript
├── synthesize.py              render a text file in the speaker's voice via Coqui XTTS-v2
├── synthesized.wav            output of synthesize.py (first pass, 4.9 MB)
├── synthesized2.wav           output of synthesize.py (second pass, 4.4 MB)
├── dataset/                   built by build_dataset.py
│   ├── metadata.csv           Coqui-style `wav|text` manifest
│   ├── wavs/                  20 × 16 kHz mono PCM clips (clip_0001.wav…)
│   ├── txts/                  matching plain-text transcripts
│   ├── corpus/kobe/           MFA layout: clip.wav + clip.lab pairs
│   └── textgrids/             MFA alignment output + alignment_analysis.csv
│       └── kobe/
├── whisperx_out/              WhisperX run on audio.wav    (json / tsv / srt / vtt / txt)
├── whisperx_synth/            WhisperX run on synthesized.wav  (sanity-check transcripts)
└── whisperx_synth2/           WhisperX run on synthesized2.wav
```
Total on-disk footprint: ~17 MB.
## Environment map
This toolkit deliberately uses three isolated Python environments because the
three toolchains pin incompatible versions of PyTorch, numpy, and numba.
| Tool                        | Env type | Path                                                   | Python |
|-----------------------------|----------|--------------------------------------------------------|--------|
| WhisperX + OpenAI Whisper   | venv     | `~/whisperx-env`                                       | 3.13   |
| Montreal Forced Aligner     | conda    | `mfa` (under `/opt/homebrew/Caskroom/miniforge/base`)  | 3.11   |
| Coqui TTS (XTTS-v2, YourTTS)| venv     | `~/tts-env`                                            | 3.10   |
> **Hard rule:** only one env active at a time. `deactivate` / `conda deactivate`
> between stages, or open a fresh shell. `use-whisper` / `use-mfa` / `use-tts`
> handle the switch automatically.
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
### Package versions present in the venv
- `whisperx 3.8.5` · `openai-whisper 20250625`
- `torch` · `torchaudio` · `faster-whisper` · `pyannote.audio`
- `transformers` · `soundfile` · `numpy` · `tqdm`
### CLI — OpenAI Whisper (simple, single model call)
```bash
whisper audio.wav --model medium                                # auto-detect language
whisper audio.wav --model large-v3 --language en --output_format srt
# Models: tiny, base, small, medium, large, large-v2, large-v3
```
### CLI — WhisperX (adds word-level alignment; used to build this repo)
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
import whisperx
device = "mps"                                              # or "cpu"
audio = whisperx.load_audio("audio.wav")
model = whisperx.load_model("large-v2", device, compute_type="float32")
result = model.transcribe(audio, batch_size=8)
model_a, meta = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, meta, audio, device)
# Optional speaker diarization (needs HF token with access to pyannote/speaker-diarization-3.1):
diarize = whisperx.DiarizationPipeline(use_auth_token="HF_TOKEN", device=device)
result = whisperx.assign_word_speakers(diarize(audio), result)
print(result["segments"])
```
## Stage 2 · Build training corpus (`build_dataset.py`)
Uses the WhisperX JSON + the source WAV to emit one clip per segment in
three parallel layouts so the same data feeds both MFA and Coqui TTS.
```bash
use-whisper                 # needs ffmpeg + python, venv is fine
python build_dataset.py
```
Defaults (edit at the top of the script):
- `MIN_DUR = 1.0 s`, `MAX_DUR = 30.0 s` — drops too-short / too-long segments
- `PAD = 0.05 s` — padding added to each boundary before slicing
- `SPEAKER = "kobe"`
- Resamples every clip to **16 kHz mono PCM** via `ffmpeg -ac 1 -ar 16000 -acodec pcm_s16le`.
Outputs into `dataset/`:
```
dataset/
├── wavs/        clip_0001.wav … clip_00NN.wav      (TTS training audio)
├── txts/        clip_0001.txt … clip_00NN.txt      (matching text)
├── corpus/
│   └── kobe/    clip_0001.wav + clip_0001.lab …    (MFA layout)
└── metadata.csv                                    (Coqui "wav|text" manifest)
```
The current build yields **20 clips** from `audio.wav`. First three lines of
`metadata.csv`:
```
wavs/clip_0001.wav|I remember getting new stuff, getting going, and then finally, the game kind of working myself into a lather.
wavs/clip_0002.wav|My body felt good.
wavs/clip_0003.wav|Here he comes!
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
The current run stashes its outputs under `dataset/textgrids/kobe/` together
with `alignment_analysis.csv`, which summarises per-clip alignment quality
(useful for throwing out the worst clips before training).
### Maintenance commands
```bash
mfa model list acoustic
mfa model list dictionary
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
mfa train ./dataset/corpus english_us_arpa ./dataset/kobe_acoustic.zip \
          --output_directory ./dataset/textgrids
```
### Installed assets
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
Already-generated example outputs in this directory:
- `synthesized.wav`  (first pass, default params)
- `synthesized2.wav` (second pass, different seed/params)
- `whisperx_synth/` and `whisperx_synth2/` contain WhisperX transcripts of
  those outputs so you can diff against the input script and measure drift.
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
| Flag                      | Default | Notes                                                      |
|---------------------------|---------|------------------------------------------------------------|
| `--text`                  | *req.*  | Path to a UTF-8 text file (the script to speak)            |
| `--speaker-wav`           | *req.*  | Reference clip (6–60 s of clean speech recommended)        |
| `--out`                   | *req.*  | Output `.wav` path                                         |
| `--language`              | `en`    | ISO 639-1 code                                             |
| `--temperature`           | `0.55`  | Lower = more stable, higher = more expressive              |
| `--repetition-penalty`    | `5.0`   | Higher reduces stutters                                    |
| `--length-penalty`        | `1.0`   |                                                            |
| `--top-k` / `--top-p`     | `50 / 0.85` | Standard sampling knobs                                |
| `--speed`                 | `1.0`   | 0.8–1.2 usable range                                       |
| `--single-shot`           | off     | Send the whole text in one call (XTTS internal splitting)  |
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
Generate an HF token, accept terms on `pyannote/speaker-diarization-3.1` and
`pyannote/segmentation-3.0`, export `HF_TOKEN=...`.
**MFA validate fails with OOV words**
Add pronunciations to a user dictionary or regenerate with:
```bash
mfa g2p ./dataset/corpus english_us_arpa ./dataset/g2p.dict
mfa align ./dataset/corpus ./dataset/g2p.dict english_us_arpa ./dataset/textgrids
```
**`ffmpeg: command not found` inside build_dataset.py**
Ensure `ffmpeg` is on `PATH` in whichever env runs the script (the WhisperX
venv is fine; `brew install ffmpeg` if missing system-wide).
**Switching envs warns about "Deactivating conda/venv first…"**
That's the `use-*` helpers being polite — they refuse to stack environments
because it produces undefined `PYTHONPATH` behaviour.
## Outputs shipped with this repo
Committed for reference / regression testing:
- `whisperx_out/audio.{json,tsv,srt,vtt,txt}`
  – transcripts of the source audio.
- `dataset/` – a fully built 20-clip corpus keyed to `audio.wav`.
- `dataset/textgrids/kobe/` – MFA alignments for those clips plus the
  `alignment_analysis.csv` quality report.
- `synthesized.wav` / `synthesized2.wav` – XTTS-v2 clones of the source speaker.
- `whisperx_synth*/` – WhisperX re-transcriptions of the two synthesised files,
  used to quantify how faithful the clone is to the input script.
## Archives
Every change to this toolkit is preceded by a timestamped archive under
`~/Projects/archives/audio-ml-toolkit-YYYYMMDD-HHMM/` (directory clone) and
a matching `.tgz` with a SHA-256 digest. To restore an archived copy:
```bash
tar -xzf ~/Projects/archives/audio-ml-toolkit-YYYYMMDD-HHMM.tgz -C /tmp
diff -r /tmp/audio-ml-toolkit ~/audio-ml-toolkit
```
## License / attribution
Scripts (`build_dataset.py`, `synthesize.py`, `activate.sh`): MIT.
Model weights (WhisperX, pyannote, Coqui XTTS-v2, MFA acoustic/dictionary)
retain their upstream licenses — check each project before redistributing.
