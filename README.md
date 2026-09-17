# Arabic Audio Deepfake Detection

**ICS 471 — Deep Learning · Course Project (proposal-supporting code)**
King Fahd University of Petroleum & Minerals (KFUPM)

**Team Members:** Reem Alghuzawi, Maiss Khalaf

Detects whether an Arabic speech clip is a **real human voice (bona fide)** or
**synthetic** speech from a text-to-speech / voice-cloning system. Supervised
binary classification on the **ArFake** dataset.

- **Input:** an Arabic audio clip → mel-spectrogram
- **Output:** real vs. fake (probability of spoof)
- **Metric:** Equal Error Rate (EER) for model selection; F1 planned as a secondary metric

---

## Dataset

[**ArFake**](https://huggingface.co/datasets/Mohammed01/ArFake) — the first
multi-dialect Arabic spoofed-speech benchmark. 54,413 utterances, 8 dialects,
bona fide + 3 in-domain TTS generators (FishSpeech, XTTS-v2, ArTST; VITS is held
out by the authors for unseen-generator tests). Official split: **31,302 train /
23,111 test**. **Licensing:** our code is Apache-2.0 (see `LICENSE`); the dataset is
**gated** and the real audio is from Casablanca (**CC-BY-NC-ND-4.0**) — see the
License section at the bottom. Accept the terms on the HF page and be logged in to
download.

**Schema** — two columns only:
- `Path` (string): e.g. `fish-speech-new/Morocco/bonafied/original_19037_3.wav`
- `Label` (int64): `0` = bona fide (real); `1` = spoof (fake)

Generator and dialect are **not** separate columns — they are encoded in the
path (`<source>/<dialect>/<bonafied|spoofed>/<file>.wav`), and our scripts parse
them out.

### What we found by inspecting the label files
- **Class imbalance:** the two splits differ — train 21,772 spoof / 9,530 real
  (69.6% spoof), test 19,026 / 4,085 (82.3% spoof), ~75% overall. EER (our metric)
  is robust to this; class weighting / balanced sampling in training is **planned**
  (not yet in the code).
- **All bona fide audio lives under `fish-speech-new`** (which holds the original
  human recordings plus fish-speech spoofs); the `XTTS2` and `ArTST` folders are
  100% spoof. So the top-level folder is a *source group*, not a clean generator
  label.
- **Dialects** are reasonably balanced, from Palestine (5,326) to Morocco (8,360).

---

## Quick start (no audio needed) — reproduce the proposal plots

The label tables (`train.csv`, `test.csv`, a few MB) are enough to reproduce
every distribution plot in the proposal — you do **not** need the 18 GB of audio.

```bash
pip install pandas matplotlib
python analyze_csv.py --train train.csv --test test.csv --out figures
```

This prints the class balance, generator, and dialect breakdowns and writes four
figures to `figures/`: `class_balance.png`, `by_generator.png`, `by_dialect.png`,
and `gen_label_stacked.png`. (The proposal's split-size bar and the mel-spectrogram
are made separately — the spectrogram comes from `inspect_arfake.py` once the audio
is downloaded.)

---

## Full setup (for training on the audio)

```bash
pip install -r requirements.txt
hf auth login                 # after accepting the dataset terms on HF

# download the full dataset (~18.2 GB)
hf download Mohammed01/ArFake --repo-type dataset --local-dir ./ArFake
# if the fast (Xet) transfer 403s mid-download, disable it and re-run (it resumes):
#   set HF_HUB_DISABLE_XET=1        # Windows
#   export HF_HUB_DISABLE_XET=1     # macOS/Linux

# tell the scripts where the wav files are
set ARFAKE_DIR=.\ArFake            # Windows
export ARFAKE_DIR=./ArFake         # macOS/Linux
```

## Usage

```bash
# Inspect samples + full distribution plots (loads audio for durations/spectrogram)
python inspect_arfake.py --out figures

# Train the baseline CNN (audio -> mel-spectrogram -> CNN -> real/fake)
python baseline_cnn.py --epochs 10 --batch-size 32
```

---

## Files

| File | Purpose |
|------|---------|
| `analyze_csv.py`    | **Start here.** Reproduces the proposal plots from `train.csv`/`test.csv` alone — no audio download needed. |
| `inspect_arfake.py` | Loads the full dataset from the Hub, prints the schema, and plots class balance / generator / dialect / a sample mel-spectrogram. |
| `baseline_cnn.py`   | The trained DL core: a small CNN over mel-spectrograms, with EER as the metric. |
| `requirements.txt`  | Dependencies. |

## Status: implemented vs. planned

**Implemented now:**
- Data inspection + all four distribution plots (`analyze_csv.py`)
- `audio → mel-spectrogram → CNN → real/fake` pipeline with dB-safe padding
- EER metric on a validation split; best-checkpoint saving (`baseline_cnn.py`)

**Planned (described in the proposal as future work, not yet in the code):**
- Speaker/source-disjoint split — the CSVs expose only `Path` + `Label` (no
  speaker ID), so the current split is random; we plan a group-disjoint split
  with an explicit disjointness check once group metadata is derived.
- Class weighting / balanced sampling for the ~70–82% spoof imbalance.
- Evaluation on the official **test** split, plus F1 and a confusion matrix
  (EER stays the single model-selection metric).
- Explicit FFT/hop settings, fixed seeds, and robust file-loading.

## Planned split & method

- Keep ArFake's **official test split untouched**; carve a 10% (seed-fixed)
  validation set from train. A **group-disjoint** version (by speaker / original
  utterance) is planned so no speaker or source recording leaks across
  train/val — see status note above.
- Pipeline: `audio → resample 16 kHz → mel-spectrogram (128 mel, 256 frames,
  dB-safe padding) → CNN → real/fake`.
- Baseline is intentionally compact; stronger front ends (RawNet2, wav2vec2 /
  WavLM) are planned as comparisons (would add `transformers` to requirements).

## License

Our project **code** is released under the Apache-2.0 license; see the `LICENSE`
file. The **ArFake dataset** is gated and must be accessed through its terms on
Hugging Face. The underlying real speech comes from the **Casablanca** corpus,
listed as **CC-BY-NC-ND-4.0** (attribution, non-commercial, no-derivatives). We
use the data only for this academic, non-commercial project, provide the required
citations, and **do not redistribute the audio or any derivative audio**. This
repository contains code, figures, and documentation only — no audio files.
