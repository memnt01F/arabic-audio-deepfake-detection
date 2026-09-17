# Arabic Audio Deepfake Detection

**ICS 471 — Deep Learning · Course Project (proposal-supporting code)**
King Fahd University of Petroleum & Minerals (KFUPM)

**Team:** [Name 1] · [Name 2] · [Name 3]

Detects whether an Arabic speech clip is a **real human voice (bona fide)** or
**synthetic** speech from a text-to-speech / voice-cloning system. Supervised
binary classification on the **ArFake** dataset.

- **Input:** an Arabic audio clip → mel-spectrogram
- **Output:** real vs. fake (probability of spoof)
- **Metric:** Equal Error Rate (EER); F1 as a secondary, readable number

---

## Dataset

[**ArFake**](https://huggingface.co/datasets/Mohammed01/ArFake) — the first
multi-dialect Arabic spoofed-speech benchmark. 54,413 utterances, 8 dialects,
bona fide + 3 in-domain TTS generators (FishSpeech, XTTS-v2, ArTST; VITS is held
out by the authors for unseen-generator tests). Official split: **31,302 train /
23,111 test**. License: Apache-2.0 but **gated** — you must accept the terms on
the HF page and be logged in.

**Schema** — two columns only:
- `Path` (string): e.g. `fish-speech-new/Morocco/bonafied/original_19037_3.wav`
- `Label` (int64): `0` = bona fide (real); `1` = spoof (fake)

Generator and dialect are **not** separate columns — they are encoded in the
path (`<source>/<dialect>/<bonafied|spoofed>/<file>.wav`), and our scripts parse
them out.

### What we found by inspecting the label files
- **Class imbalance:** ~**75% spoof / 25% bona fide**
  (train 21,772 / 9,530; test 19,026 / 4,085). We handle this with EER as the
  metric and class weighting / balanced sampling in training.
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
and `gen_label_stacked.png`.

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

## Planned split & method

- Keep ArFake's **official test split untouched**; carve a 10% (seed-fixed)
  validation set from train, speaker-disjoint so no speaker leaks across
  train/val.
- Pipeline: `audio → resample 16 kHz → mel-spectrogram → CNN → real/fake`.
- Baseline is intentionally compact; stronger front ends (RawNet2, wav2vec2 /
  WavLM) are planned as comparisons.

## Data & license note

This repo contains **code only** — no dataset files. ArFake is Apache-2.0 and
gated; obtain it from the Hugging Face page above under its terms.

