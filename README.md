# Arabic Audio Deepfake Detection

**ICS 471 â€” Deep Learning course project proposal**  
King Fahd University of Petroleum & Minerals (KFUPM)

**Team:** Reem Alghuzawi and Maiss Khalaf

## Project idea

The project will study whether an Arabic speech clip is a real human recording
or synthetic speech produced by a text-to-speech or voice-cloning system.

This is a supervised binary-classification problem:

- **Input:** an Arabic speech clip.
- **Output:** `0` for bona fide (real) or `1` for spoof (fake).
- **Future model:** a PyTorch model will learn from audio features such as
  mel-spectrograms.

This repository currently supports the **proposal and dataset-analysis phase**.
Model training will be implemented in the next phase.

## Dataset

We use the [ArFake dataset](https://huggingface.co/datasets/Mohammed01/ArFake),
a multi-dialect Arabic spoofed-speech dataset.

- 54,413 speech utterances in total.
- 8 Arabic dialects.
- Official split: 31,302 training rows and 23,111 test rows.
- `Label = 0`: bona fide, or real human speech.
- `Label = 1`: spoof, or synthetic speech.

The CSV files contain two columns:

- `Path`: the location and metadata path of an audio clip.
- `Label`: the real/fake label.

The path has the form:

```text
source/dialect/class/file.wav
```

The first part is treated as a **source group**, not automatically as a
generator, because `fish-speech-new` contains both real and spoofed clips.

The data is imbalanced. Spoof clips make up approximately 69.6% of the
training split and 82.3% of the official test split.

## Current proposal work

The current code performs dataset inspection without downloading the large
audio collection. It:

1. Reads `train.csv` and `test.csv`.
2. Prints dataset sizes, label counts, source groups, and dialect counts.
3. Creates four distribution figures in the `figures/` folder.

The repository also contains `sample_melspectrogram.png`, a representative
visualization of an audio sample. It is included as a figure for the proposal;
the current CSV-analysis script does not need the audio files to run.

## How to run the analysis

Install the two packages needed for the current phase:

```bash
pip install pandas matplotlib
```

Make sure `train.csv`, `test.csv`, and `analyze_csv.py` are in the same folder.
Then run:

```bash
python analyze_csv.py
```

The script creates these files inside `figures/`:

- `class_balance.png`
- `by_source.png`
- `by_dialect.png`
- `source_label_stacked.png`

## Planned next phase

In the next phase, we plan to:

- download and preprocess the audio;
- convert audio into mel-spectrograms;
- implement and train a CNN using PyTorch;
- keep the official test split untouched;
- evaluate the model using Equal Error Rate (EER) as the main metric;
- later investigate group-disjoint splitting, class imbalance handling, F1,
  and confusion matrices.

These training steps are planned and are not claimed to be completed in this
proposal repository.

## Repository contents

```text
README.md
analyze_csv.py
train.csv
test.csv
figures/
    class_balance.png
    by_source.png
    by_dialect.png
    source_label_stacked.png
    sample_melspectrogram.png
requirements.txt
LICENSE
```

## Data access and licensing

The ArFake dataset is gated and must be accessed through its terms on
Hugging Face. The underlying real speech comes from the Casablanca corpus,
which is listed as **CC-BY-NC-ND-4.0**: attribution, non-commercial use, and
no derivatives.

Our project code is released under the **Apache-2.0** license. This repository
does not contain the large audio files. We use the dataset only for this
academic, non-commercial project, provide the required citations, and do not
redistribute the audio or derivative audio.
