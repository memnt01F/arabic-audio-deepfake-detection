"""
make_real_spectrogram.py
------------------------
Generates a REAL mel-spectrogram from a local ArFake wav file, using the SAME
preprocessing as the CNN (16 kHz, 128 mel bands, 256 frames, dB-safe padding),
and also makes a duration histogram. Reads the local CSVs — no `datasets`
library and no re-download needed.

Run from inside the ArFake folder (where train.csv and the audio folders are):

  python make_real_spectrogram.py --out figures
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display

SR = 16000
N_MELS = 128
MAX_FRAMES = 256


def load_fixed_melspec(path):
    """Same pipeline as baseline_cnn.py: load, mel, dB, dB-safe pad/crop."""
    y, _ = librosa.load(path, sr=SR)
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    S = librosa.power_to_db(S, ref=np.max)
    if S.shape[1] < MAX_FRAMES:
        pad_val = float(S.min())          # dB-safe: pad with silence, not 0 dB
        S = np.pad(S, ((0, 0), (0, MAX_FRAMES - S.shape[1])),
                   mode="constant", constant_values=pad_val)
    else:
        S = S[:, :MAX_FRAMES]
    return S, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--csv", default="train.csv")
    ap.add_argument("--n-duration", type=int, default=1500,
                    help="how many clips to sample for the duration histogram")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    print("rows:", len(df), "columns:", list(df.columns))

    # pick one bona fide (real) clip that exists on disk, for the spectrogram
    real_rows = df[df.Label == 0]
    chosen = None
    for p in real_rows.Path.tolist():
        if os.path.exists(p):
            chosen = p
            break
    if chosen is None:  # fall back to any existing file
        for p in df.Path.tolist():
            if os.path.exists(p):
                chosen = p
                break
    if chosen is None:
        raise SystemExit("No audio files found on disk next to this script. "
                         "Run from inside the ArFake folder.")

    # parse metadata from the path for a labelled caption
    parts = chosen.replace("\\", "/").split("/")
    source, dialect = parts[0], parts[1]
    is_real = "bona" in chosen.lower()
    label_txt = "bona fide (real)" if is_real else "spoof (fake)"
    print(f"Spectrogram from: {chosen}")
    print(f"  source={source}  dialect={dialect}  label={label_txt}")

    S, y = load_fixed_melspec(chosen)
    dur = len(y) / SR

    plt.figure(figsize=(6.2, 3.8))
    librosa.display.specshow(S, sr=SR, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel-spectrogram — {label_txt}, {dialect}\n"
              f"source: {source} ({dur:.1f}s clip, model input 256 frames)")
    plt.tight_layout()
    sppath = os.path.join(args.out, "sample_melspectrogram.png")
    plt.savefig(sppath, dpi=150)
    plt.close()
    print("saved", sppath)

    # duration histogram over a random sample of existing files
    paths = df.Path.tolist()
    random.seed(42)
    random.shuffle(paths)
    durations = []
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            yy, _ = librosa.load(p, sr=SR)
            durations.append(len(yy) / SR)
        except Exception:
            pass
        if len(durations) >= args.n_duration:
            break

    if durations:
        plt.figure(figsize=(6.2, 3.8))
        plt.hist(durations, bins=40, color="#1F2A44", edgecolor="white")
        plt.title(f"Clip duration distribution (n={len(durations)})")
        plt.xlabel("Duration (seconds)"); plt.ylabel("Count")
        plt.tight_layout()
        dpath = os.path.join(args.out, "duration_hist.png")
        plt.savefig(dpath, dpi=150)
        plt.close()
        print(f"saved {dpath}: mean={np.mean(durations):.2f}s "
              f"median={np.median(durations):.2f}s")

    print("done ->", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
