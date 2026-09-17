"""
inspect_arfake.py
-----------------
Loads the ArFake dataset (Mohammed01/ArFake on the HuggingFace Hub),
inspects real samples, reports size / labels / class balance, and produces
the distribution plots required by the project proposal:

  1. Class balance (bona fide vs spoof)
  2. Breakdown by TTS generator (parsed from the file path)
  3. Breakdown by dialect (parsed from the file path)
  4. A sample mel-spectrogram (representative data visualization)

REAL SCHEMA (confirmed from the HF dataset viewer):
  * Two columns only:
      - Path  (string)  e.g. "fish-speech-new/Morocco/bonafied/original_19037_3.wav"
      - Label (int64)   0 = bona fide (real); non-zero = spoof (fake)
  * Generator and dialect are NOT separate columns; they are encoded in the
    path: <generator>/<dialect>/<bonafied|spoofed>/<filename>.wav
  * Splits: train (~31.3k) and test (~23.1k).

USAGE
  # 1. Accept the dataset terms + log in:  hf auth login
  # 2. Run:
  #        python inspect_arfake.py --out ../figures
"""

import argparse
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY = "#1F2A44"; RED = "#9E2B25"; GREEN = "#2E6B3E"


def parse_path(path):
    """
    Extract (generator, dialect, is_bonafide) from a path like
    'fish-speech-new/Morocco/bonafied/original_19037_3.wav'.
    Robust to minor variations in segment order / spelling.
    """
    parts = path.replace("\\", "/").split("/")
    generator = parts[0] if len(parts) > 0 else "unknown"
    dialect = parts[1] if len(parts) > 1 else "unknown"
    low = path.lower()
    # 'bonafied' / 'bonafide' / 'bona_fide' vs 'spoof'/'fake'
    is_bonafide = bool(re.search(r"bona", low))
    return generator, dialect, is_bonafide


def bar(counts, title, ylabel, outpath, color=NAVY, rotate=0):
    keys = list(counts.keys())
    vals = [counts[k] for k in keys]
    plt.figure(figsize=(6, 4))
    plt.bar([str(k) for k in keys], vals, color=color)
    plt.title(title); plt.ylabel(ylabel)
    if rotate:
        plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150); plt.close()
    print(f"Saved {outpath}: {dict(counts)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../figures", help="output dir for figures")
    ap.add_argument("--split", default="train", help="which split to inspect")
    ap.add_argument("--max-audio-samples", type=int, default=2000,
                    help="clips to sample for the duration histogram")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset

    print("Loading ArFake (requires accepted terms + login)...")
    ds = load_dataset("Mohammed01/ArFake")
    print("\n=== SPLITS ===")
    for split in ds:
        print(f"  {split}: {len(ds[split])} rows")

    split = args.split if args.split in ds else list(ds.keys())[0]
    cols = ds[split].column_names
    print(f"\n=== COLUMNS ({split}) ===\n  {cols}")

    # Identify the path column and label column (schema is Path + Label)
    path_col = next((c for c in cols if c.lower() == "path"), cols[0])
    label_col = next((c for c in cols if c.lower() == "label"), None)

    print("\n=== FIRST 5 EXAMPLES ===")
    for i in range(min(5, len(ds[split]))):
        row = ds[split][i]
        print(f"  {row[path_col]}  |  label={row.get(label_col)}")

    paths = ds[split][path_col]
    labels = ds[split][label_col] if label_col else [None] * len(paths)

    # ---- Plot 1: class balance from Label ----
    if label_col is not None:
        counts = Counter(labels)
        # map 0 -> bona fide, else spoof, for a readable plot
        readable = Counter()
        for k, v in counts.items():
            readable["bona fide (0)" if k == 0 else f"spoof ({k})"] += v
        bar(readable, f"Class balance ({split})", "Number of clips",
            os.path.join(args.out, "class_balance.png"), color=[GREEN, RED][:len(readable)])

    # ---- Parse metadata from paths ----
    gens, dialects, bona_flags = [], [], []
    for p in paths:
        g, d, isb = parse_path(p)
        gens.append(g); dialects.append(d); bona_flags.append(isb)

    # ---- Plot 2: by generator ----
    bar(Counter(gens), f"Samples by generator ({split})", "Number of clips",
        os.path.join(args.out, "by_generator.png"), color=NAVY, rotate=30)

    # ---- Plot 3: by dialect ----
    bar(Counter(dialects), f"Samples by dialect ({split})", "Number of clips",
        os.path.join(args.out, "by_dialect.png"), color=NAVY, rotate=30)

    # cross-check: path-derived bona fide vs Label==0
    if label_col is not None:
        path_bona = sum(bona_flags)
        label_bona = sum(1 for x in labels if x == 0)
        print(f"\nSanity check: bona fide by path={path_bona}, by label==0={label_bona}")

    # ---- Plot 4: duration histogram + a sample mel-spectrogram ----
    # Audio must be loaded from the wav files. If the dataset exposes an audio
    # feature, use it; otherwise load via soundfile from the local path.
    try:
        import librosa
        import librosa.display

        # cast Path to audio if needed
        sample_paths = paths[:args.max_audio_samples]
        durations = []
        base = os.environ.get("ARFAKE_DIR", "")  # if downloaded locally
        for rel in sample_paths:
            fp = os.path.join(base, rel) if base else rel
            try:
                y, sr = librosa.load(fp, sr=None)
                durations.append(len(y) / sr)
            except Exception:
                pass

        if durations:
            plt.figure(figsize=(6, 4))
            plt.hist(durations, bins=40, color=NAVY, edgecolor="white")
            plt.title(f"Clip duration distribution (n={len(durations)})")
            plt.xlabel("Duration (seconds)"); plt.ylabel("Count")
            plt.tight_layout()
            dp = os.path.join(args.out, "duration_hist.png")
            plt.savefig(dp, dpi=150); plt.close()
            print(f"Saved {dp}: mean={np.mean(durations):.2f}s "
                  f"median={np.median(durations):.2f}s")

            # mel-spectrogram of the first loadable clip
            fp0 = os.path.join(base, sample_paths[0]) if base else sample_paths[0]
            y, sr = librosa.load(fp0, sr=None)
            S = librosa.power_to_db(
                librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
            plt.figure(figsize=(6, 4))
            librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel")
            plt.colorbar(format="%+2.0f dB")
            plt.title("Sample mel-spectrogram"); plt.tight_layout()
            mp = os.path.join(args.out, "sample_melspectrogram.png")
            plt.savefig(mp, dpi=150); plt.close()
            print(f"Saved {mp}")
        else:
            print("Could not load audio files for duration/mel plots. "
                  "If you downloaded the dataset, set ARFAKE_DIR to its folder.")
    except ImportError:
        print("librosa not installed; skipping duration + mel plots. "
              "`pip install librosa soundfile`")

    print("\nDone. Figures written to:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
