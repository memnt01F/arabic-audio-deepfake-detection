"""
analyze_csv.py
--------------
Analyzes ArFake's label tables.
Prints class balance, generator, and dialect breakdowns, and saves
the distribution plots required by the proposal.

Each CSV has two columns:
  Path   e.g. "fish-speech-new/Morocco/bonafied/original_19037_3.wav"
  Label  0 = bona fide (real), 1 = spoof (fake)
Generator and dialect are parsed from the Path.


"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1F2A44"; RED = "#9E2B25"; GREEN = "#2E6B3E"
plt.rcParams["font.size"] = 11


def parse_path(p):
    """Extract (generator, dialect, is_bonafide) from the Path string."""
    parts = str(p).replace("\\", "/").split("/")
    gen = parts[0] if len(parts) > 0 else "unknown"
    dia = parts[1] if len(parts) > 1 else "unknown"
    is_bona = bool(re.search("bona", str(p).lower()))
    return gen, dia, is_bona


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tr = pd.read_csv(args.train)
    te = pd.read_csv(args.test)
    print("train:", tr.shape, " test:", te.shape)
    print("columns:", list(tr.columns))

    print("\n=== Label counts (train) ===")
    print(tr.Label.value_counts())
    print("\n=== Label counts (test) ===")
    print(te.Label.value_counts())

    # parse metadata from paths
    for df in (tr, te):
        g, d, b = zip(*[parse_path(p) for p in df.Path])
        df["generator"] = g
        df["dialect"] = d
        df["is_bona_path"] = b

    alldf = pd.concat([tr.assign(split="train"), te.assign(split="test")],
                      ignore_index=True)

    print("\n=== Generator breakdown (all) ===")
    print(alldf.generator.value_counts())
    print("\n=== Dialect breakdown (all) ===")
    print(alldf.dialect.value_counts())
    print("\n=== Label vs generator (all) ===")
    print(pd.crosstab(alldf.generator, alldf.Label))
    print("\n=== Sanity: bona-by-path vs Label==0 ===")
    print("path-bona:", int(alldf.is_bona_path.sum()),
          " label0:", int((alldf.Label == 0).sum()))

    imb = (alldf.Label == 1).mean() * 100
    print(f"\n>>> Spoof share overall: {imb:.1f}%  (dataset is imbalanced)")

    # ---- Fig 1: class balance train vs test ----
    labels_sorted = sorted(alldf.Label.unique())
    names = ["bona fide (0)" if l == 0 else f"spoof ({l})" for l in labels_sorted]
    tr_counts = [(tr.Label == l).sum() for l in labels_sorted]
    te_counts = [(te.Label == l).sum() for l in labels_sorted]
    x = np.arange(len(labels_sorted)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(x - w/2, tr_counts, w, label="train", color=NAVY)
    ax.bar(x + w/2, te_counts, w, label="test", color=RED)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Number of clips"); ax.set_title("Class balance: bona fide vs spoof")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "class_balance.png"), dpi=150); plt.close()

    # ---- Fig 2: by generator ----
    gc = alldf.generator.value_counts()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(gc.index, gc.values, color=NAVY)
    ax.set_ylabel("Number of clips"); ax.set_title("Samples by source / generator")
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "by_generator.png"), dpi=150); plt.close()

    # ---- Fig 3: by dialect ----
    dc = alldf.dialect.value_counts()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(dc.index, dc.values, color=NAVY)
    ax.set_ylabel("Number of clips"); ax.set_title("Samples by dialect")
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "by_dialect.png"), dpi=150); plt.close()

    # ---- Fig 4: real vs fake within each source ----
    ct = pd.crosstab(alldf.generator, alldf.Label)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bottom = np.zeros(len(ct)); colors = [GREEN, RED, "#c77", "#a44"]
    for i, l in enumerate(ct.columns):
        ax.bar(ct.index, ct[l].values, bottom=bottom,
               label=("bona fide" if l == 0 else f"spoof {l}"),
               color=colors[i % len(colors)])
        bottom += ct[l].values
    ax.set_ylabel("Number of clips"); ax.set_title("Real vs fake within each source")
    plt.xticks(rotation=25, ha="right"); ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "gen_label_stacked.png"), dpi=150); plt.close()

    print(f"\nFigures written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
