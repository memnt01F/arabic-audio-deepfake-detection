"""
baseline_cnn.py
---------------
Skeleton of the deep-learning core for the proposal: a CNN that classifies a
mel-spectrogram of an audio clip as bona fide (real) or spoof (fake).

This is intentionally a compact, readable baseline — the "clearly ours"
trained model referenced in the proposal. It is NOT meant to reach SOTA;
it establishes the pipeline (audio -> mel-spectrogram -> CNN -> real/fake)
and the EER metric that later, stronger front ends (RawNet2, wav2vec2)
will be compared against.

Run only after inspect_arfake.py confirms dataset access.

  python baseline_cnn.py --epochs 10 --batch-size 32
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Metric: Equal Error Rate (the anti-spoofing standard)
# ---------------------------------------------------------------------------
def compute_eer(scores, labels):
    """
    scores: model probability that a clip is SPOOF (higher = more likely fake)
    labels: 1 for spoof, 0 for bona fide
    Returns EER in percent.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    thresholds = np.sort(np.unique(scores))
    fars, frrs = [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        # False acceptance: spoof accepted as real  -> predicted 0 but label 1
        fa = np.sum((pred == 0) & (labels == 1))
        # False rejection: real rejected as spoof   -> predicted 1 but label 0
        fr = np.sum((pred == 1) & (labels == 0))
        n_spoof = max(np.sum(labels == 1), 1)
        n_real = max(np.sum(labels == 0), 1)
        fars.append(fa / n_spoof)
        frrs.append(fr / n_real)
    fars, frrs = np.array(fars), np.array(frrs)
    idx = np.nanargmin(np.abs(fars - frrs))
    return (fars[idx] + frrs[idx]) / 2 * 100.0


# ---------------------------------------------------------------------------
# Dataset: wraps a HuggingFace split, turns audio into fixed-size mel-specs
# ---------------------------------------------------------------------------
class MelSpecDataset(Dataset):
    """
    ArFake real schema: each row has 'Path' (relative wav path) and 'Label'
    (0 = bona fide, non-zero = spoof). Audio is loaded from the wav file.
    Set env var ARFAKE_DIR to the download folder if paths are relative.
    """
    def __init__(self, hf_split, path_col="Path", label_col="Label", sr=16000,
                 n_mels=128, max_frames=256):
        self.ds = hf_split
        self.path_col = path_col
        self.label_col = label_col
        self.sr = sr
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.base = os.environ.get("ARFAKE_DIR", "")
        import librosa
        self.librosa = librosa

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        rel = row[self.path_col]
        fp = os.path.join(self.base, rel) if self.base else rel
        y, sr = self.librosa.load(fp, sr=self.sr)  # resamples to self.sr
        S = self.librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        S = self.librosa.power_to_db(S, ref=np.max)
        # pad / crop to fixed number of frames
        if S.shape[1] < self.max_frames:
            S = np.pad(S, ((0, 0), (0, self.max_frames - S.shape[1])))
        else:
            S = S[:, :self.max_frames]
        S = (S - S.mean()) / (S.std() + 1e-6)
        x = torch.from_numpy(S).unsqueeze(0).float()  # (1, n_mels, frames)
        # binarize: 0 -> bona fide (0), any non-zero -> spoof (1)
        label = 0 if int(row[self.label_col]) == 0 else 1
        return x, label


# ---------------------------------------------------------------------------
# Model: small CNN over the mel-spectrogram
# ---------------------------------------------------------------------------
class SpecCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64), block(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def run_epoch(model, loader, criterion, optim, device, train=True):
    model.train() if train else model.eval()
    total_loss, scores, labels = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optim.zero_grad(); loss.backward(); optim.step()
            total_loss += loss.item() * x.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(spoof)
            scores.extend(probs.detach().cpu().numpy())
            labels.extend(y.cpu().numpy())
    eer = compute_eer(scores, labels)
    return total_loss / len(loader.dataset), eer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    from datasets import load_dataset
    ds = load_dataset("Mohammed01/ArFake")

    # Real ArFake schema: 'Path' + 'Label'
    path_col, label_col = "Path", "Label"

    # ArFake provides train/test; carve a validation set out of train.
    train_split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    split = train_split.train_test_split(test_size=args.val_frac, seed=42)
    tr, va = split["train"], split["test"]

    train_ds = MelSpecDataset(tr, path_col, label_col)
    val_ds = MelSpecDataset(va, path_col, label_col)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    model = SpecCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_eer = 100.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_eer = run_epoch(model, train_dl, criterion, optim, device, train=True)
        va_loss, va_eer = run_epoch(model, val_dl, criterion, optim, device, train=False)
        print(f"epoch {ep:02d} | train loss {tr_loss:.3f} EER {tr_eer:.2f}% "
              f"| val loss {va_loss:.3f} EER {va_eer:.2f}%")
        if va_eer < best_eer:
            best_eer = va_eer
            torch.save(model.state_dict(), "best_speccnn.pt")
    print(f"Best validation EER: {best_eer:.2f}%  (saved best_speccnn.pt)")


if __name__ == "__main__":
    main()

