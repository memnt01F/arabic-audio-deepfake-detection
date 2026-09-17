"""Analyze the ArFake label files.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# File locations.
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
figures_folder = Path("figures")
figures_folder.mkdir(exist_ok=True)


# Convert the numeric labels into readable names.
label_names = {
    0: "Bona fide (real)",
    1: "Spoof (fake)",
}
label_order = ["Bona fide (real)", "Spoof (fake)"]


def add_information(data):
    """Extract the source and dialect from each Path."""
    data = data.copy()

    # Example path:
    # fish-speech-new/Morocco/bonafied/file.wav
    path_parts = (
        data["Path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/")
    )

    data["source"] = path_parts.str[0]
    data["dialect"] = path_parts.str[1]
    data["class_name"] = data["Label"].map(label_names)

    return data


train = add_information(train)
test = add_information(test)

all_data = pd.concat(
    [train.assign(split="train"), test.assign(split="test")],
    ignore_index=True,
)


print(f"Train rows: {len(train):,}")
print(f"Test rows:  {len(test):,}")
print(f"Total rows: {len(all_data):,}")

print("\nTrain labels:")
print(train["class_name"].value_counts().reindex(label_order, fill_value=0))

print("\nTest labels:")
print(test["class_name"].value_counts().reindex(label_order, fill_value=0))

print("\nSamples by source group:")
print(all_data["source"].value_counts())

print("\nSamples by dialect:")
print(all_data["dialect"].value_counts())

print("\nReal/fake counts within each source group:")
print(
    pd.crosstab(all_data["source"], all_data["class_name"])
    .reindex(columns=label_order, fill_value=0)
)

spoof_percentage = all_data["Label"].eq(1).mean() * 100
print(f"\nOverall spoof percentage: {spoof_percentage:.1f}%")


def save_plot(filename):
    """Save the current plot in the figures folder."""
    plt.tight_layout()
    plt.savefig(figures_folder / filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {figures_folder / filename}")


# Figure 1: class balance in train and test.
class_counts = pd.crosstab(all_data["split"], all_data["class_name"])
class_counts = class_counts.reindex(
    index=["train", "test"],
    columns=label_order,
    fill_value=0,
)
class_counts.plot(
    kind="bar",
    color=["#2E6B3E", "#9E2B25"],
    figsize=(7, 4),
)
plt.title("Class balance: bona fide vs spoof")
plt.xlabel("Dataset split")
plt.ylabel("Number of clips")
plt.xticks(rotation=0)
plt.legend(title="Class")
save_plot("class_balance.png")


# Figure 2: number of clips from each source group.
all_data["source"].value_counts().sort_values().plot(
    kind="barh",
    color="#1F2A44",
    figsize=(7, 4),
)
plt.title("Samples by source group")
plt.xlabel("Number of clips")
plt.ylabel("Source group")
save_plot("by_source.png")


# Figure 3: number of clips from each dialect.
all_data["dialect"].value_counts().sort_values().plot(
    kind="barh",
    color="#1F2A44",
    figsize=(7, 4),
)
plt.title("Samples by dialect")
plt.xlabel("Number of clips")
plt.ylabel("Dialect")
save_plot("by_dialect.png")


# Figure 4: real and fake clips within each source group.
source_class_counts = pd.crosstab(
    all_data["source"], all_data["class_name"]
).reindex(columns=label_order, fill_value=0)
source_class_counts.plot(
    kind="bar",
    stacked=True,
    color=["#2E6B3E", "#9E2B25"],
    figsize=(7, 4),
)
plt.title("Real vs fake clips within each source group")
plt.xlabel("Source group")
plt.ylabel("Number of clips")
plt.xticks(rotation=25)
plt.legend(title="Class")
save_plot("source_label_stacked.png")

print(f"\nAll figures were saved in: {figures_folder.resolve()}")

