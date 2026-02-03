import pandas as pd
from pathlib import Path

BASE_DIR = Path.cwd()

# Result folders
folders = [BASE_DIR / f"Results - I{i}" for i in range(1, 11)]

dfs = []

for i, folder in enumerate(folders, start=1):
    file = folder / "clustering_comparison_results_UML.csv"
    df = pd.read_csv(file)
    df["instance"] = i
    dfs.append(df)

# Combine all instances
all_results = pd.concat(dfs, ignore_index=True)

# Metrics of interest
metrics = [
    "precision_ratio",
    "recall_ratio",
    "f1_ratio",
    "accuracy_ratio"
]

# Group and summarise
summary = (
    all_results
    .groupby(["method", "cluster_distance", "cluster_std"])[metrics]
    .agg(["mean", "std"])
    .reset_index()
)

# Flatten column names
summary.columns = [
    "_".join(col).strip("_") for col in summary.columns
]

# Add coefficient of variation (CV)
for metric in metrics:
    summary[f"{metric}_cv"] = (
        summary[f"{metric}_std"] / summary[f"{metric}_mean"]
    )

# Save
summary.to_csv("clustering_variability_summary.csv", index=False)

print("Variability summary saved to clustering_variability_summary.csv")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as ticker

# Paths
BASE_DIR = Path.cwd()
SUMMARY_FILE = BASE_DIR / "clustering_variability_summary.csv"

# Load summary
df = pd.read_csv(SUMMARY_FILE)

metrics = [
    "precision_ratio_cv",
    "recall_ratio_cv",
    "f1_ratio_cv",
    "accuracy_ratio_cv"
]

titles = [
    "Precision Ratio CV",
    "Recall Ratio CV",
    "F1-score Ratio CV",
    "Accuracy Ratio CV"
]

sns.set_theme(style="white")

methods = df["method"].unique()

for method in methods:
    df_m = df[df["method"] == method]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, metric, title in zip(axes, metrics, titles):
        pivot = df_m.pivot_table(
            index="cluster_std",
            columns="cluster_distance",
            values=metric
        )

        # Format tick labels to one decimal place
        y_labels = [f"{v:.1f}" for v in sorted(df_m["cluster_std"].unique())]
        x_labels = [f"{v:.1f}" for v in sorted(df_m["cluster_distance"].unique())]

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="viridis",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Coefficient of Variation"},
            xticklabels=x_labels,
            yticklabels=y_labels
        )

        ax.set_title(title)
        ax.set_xlabel("Cluster Distance")
        ax.set_ylabel("Cluster Standard Deviation")

    fig.suptitle(f"{method} – Variability of Performance Ratios", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    outname = f"variability_CV_{method.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(outname, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {outname}")
