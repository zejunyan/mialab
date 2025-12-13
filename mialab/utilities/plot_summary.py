import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===========================================================
# CONFIG
# ===========================================================

# Folder containing all summary CSVs
FOLDER = "/Users/zejunyan/Desktop/mia-lab-project/mialab/mia-result/k=10_summary"

# Baseline feature set (CSV filename without .csv)
BASELINE_NAME = "baseline"


# ===========================================================
# LOAD DATA
# ===========================================================

def load_datasets(folder):
    datasets = {}
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            name = file.replace(".csv", "")
            df = pd.read_csv(path, sep=";")
            datasets[name] = df
    return datasets


datasets = load_datasets(FOLDER)

print(f"Loaded {len(datasets)} files from {FOLDER}:")
for name in datasets:
    print("  -", name)

first_df = next(iter(datasets.values()))
labels = first_df["LABEL"].unique()
metrics = first_df["METRIC"].unique()
print("Labels:", labels)
print("Metrics:", metrics)


# ===========================================================
# HELPERS
# ===========================================================

def collect_stats_for_metric(metric, datasets, labels):
    """Return stats[name] = (means, stds) for a given metric."""
    stats = {}
    for name, df in datasets.items():
        means = []
        stds = []
        for lab in labels:
            sub = df[(df["LABEL"] == lab) & (df["METRIC"] == metric)]
            mean = sub[sub["STATISTIC"] == "MEAN"]["VALUE"].iloc[0]
            std = sub[sub["STATISTIC"] == "STD"]["VALUE"].iloc[0]
            means.append(float(mean))
            stds.append(float(std))
        stats[name] = (np.array(means, float), np.array(stds, float))
    return stats


def metric_higher_is_better(metric: str) -> bool:
    metric = metric.upper()
    if metric == "DICE":
        return True
    # HDRFDST: lower is better
    return False


# ===========================================================
# 1) OVERALL PLOT (no baseline line)
# ===========================================================

def plot_metric_overall(metric, ylabel, filename, datasets, labels, folder):
    stats = collect_stats_for_metric(metric, datasets, labels)
    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))

    width = 0.12
    offsets = {}
    counter = 0
    for name in stats:
        offsets[name] = (counter - len(stats) / 2) * width
        counter += 1

    for name, (means, stds) in stats.items():
        plt.errorbar(
            x + offsets[name],
            means,
            yerr=stds,
            fmt="o",
            linestyle="none",
            markersize=6,
            capsize=4,
            label=name,
        )

    plt.xticks(x, labels, rotation=30)
    plt.ylabel(ylabel)
    plt.title(f"{metric} (mean ± SD) across labels and all feature sets")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join(folder, filename)
    plt.savefig(outpath, dpi=300)
    print(f"Saved overall plot: {outpath}")
    plt.close()


# ===========================================================
# 2) PER-LABEL PLOTS (baseline line + green/red points)
# ===========================================================

def plot_metric_per_label(metric, ylabel, filename_prefix, datasets, labels,
                          folder, baseline_name=None, zoom_margin_factor=0.2):

    higher_is_better = metric_higher_is_better(metric)

    for lab in labels:
        names = []
        means = []
        stds = []

        for name, df in datasets.items():
            sub = df[(df["LABEL"] == lab) & (df["METRIC"] == metric)]
            mean = sub[sub["STATISTIC"] == "MEAN"]["VALUE"].iloc[0]
            std = sub[sub["STATISTIC"] == "STD"]["VALUE"].iloc[0]
            names.append(name)
            means.append(float(mean))
            stds.append(float(std))

        means = np.array(means, float)
        stds = np.array(stds, float)
        x = np.arange(len(names))

        plt.figure(figsize=(8, 4))

        baseline_mean = None
        if baseline_name is not None and baseline_name in names:
            baseline_mean = means[names.index(baseline_name)]
            plt.axhline(
                baseline_mean,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label=f"baseline ({baseline_name})"
            )

        for xi, m, s, name in zip(x, means, stds, names):
            if baseline_mean is None:
                color = "black"
            else:
                if higher_is_better:
                    diff = m - baseline_mean
                else:
                    diff = baseline_mean - m

                if np.isclose(diff, 0.0, atol=1e-6):
                    color = "black"
                elif diff > 0:
                    color = "green"
                else:
                    color = "red"

            plt.errorbar(
                xi,
                m,
                yerr=s,
                fmt="o",
                color=color,
                linestyle="none",
                markersize=6,
                capsize=4,
            )

        plt.xticks(x, names, rotation=45, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"{metric} (mean ± SD) for label {lab}")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        y_min = np.min(means - stds)
        y_max = np.max(means + stds)
        if y_max > y_min:
            margin = (y_max - y_min) * zoom_margin_factor
        else:
            margin = 0.05 * abs(y_max) if y_max != 0 else 0.01
        plt.ylim(y_min - margin, y_max + margin)

        safe_label = str(lab).replace(" ", "_")
        outname = f"{filename_prefix}_{metric}_{safe_label}.png"
        outpath = os.path.join(folder, outname)
        plt.savefig(outpath, dpi=300)
        print(f"Saved per-label plot: {outpath}")
        plt.close()


# ===========================================================
# 3) SUBPLOT FIGURE (baseline line + green/red points)
# ===========================================================

def plot_metric_subplots(metric, ylabel, filename, datasets, labels,
                         folder, baseline_name=None, ncols=3, zoom_margin_factor=0.2):

    higher_is_better = metric_higher_is_better(metric)

    n_labels = len(labels)
    nrows = math.ceil(n_labels / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        squeeze=False
    )
    axes = axes.flatten()

    for idx, lab in enumerate(labels):
        ax = axes[idx]

        names = []
        means = []
        stds = []

        for name, df in datasets.items():
            sub = df[(df["LABEL"] == lab) & (df["METRIC"] == metric)]
            mean = sub[sub["STATISTIC"] == "MEAN"]["VALUE"].iloc[0]
            std = sub[sub["STATISTIC"] == "STD"]["VALUE"].iloc[0]
            names.append(name)
            means.append(float(mean))
            stds.append(float(std))

        means = np.array(means, float)
        stds = np.array(stds, float)
        x = np.arange(len(names))

        baseline_mean = None
        if baseline_name is not None and baseline_name in names:
            baseline_mean = means[names.index(baseline_name)]
            ax.axhline(
                baseline_mean,
                color="gray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8
            )

        for xi, m, s, name in zip(x, means, stds, names):
            if baseline_mean is None:
                color = "black"
            else:
                if higher_is_better:
                    diff = m - baseline_mean
                else:
                    diff = baseline_mean - m

                if np.isclose(diff, 0.0, atol=1e-6):
                    color = "black"
                elif diff > 0:
                    color = "green"
                else:
                    color = "red"

            ax.errorbar(
                xi,
                m,
                yerr=s,
                fmt="o",
                color=color,
                linestyle="none",
                markersize=5,
                capsize=3,
            )

        ax.set_title(f"Label {lab}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        y_min = np.min(means - stds)
        y_max = np.max(means + stds)
        if y_max > y_min:
            margin = (y_max - y_min) * zoom_margin_factor
        else:
            margin = 0.05 * abs(y_max) if y_max != 0 else 0.01
        ax.set_ylim(y_min - margin, y_max + margin)

    for j in range(n_labels, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{metric} (mean ± SD) per label", fontsize=14)
    fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')
    fig.tight_layout(rect=[0.06, 0.04, 1.0, 0.95])

    outpath = os.path.join(folder, filename)
    fig.savefig(outpath, dpi=300)
    print(f"Saved subplot figure: {outpath}")
    plt.close(fig)


# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":
    # overall plots (no baseline line)
    plot_metric_overall("DICE",    "DICE score",
                        "DICE_overall.png",
                        datasets, labels, FOLDER)

    plot_metric_overall("HDRFDST", "Hausdorff distance",
                        "HDRFDST_overall.png",
                        datasets, labels, FOLDER)

    # per-label detailed plots
    plot_metric_per_label("DICE",    "DICE score",
                          "per_label_DICE",
                          datasets, labels, FOLDER,
                          baseline_name=BASELINE_NAME)

    plot_metric_per_label("HDRFDST", "Hausdorff distance",
                          "per_label_HDRFDST",
                          datasets, labels, FOLDER,
                          baseline_name=BASELINE_NAME)

    # subplots (all labels in one figure)
    plot_metric_subplots("DICE",    "DICE score",
                         "DICE_subplots.png",
                         datasets, labels, FOLDER,
                         baseline_name=BASELINE_NAME, ncols=3)

    plot_metric_subplots("HDRFDST", "Hausdorff distance",
                         "HDRFDST_subplots.png",
                         datasets, labels, FOLDER,
                         baseline_name=BASELINE_NAME, ncols=3)