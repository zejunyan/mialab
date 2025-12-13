import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric_mean_std(csv_path, metric="DICE", title_prefix="Baseline Test"):
    """
    Create an error-bar plot (mean ± std) per label for a given metric,
    WITHOUT connecting lines between points.
    """

    # --- Load data ---
    df = pd.read_csv(csv_path, sep=';')

    # keep only the chosen metric
    sub = df[df["METRIC"] == metric].copy()

    # Pivot to extract MEAN and STD
    table = sub.pivot(index="LABEL", columns="STATISTIC", values="VALUE")

    # enforce label order
    label_order = ["WhiteMatter", "GreyMatter", "Hippocampus", "Thalamus", "Amygdala"]
    table = table.reindex(label_order).dropna(how="any")

    labels = table.index.tolist()
    means = table["MEAN"].values
    stds = table["STD"].values

    x = range(len(labels))

    # --- Plot without connecting lines ---
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x, means, yerr=stds,
        fmt="o",           # <-- marker only, NO line
        capsize=5,
        markersize=8
    )

    plt.xticks(x, labels, rotation=45)
    plt.ylabel(metric)
    plt.title(f"{title_prefix}: {metric} (mean ± SD)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save figure in same directory as CSV
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    out_file = os.path.join(out_dir, f"{metric}_mean_std.png")
    plt.savefig(out_file, dpi=300)
    print(f"Saved plot to: {out_file}")

    plt.close()


# ---------------------- Terminal entry -----------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_metric_mean_std.py <stats.csv> <METRIC>")
        print("Example: python plot_metric_mean_std.py stats.csv DICE")
        sys.exit(1)

    csv_file = sys.argv[1]
    metric_name = sys.argv[2]

    plot_metric_mean_std(csv_file, metric=metric_name)