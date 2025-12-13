import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_pp_comparison(csv_path, metric="DICE", title_prefix="Baseline Test"):
    # Load CSV
    df = pd.read_csv(csv_path, sep=';')

    # Create PP/Non-PP indicator
    df["PP"] = df["SUBJECT"].apply(lambda x: "PP" if str(x).endswith("-PP") else "Non-PP")
    df["SUBJECT_CLEAN"] = df["SUBJECT"].str.replace("-PP", "", regex=False)

    labels = ["WhiteMatter", "GreyMatter", "Hippocampus", "Thalamus", "Amygdala"]

    data_nonpp = []
    data_pp = []
    positions = []
    xticks = []

    pos = 0
    width = 0.35

    for label in labels:
        sub = df[df["LABEL"] == label]
        if sub.empty:
            continue

        data_nonpp.append(sub[sub["PP"] == "Non-PP"][metric])
        data_pp.append(sub[sub["PP"] == "PP"][metric])

        positions.append(pos)
        xticks.append(label)
        pos += 1

    # Plot
    positions_nonpp = [p - width/2 for p in positions]
    positions_pp     = [p + width/2 for p in positions]

    plt.figure(figsize=(11, 6))
    plt.boxplot(data_nonpp, positions=positions_nonpp, widths=width, patch_artist=True)
    plt.boxplot(data_pp, positions=positions_pp, widths=width, patch_artist=True)

    plt.xticks(positions, xticks, rotation=45)
    plt.ylabel(metric)
    plt.title(f"{title_prefix}: {metric} per Label (PreP vs PostP)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Automatically detect the timestamped folder
    output_dir = os.path.dirname(os.path.abspath(csv_path))

    # Save plot inside same folder
    output_file = os.path.join(output_dir, f"{metric}_PostP_vs_PreP.png")
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to: {output_file}")

    plt.close()

# ---------------- TERMINAL ENTRY POINT -----------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_pp.py <results.csv> <metric>")
        print("Example: python plot_pp.py results.csv DICE")
        sys.exit(1)

    csv_file = sys.argv[1]
    metric_name = sys.argv[2]

    plot_pp_comparison(csv_file, metric=metric_name)