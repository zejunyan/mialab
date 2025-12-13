import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Folder containing the MI CSV files
# --------------------------------------------------
BASE_DIR = "/Users/zejunyan/Desktop/mia-lab-project/mialab/mia-result/Core_radiomic"
csv_files = sorted(glob.glob(os.path.join(BASE_DIR, "*.csv")))

assert len(csv_files) >= 2, "Need at least 2 MI CSV files in the folder."

print(f"Found {len(csv_files)} MI files:")
for f in csv_files:
    print(" -", os.path.basename(f))

# --------------------------------------------------
# Collect per-file distributions
# --------------------------------------------------
labels = []
top7_per_file = []
rest_per_file = []

for path in csv_files:
    df = pd.read_csv(path)

    # If not sorted, uncomment next line:
    # df = df.sort_values("MI_score", ascending=False).reset_index(drop=True)

    labels.append(os.path.basename(path).replace(".csv", ""))
    top7_per_file.append(df.iloc[:7]["MI_score"].values)
    rest_per_file.append(df.iloc[7:]["MI_score"].values)

# --------------------------------------------------
# Plot: 2 subplots, each subplot has one box per file
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.3 * len(labels)), 5), sharey=True)

# Left: Top 7 boxplots per file
axes[0].boxplot(top7_per_file, labels=labels, showfliers=True)
axes[0].set_title("Top 7 MI per file (core)")
axes[0].set_ylabel("Mutual Information (MI)")
axes[0].grid(axis="y", linestyle="--", alpha=0.4)
axes[0].tick_params(axis="x", rotation=45)

# Right: Rest boxplots per file
axes[1].boxplot(rest_per_file, labels=labels, showfliers=True)
axes[1].set_title("Remaining MI per file (radiomics)")
axes[1].grid(axis="y", linestyle="--", alpha=0.4)
axes[1].tick_params(axis="x", rotation=45)

# Log scale recommended (MI spans orders of magnitude)
axes[0].set_yscale("log")
axes[1].set_yscale("log")

plt.tight_layout()

# --------------------------------------------------
# Save (terminal-safe)
# --------------------------------------------------
out_path = os.path.join(BASE_DIR, "mi_boxplots_per_file_top7_vs_rest.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"\nSaved plot to:\n{out_path}")