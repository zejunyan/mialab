import os
import math
import pandas as pd
import matplotlib.pyplot as plt


# # ============================
# # 1) CONFIG
# # ============================
# FOLDER = "/Users/zejunyan/Desktop/mia-lab-project/mialab/mia-result/K_summary"

# FEATURES = ["shape", "fo", "glcm", "gldm", "glrlm", "glszm", "ngtdm"]  # edit as needed

# ORDER = ["baseline", "All", "K3", "K5", "K10"]

# SAVE = True
# OUTDIR = os.path.join(FOLDER, "plots_compare", "all_features")
# os.makedirs(OUTDIR, exist_ok=True)


# # ============================
# # 2) ROBUST CSV LOADER
# # ============================
# def read_csv_flexible(path):
#     for sep in [",", ";", "\t"]:
#         try:
#             df = pd.read_csv(path, sep=sep, engine="python")
#             if df.shape[1] == 1:
#                 continue
#             return df
#         except Exception:
#             continue
#     return pd.read_csv(path, sep=None, engine="python")


# def load_csvs_for_feature(folder, feature):
#     files = {
#         "baseline": "baseline.csv",
#         "All": f"All_baseline_{feature}.csv",
#         "K3": f"K3_baseline_{feature}.csv",
#         "K5": f"K5_baseline_{feature}.csv",
#         "K10": f"K10_baseline_{feature}.csv",
#     }

#     data = {}
#     for setting, fname in files.items():
#         path = os.path.join(folder, fname)
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Missing file for feature='{feature}', setting='{setting}': {path}")

#         df = read_csv_flexible(path)
#         df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

#         required = {"LABEL", "METRIC", "STATISTIC", "VALUE"}
#         missing = required - set(df.columns)
#         if missing:
#             print(f"\nDEBUG: columns in {fname} = {list(df.columns)}")
#             raise ValueError(f"{fname} is missing columns: {missing}")

#         data[setting] = df

#     return data


# def get_labels(data_dict):
#     labels = set()
#     for df in data_dict.values():
#         labels |= set(df["LABEL"].unique())
#     return sorted(labels)


# def get_stat_value(df, label, metric, stat="MEAN"):
#     sub = df[(df["LABEL"] == label) & (df["METRIC"] == metric) & (df["STATISTIC"] == stat)]
#     if len(sub) == 0:
#         return None
#     return float(sub["VALUE"].values[0])


# # ============================
# # 3) PLOT ALL FEATURES IN ONE FIGURE
# # ============================
# def plot_all_features(metric, ylabel, title, outname):
#     n = len(FEATURES)
#     ncols = 3
#     nrows = math.ceil(n / ncols)

#     fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
#     x_positions = list(range(len(ORDER)))

#     for idx, feature in enumerate(FEATURES):
#         r, c = divmod(idx, ncols)
#         ax = axes[r][c]

#         data = load_csvs_for_feature(FOLDER, feature)
#         labels = get_labels(data)

#         for lab in labels:
#             means, stds = [], []
#             for setting in ORDER:
#                 means.append(get_stat_value(data[setting], lab, metric, "MEAN"))
#                 stds.append(get_stat_value(data[setting], lab, metric, "STD"))

#             ax.errorbar(
#                 x_positions, means, yerr=stds,
#                 fmt="o", capsize=3, label=lab
#             )

#         ax.set_title(feature)
#         ax.set_xticks(x_positions)
#         ax.set_xticklabels(ORDER)
#         ax.set_ylabel(ylabel)
#         ax.grid(True, axis="y")

#     # Hide any unused subplots
#     for j in range(n, nrows * ncols):
#         r, c = divmod(j, ncols)
#         axes[r][c].axis("off")

#     # One shared legend (bottom-right)
#     handles, labels = None, None
#     for ax in fig.axes:
#         h, l = ax.get_legend_handles_labels()
#         if h:
#             handles, labels = h, l
#             break

#     if handles:
#         fig.legend(
#             handles,
#             labels,
#             loc="lower right",
#             bbox_to_anchor=(0.98, 0.02),  # bottom-right corner
#             fontsize=9,
#             frameon=True
#         )

#     fig.suptitle(title, y=0.98)
#     fig.tight_layout(rect=[0, 0, 1, 0.93])

#     if SAVE:
#         path = os.path.join(OUTDIR, outname)
#         fig.savefig(path, dpi=200)
#         print(f"Saved: {path}")

#     plt.show()


# def main():
#     # One figure for DICE (all features)
#     plot_all_features(
#         metric="DICE",
#         ylabel="Dice (higher is better)",
#         title="DICE comparison across settings (all features)",
#         outname="ALLFEATURES_compare_DICE.png"
#     )

#     # One figure for HD (all features)
#     plot_all_features(
#         metric="HDRFDST",
#         ylabel="Hausdorff Distance (lower is better)",
#         title="HDRFDST comparison across settings (all features)",
#         outname="ALLFEATURES_compare_HDRFDST.png"
#     )


# if __name__ == "__main__":
#     main()

import os
import math
import pandas as pd
import matplotlib.pyplot as plt


# ============================
# 1) CONFIG
# ============================
FOLDER = "/Users/zejunyan/Desktop/mia-lab-project/mialab/mia-result/K_summary"

FEATURES = ["shape", "fo", "glcm", "gldm", "glrlm", "glszm", "ngtdm"]
ORDER = ["baseline", "All", "K3", "K5", "K10"]

SAVE = True
OUTDIR = os.path.join(FOLDER, "plots_compare", "per_feature_per_label")
os.makedirs(OUTDIR, exist_ok=True)


# ============================
# 2) ROBUST CSV LOADER
# ============================
def read_csv_flexible(path):
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] == 1:
                continue
            return df
        except Exception:
            continue
    return pd.read_csv(path, sep=None, engine="python")


def load_csvs_for_feature(folder, feature):
    files = {
        "baseline": "baseline.csv",
        "All": f"All_baseline_{feature}.csv",
        "K3": f"K3_baseline_{feature}.csv",
        "K5": f"K5_baseline_{feature}.csv",
        "K10": f"K10_baseline_{feature}.csv",
    }

    data = {}
    for setting, fname in files.items():
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file for feature='{feature}', setting='{setting}': {path}"
            )

        df = read_csv_flexible(path)
        df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

        required = {"LABEL", "METRIC", "STATISTIC", "VALUE"}
        missing = required - set(df.columns)
        if missing:
            print(f"\nDEBUG: columns in {fname} = {list(df.columns)}")
            raise ValueError(f"{fname} is missing columns: {missing}")

        data[setting] = df

    return data


def get_labels(data_dict):
    labels = set()
    for df in data_dict.values():
        labels |= set(df["LABEL"].unique())
    return sorted(labels)


def get_stat_value(df, label, metric, stat="MEAN"):
    sub = df[
        (df["LABEL"] == label)
        & (df["METRIC"] == metric)
        & (df["STATISTIC"] == stat)
    ]
    if len(sub) == 0:
        return None
    return float(sub["VALUE"].values[0])


# ============================
# 3) PER LABEL, PER FEATURE
# ============================
def plot_per_feature_per_label(feature, metric, ylabel, outname):
    data = load_csvs_for_feature(FOLDER, feature)
    labels = get_labels(data)

    n = len(labels)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False
    )

    x_positions = list(range(len(ORDER)))

    for idx, lab in enumerate(labels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        means, stds = [], []
        for setting in ORDER:
            means.append(get_stat_value(data[setting], lab, metric, "MEAN"))
            stds.append(get_stat_value(data[setting], lab, metric, "STD"))

        # --- Baseline (black) ---
        baseline_mean = means[0]
        baseline_std = stds[0]

        ax.errorbar(
            x_positions[0],
            baseline_mean,
            yerr=baseline_std,
            fmt="o",
            color="black",
            capsize=4
        )

        ax.axhline(
            y=baseline_mean,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7
        )

        # --- Other settings (no lines) ---
        ax.errorbar(
            x_positions[1:],
            means[1:],
            yerr=stds[1:],
            fmt="o",
            capsize=3
        )

        ax.set_title(lab)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(ORDER)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle(f"{feature} — {metric} (baseline vs K)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if SAVE:
        path = os.path.join(OUTDIR, outname)
        fig.savefig(path, dpi=200)
        print(f"Saved: {path}")

    plt.show()


def main():
    for feature in FEATURES:
        plot_per_feature_per_label(
            feature=feature,
            metric="DICE",
            ylabel="Dice (higher is better)",
            outname=f"{feature}_PERLABEL_DICE.png"
        )

        plot_per_feature_per_label(
            feature=feature,
            metric="HDRFDST",
            ylabel="Hausdorff Distance (lower is better)",
            outname=f"{feature}_PERLABEL_HDRFDST.png"
        )


if __name__ == "__main__":
    main()