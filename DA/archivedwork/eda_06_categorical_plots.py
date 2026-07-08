"""
EDA Script 6 — Categorical Encoded Features vs cost_overrun (box plots + bar charts)
Outputs: boxplot_categoricals.png, barplot_binary_flags.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("DA\hydropower_ml_ready.csv").dropna(subset=["cost_overrun"])

# ── 1. Box plots for ordinal / multi-level categoricals ─────────────────────
ORDINAL_COLS = {
    "category_encoded":    {1: "small", 2: "large", 3: "mega"},
    "seismic_encoded":     {3.0: "Zone IV", 3.5: "Zone IV/V", 4.0: "Zone V"},
    "retendered_encoded":  {0: "NO", 1: "NO-LD", 2: "YES"},
    "contract_encoded":    None,   # too many values, use as-is
    "state_encoded":       {0: "Himachal", 1: "Uttarakhand"},
}
ORDINAL_COLS = {k: v for k, v in ORDINAL_COLS.items() if k in df.columns}

fig, axes = plt.subplots(1, len(ORDINAL_COLS), figsize=(len(ORDINAL_COLS) * 4, 6))
if len(ORDINAL_COLS) == 1:
    axes = [axes]

for ax, (col, label_map) in zip(axes, ORDINAL_COLS.items()):
    groups = df.groupby(col)["cost_overrun"].apply(list)
    labels = [label_map.get(k, str(k)) if label_map else str(k) for k in groups.index]

    bp = ax.boxplot(groups.values, patch_artist=True, labels=labels,
                    medianprops=dict(color="tomato", linewidth=2))
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # overlay points
    for j, (_, group_data) in enumerate(groups.items()):
        ax.scatter(np.full(len(group_data), j + 1) +
                   np.random.uniform(-0.05, 0.05, len(group_data)),
                   group_data, s=25, alpha=0.6, color="navy", zorder=3)

    ax.set_title(col, fontsize=9)
    ax.set_yscale("symlog")
    ax.set_ylabel("cost_overrun (%)" if ax == axes[0] else "")
    ax.tick_params(axis="x", labelsize=8, rotation=15)

plt.suptitle("cost_overrun Distribution by Categorical Features", fontsize=13)
plt.tight_layout()

plt.savefig("boxplot_categoricals.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 2. Bar charts for binary flags ───────────────────────────────────────────
BINARY_COLS = ["has_geo_prob", "state_encoded", "funder_multilateral",
               "funder_central_govt", "funder_state_govt",
               "funder_private", "funder_psu", "contract_is_hybrid"]
BINARY_COLS = [c for c in BINARY_COLS if c in df.columns]

ncols = 4
nrows = (len(BINARY_COLS) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
axes = axes.flatten()

for i, col in enumerate(BINARY_COLS):
    group_means = df.groupby(col)["cost_overrun"].agg(["mean", "sem"])
    labels = ["No (0)", "Yes (1)"][:len(group_means)]
    colors = ["steelblue", "tomato"][:len(group_means)]

    bars = axes[i].bar(labels, group_means["mean"], color=colors,
                       edgecolor="white", alpha=0.8,
                       yerr=group_means["sem"], capsize=5)

    # t-test between groups
    g0 = df[df[col] == 0]["cost_overrun"].dropna()
    g1 = df[df[col] == 1]["cost_overrun"].dropna()
    if len(g0) > 1 and len(g1) > 1:
        _, p = stats.ttest_ind(g0, g1)
        axes[i].set_title(f"{col}\np={p:.3f}", fontsize=8,
                          color="green" if p < 0.05 else "black")
    else:
        axes[i].set_title(col, fontsize=8)

    axes[i].set_ylabel("mean cost_overrun (%)", fontsize=8)
    axes[i].tick_params(labelsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Mean cost_overrun by Binary Flags\n(error bars = SEM, green title = p<0.05)",
             fontsize=12)
plt.tight_layout()
plt.savefig("barplot_binary_flags.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: boxplot_categoricals.png, barplot_binary_flags.png")
