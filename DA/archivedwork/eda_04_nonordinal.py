"""
EDA Script 4b — Group Difference Tests for Non-Ordinal Encoded Columns
Replaces correlation for contract_encoded, geo_prob_encoded, funder_encoded
Outputs: anova_nonordinal.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, f_oneway

df = pd.read_csv("DA\hydropower_ml_ready.csv").dropna(subset=["cost_overrun"])

# Columns where integer codes have NO meaningful order
NON_ORDINAL = ["contract_encoded", "geo_prob_encoded", "funder_encoded"]
NON_ORDINAL = [c for c in NON_ORDINAL if c in df.columns]

fig, axes = plt.subplots(1, len(NON_ORDINAL), figsize=(len(NON_ORDINAL) * 6, 6))
if len(NON_ORDINAL) == 1:
    axes = [axes]

for ax, col in zip(axes, NON_ORDINAL):
    groups = {k: v["cost_overrun"].values
              for k, v in df.groupby(col)
              if len(v) > 1}   # need at least 2 samples per group

    if len(groups) < 2:
        ax.set_title(f"{col}\n(not enough groups)")
        continue

    # Kruskal-Wallis (non-parametric, no normality assumption — better for small n)
    stat, p = kruskal(*groups.values())

    # Box plot per group
    labels = [str(int(k)) for k in groups.keys()]
    bp = ax.boxplot(list(groups.values()), labels=labels,
                    patch_artist=True,
                    medianprops=dict(color="tomato", linewidth=2))
    colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title(f"{col}\nKruskal-Wallis p={p:.3f}"
                 + (" ✓ significant" if p < 0.05 else " ✗ not significant"),
                 fontsize=9,
                 color="green" if p < 0.05 else "black")
    ax.set_xlabel("Group code")
    ax.set_yscale("symlog")
    ax.set_ylabel("cost_overrun (%)")
    ax.tick_params(axis="x", labelsize=7, rotation=45)

plt.suptitle("Group Difference Tests (Kruskal-Wallis)\nfor Non-Ordinal Encoded Features",
             fontsize=12)
plt.tight_layout()
plt.savefig("anova_nonordinal.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: anova_nonordinal.png")