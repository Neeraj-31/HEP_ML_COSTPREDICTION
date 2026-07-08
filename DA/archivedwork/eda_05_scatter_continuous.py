"""
EDA Script 5 — Continuous Features vs cost_overrun (scatter plots)
Outputs: scatter_continuous.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("DA\hydropower_ml_ready.csv").dropna(subset=["cost_overrun"])

# Continuous features to plot (exclude encoded categoricals and target)
CONTINUOUS = [
    "installed_cap", "dam_height", "initial_cost", "rainfall",
    "forest_area_div", "dist_road", "tunnel length", "elevation",
    "damlength", "transmission line length cktkm", "landowner_displaced",
    "duration_overrun", "planned_dur", "actual_dur"
]
CONTINUOUS = [c for c in CONTINUOUS if c in df.columns]

target = df["cost_overrun"]
ncols = 3
nrows = (len(CONTINUOUS) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
axes = axes.flatten()

for i, col in enumerate(CONTINUOUS):
    x = df[col].dropna()
    common_idx = x.index.intersection(target.dropna().index)
    x_plot = x.loc[common_idx]
    y_plot = target.loc[common_idx]

    # Run the linear regression on log-transformed target for better stability
    slope, intercept, r, p, _ = stats.linregress(x_plot, np.log1p(y_plot))
    x_line = np.linspace(x_plot.min(), x_plot.max(), 200)

    axes[i].scatter(x_plot, y_plot, color="steelblue", s=45,
                    alpha=0.7, edgecolors="white", zorder=3)
    
    # Plot the converted exponential line to match the raw scale data
    axes[i].plot(x_line, np.expm1(slope * x_line + intercept), color="tomato", linewidth=1.5)
    
    axes[i].set_yscale("symlog") # <-- Crucial for keeping the plot viewable
    axes[i].set_xlabel(col, fontsize=8)
    axes[i].set_ylabel("cost_overrun", fontsize=8)
    axes[i].set_title(f"{col}\nr={r:.2f}, p={p:.3f}", fontsize=9)
    axes[i].tick_params(labelsize=7)

    # shade p < 0.05 title green
    if p < 0.05:
        axes[i].set_facecolor("#e8f5e9")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Continuous Features vs cost_overrun\n(green background = p < 0.05)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("scatter_continuous.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: scatter_continuous.png")
print("(Green background subplots = statistically significant at p<0.05)")
