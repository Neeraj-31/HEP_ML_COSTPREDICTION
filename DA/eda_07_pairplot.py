"""
EDA Script 7 — Pair Plot of Top Correlated Features
Outputs: pairplot_top_features.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

df = pd.read_csv("DA\hydropower_ml_ready.csv").dropna(subset=["cost_overrun"])

# Pick top N features by absolute Spearman correlation with cost_overrun
TARGET = "cost_overrun"
TOP_N  = 6

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
features = [c for c in numeric_cols if c != TARGET]

spearman_corrs = {
    f: abs(spearmanr(df[f].fillna(df[f].median()), df[TARGET])[0])
    for f in features
}
top_features = sorted(spearman_corrs, key=spearman_corrs.get, reverse=True)[:TOP_N]
plot_cols = top_features + [TARGET]

print(f"Top {TOP_N} features by |Spearman| with {TARGET}:")
for f in top_features:
    print(f"  {f:<45} {spearman_corrs[f]:.3f}")

# Replace the plotting block at the bottom with this:
plot_df = df[plot_cols].dropna().copy()

# Log transform the target column inside the plotting dataframe to distribute points evenly
plot_df[TARGET] = np.log1p(plot_df[TARGET])

hue_col = "category_encoded" if "category_encoded" in plot_df.columns else None
if hue_col:
    cat_map = {1: "small", 2: "large", 3: "mega"}
    plot_df["category"] = plot_df[hue_col].map(cat_map)
    hue_col = "category"
    plot_cols_final = [c for c in plot_cols if c != "category_encoded"]
else:
    plot_cols_final = plot_cols

# Generate clean pairplot
g = sns.pairplot(plot_df, vars=plot_cols_final, hue=hue_col, palette="Set2", 
                 diag_kind="kde", plot_kws={'alpha': 0.6, 'edgecolor': 'w', 'linewidth': 0.5})

# Relabel the target axis to remind you it's log-scale
for ax in g.axes.flatten():
    if ax is not None:
        if ax.get_xlabel() == TARGET: ax.set_xlabel(f"{TARGET} (Log Scale)")
        if ax.get_ylabel() == TARGET: ax.set_ylabel(f"{TARGET} (Log Scale)")

plt.savefig("pairplot_top_features.png", dpi=150)
plt.close()