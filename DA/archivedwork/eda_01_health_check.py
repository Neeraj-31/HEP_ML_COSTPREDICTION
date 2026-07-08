"""
EDA Script 1 — Data Health Check
Outputs: health_missing.png, health_distributions.png, health_skewness.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("DA\hydropower_ml_ready.csv")

# ── 1. Missing values heatmap ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False,
            cmap="viridis", ax=ax)
ax.set_title("Missing Value Map (yellow = missing)", fontsize=13)
plt.tight_layout()
plt.savefig("health_missing.png", dpi=150)
plt.close()

# ── 2. Distribution of every numeric column ──────────────────────────────────
# ── 2. Distribution of every numeric column ──────────────────────────────────
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
n = len(numeric_cols)
ncols = 4
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    data_to_plot = df[col].dropna()
    col_skew = data_to_plot.skew()
    
    # If heavily right-skewed and contains zeros/positives, use log1p scale
    if col_skew > 1.5 and (data_to_plot >= 0).all():
        axes[i].hist(np.log1p(data_to_plot), bins=15, color="teal", edgecolor="white", linewidth=0.5)
        axes[i].set_title(f"{col}\n[Log(1+x) Transformed]", fontsize=8)
    else:
        axes[i].hist(data_to_plot, bins=15, color="steelblue", edgecolor="white", linewidth=0.5)
        axes[i].set_title(col, fontsize=8)
        
    axes[i].set_xlabel("")
    axes[i].tick_params(labelsize=7)

# hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Distribution of All Numeric Features (Log scales applied to high skew)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("health_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
# ── 3. Skewness bar chart ────────────────────────────────────────────────────
skew = df[numeric_cols].skew().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 5))
colors = ["tomato" if abs(s) > 1 else "steelblue" for s in skew]
ax.bar(skew.index, skew.values, color=colors, edgecolor="white")
ax.axhline(1,  color="tomato", linestyle="--", linewidth=0.8, label="|skew| > 1 (red)")
ax.axhline(-1, color="tomato", linestyle="--", linewidth=0.8)
ax.axhline(0,  color="black",  linestyle="-",  linewidth=0.5)
ax.set_xticklabels(skew.index, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Skewness")
ax.set_title("Feature Skewness (red = |skew| > 1, consider log transform)")
ax.legend()
plt.tight_layout()
plt.savefig("health_skewness.png", dpi=150)
plt.close()

# ── Console summary ──────────────────────────────────────────────────────────
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nSkewness (top 10):\n", skew.head(10))
print("\nBasic stats:\n", df.describe().round(2))
print("\nSaved: health_missing.png, health_distributions.png, health_skewness.png")
