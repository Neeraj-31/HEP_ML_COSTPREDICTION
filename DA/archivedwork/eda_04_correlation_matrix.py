import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

df = pd.read_csv(r"DA\merged_dam_features.csv", encoding="cp1252")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
df_num = df[numeric_cols].dropna()

# ── 1. Full Pearson correlation heatmap ──────────────────────────────────────
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# ── 1. Full Pearson correlation heatmap ──────────────────────────────────────
# Pandas .corr() drops missing values pairwise automatically!
corr = df[numeric_cols].corr(method="pearson")

fig, ax = plt.subplots(figsize=(18, 15))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.3, annot_kws={"size": 6},
            ax=ax, square=True)
ax.set_title("Pearson Correlation Matrix (Pairwise Deletion)", fontsize=13)
ax.tick_params(axis="x", labelsize=7, rotation=45)
ax.tick_params(axis="y", labelsize=7)
plt.tight_layout()
plt.savefig("corr_full_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 2. Correlation with target bar chart ─────────────────────────────────────
target = "cost_overrun"
features = [c for c in numeric_cols if c != target]

pearson_corrs = []
spearman_corrs = []

for f in features:
    # Compute correlation column by column, dropping NaNs only for those two columns
    valid_data = df[[f, target]].dropna()
    if len(valid_data) > 1:
        p_r = valid_data[f].corr(valid_data[target], method="pearson")
        s_r = valid_data[f].corr(valid_data[target], method="spearman")
    else:
        p_r, s_r = 0, 0
    pearson_corrs.append(p_r)
    spearman_corrs.append(s_r)
corr_df = pd.DataFrame({
    "feature":  features,
    "pearson":  pearson_corrs,
    "spearman": spearman_corrs,
}).set_index("feature").sort_values("spearman", ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(features) * 0.35)))

for ax, col, color, title in [
    (axes[0], "pearson",  "steelblue", "Pearson r"),
    (axes[1], "spearman", "teal",      "Spearman rho"),
]:
    bars = ax.barh(corr_df.index, corr_df[col], color=[
        "tomato" if v > 0 else "steelblue" for v in corr_df[col]
    ], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{title} with cost_overrun")
    ax.set_xlabel("Correlation coefficient")
    ax.tick_params(labelsize=8)
    # annotate values
    for bar, val in zip(bars, corr_df[col]):
        ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=7)

plt.suptitle("Feature Correlations with cost_overrun", fontsize=13)
plt.tight_layout()
plt.savefig("corr_with_target.png", dpi=150, bbox_inches="tight")
plt.close()

print("Top 10 Spearman correlations with cost_overrun:")
print(corr_df["spearman"].sort_values(ascending=False).head(10).round(3))
print("\nSaved: corr_full_heatmap.png, corr_with_target.png")
