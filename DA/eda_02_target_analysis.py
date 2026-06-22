"""
EDA Script 2 — Target Variable Analysis (cost_overrun)
Outputs: target_histogram.png, target_boxplot.png, target_outliers.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("DA\hydropower_ml_ready.csv")
target = df["cost_overrun"].dropna()

# ── 1. Histogram + KDE ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw
axes[0].hist(target, bins=20, color="steelblue",
             edgecolor="white", density=True, alpha=0.7, label="histogram")
xmin, xmax = axes[0].get_xlim()
x = np.linspace(xmin, xmax, 200)
kde = stats.gaussian_kde(target)
axes[0].plot(x, kde(x), color="tomato", linewidth=2, label="KDE")
axes[0].axvline(target.mean(),   color="green",  linestyle="--", label=f"mean={target.mean():.1f}")
axes[0].axvline(target.median(), color="orange", linestyle="--", label=f"median={target.median():.1f}")
axes[0].set_title("cost_overrun — Raw Distribution")
axes[0].set_xlabel("cost_overrun (%)")
axes[0].legend(fontsize=8)

# Log-transformed (handles skew)
log_target = np.log1p(target.clip(lower=0))
axes[1].hist(log_target, bins=20, color="teal",
             edgecolor="white", density=True, alpha=0.7, label="histogram")
kde2 = stats.gaussian_kde(log_target)
x2 = np.linspace(log_target.min(), log_target.max(), 200)
axes[1].plot(x2, kde2(x2), color="tomato", linewidth=2, label="KDE")
axes[1].set_title("cost_overrun — Log(1+x) Transformed")
axes[1].set_xlabel("log(1 + cost_overrun)")
axes[1].legend(fontsize=8)

plt.suptitle("Target Variable Distribution", fontsize=13)
plt.tight_layout()
plt.savefig("target_histogram.png", dpi=150)
plt.close()

# ── 2. Box plot + strip plot ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(target, vert=True, patch_artist=True,
           boxprops=dict(facecolor="steelblue", alpha=0.5),
           medianprops=dict(color="tomato", linewidth=2))
# overlay individual points
ax.scatter(np.ones(len(target)) + np.random.uniform(-0.05, 0.05, len(target)),
           target, alpha=0.6, color="navy", s=30, zorder=3)
ax.set_xticks([1])
ax.set_xticklabels(["cost_overrun"])
ax.set_ylabel("cost_overrun (%)")
ax.set_title("Box Plot — cost_overrun\n(each dot = one project)")
plt.tight_layout()
plt.savefig("target_boxplot.png", dpi=150)
plt.close()

# ── 3. Outlier identification (IQR method) ───────────────────────────────────
Q1, Q3 = target.quantile(0.25), target.quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

# reload full df to get project names if available
df_full = pd.read_csv("DA\hydropower_ml_ready.csv")
outlier_mask = (df_full["cost_overrun"] < lower) | (df_full["cost_overrun"] > upper)
outliers = df_full[outlier_mask][["cost_overrun"]].copy()
outliers["z_score"] = np.abs(stats.zscore(df_full["cost_overrun"].fillna(df_full["cost_overrun"].median())))

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["tomato" if o else "steelblue" for o in outlier_mask]
ax.scatter(range(len(df_full)), df_full["cost_overrun"], c=colors, s=50, zorder=3)
ax.axhline(upper, color="tomato", linestyle="--", linewidth=1, label=f"IQR upper fence ({upper:.1f})")
ax.axhline(lower, color="orange", linestyle="--", linewidth=1, label=f"IQR lower fence ({lower:.1f})")
ax.set_xlabel("Project index")
ax.set_ylabel("cost_overrun (%)")
ax.set_title("Outlier Detection — IQR Method (red = outlier)")
ax.legend()
plt.tight_layout()
plt.savefig("target_outliers.png", dpi=150)
plt.close()

print(f"cost_overrun stats:\n{target.describe().round(2)}")
print(f"\nSkewness: {target.skew():.3f}  |  Kurtosis: {target.kurtosis():.3f}")
print(f"\nIQR bounds: [{lower:.1f}, {upper:.1f}]")
print(f"Outlier project indices:\n{outliers}")
print("\nSaved: target_histogram.png, target_boxplot.png, target_outliers.png")
