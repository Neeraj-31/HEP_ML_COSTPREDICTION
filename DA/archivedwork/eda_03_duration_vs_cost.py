"""
EDA Script 3 — Duration Overrun vs Cost Overrun (your key question)
Outputs: dur_vs_cost_scatter.png, dur_vs_cost_bygroup.png, dur_vs_cost_corr.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("DA\hydropower_ml_ready.csv").dropna(subset=["cost_overrun", "duration_overrun"])

x = df["duration_overrun"]
y = df["cost_overrun"]

# ── 1. Scatter + regression line ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw scale
slope, intercept, r, p, se = stats.linregress(x, y)
x_line = np.linspace(x.min(), x.max(), 200)
axes[0].scatter(x, y, color="steelblue", s=60, alpha=0.7, edgecolors="white", zorder=3)
axes[0].plot(x_line, slope * x_line + intercept, color="tomato",
             linewidth=2, label=f"r={r:.3f}, p={p:.3f}")
axes[0].set_xlabel("duration_overrun (actual/planned)")
axes[0].set_ylabel("cost_overrun (%)")
axes[0].set_title("Duration Overrun vs Cost Overrun")
axes[0].legend()

# Log-log scale (better for skewed data)
log_x = np.log1p(x.clip(lower=0))
log_y = np.log1p(y.clip(lower=0))
slope2, intercept2, r2, p2, _ = stats.linregress(log_x, log_y)
x_line2 = np.linspace(log_x.min(), log_x.max(), 200)
axes[1].scatter(log_x, log_y, color="teal", s=60, alpha=0.7, edgecolors="white", zorder=3)
axes[1].plot(x_line2, slope2 * x_line2 + intercept2, color="tomato",
             linewidth=2, label=f"r={r2:.3f}, p={p2:.3f}")
axes[1].set_xlabel("log(1 + duration_overrun)")
axes[1].set_ylabel("log(1 + cost_overrun)")
axes[1].set_title("Log-Log Scale (cleaner for skewed data)")
axes[1].legend()

plt.suptitle("Does Duration Overrun Drive Cost Overrun?", fontsize=13)
plt.tight_layout()
plt.savefig("dur_vs_cost_scatter.png", dpi=150)
plt.close()

# ── 2. Scatter coloured by category and state ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# By category
cat_map = {1: "small", 2: "large", 3: "mega"}
palette = {1: "steelblue", 2: "orange", 3: "tomato"}
for cat_val, cat_name in cat_map.items():
    mask = df["category_encoded"] == cat_val
    axes[0].scatter(x[mask], y[mask], label=cat_name,
                    color=palette[cat_val], s=60, alpha=0.8, edgecolors="white")
axes[0].set_xlabel("duration_overrun")
axes[0].set_ylabel("cost_overrun (%)")
axes[0].set_yscale("symlog")
axes[0].set_title("By Project Category")
axes[0].legend()

# By state
for state_val, state_name, color in [(0, "Himachal Pradesh", "steelblue"),
                                      (1, "Uttarakhand", "tomato")]:
    mask = df["state_encoded"] == state_val
    axes[1].scatter(x[mask], y[mask], label=state_name,
                    color=color, s=60, alpha=0.8, edgecolors="white")
axes[1].set_xlabel("duration_overrun")
axes[1].set_ylabel("cost_overrun (%)")
axes[1].set_title("By State")
axes[1].set_yscale("symlog")
axes[1].legend()

plt.suptitle("Duration vs Cost Overrun — Subgroup Breakdown", fontsize=13)
plt.tight_layout()
plt.savefig("dur_vs_cost_bygroup.png", dpi=150)
plt.close()

# ── 3. Pearson vs Spearman comparison bar chart ───────────────────────────────
pearson_r,  pearson_p  = stats.pearsonr(x, y)
spearman_r, spearman_p = stats.spearmanr(x, y)

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(["Pearson r", "Spearman rho"], [pearson_r, spearman_r],
              color=["steelblue", "teal"], edgecolor="white", width=0.4)
for bar, p_val in zip(bars, [pearson_p, spearman_p]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"p={p_val:.3f}", ha="center", va="bottom", fontsize=10)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylim(-0.1, 1.0)
ax.set_ylabel("Correlation coefficient")
ax.set_title("duration_overrun ↔ cost_overrun\nPearson vs Spearman")
plt.tight_layout()
plt.savefig("dur_vs_cost_corr.png", dpi=150)
plt.close()

print(f"Pearson  r={pearson_r:.3f}  p={pearson_p:.4f}")
print(f"Spearman r={spearman_r:.3f}  p={spearman_p:.4f}")
print("\nSaved: dur_vs_cost_scatter.png, dur_vs_cost_bygroup.png, dur_vs_cost_corr.png")
