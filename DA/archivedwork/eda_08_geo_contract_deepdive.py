"""
EDA Script 8 — Geological Problem & Contract Type Deep Dive
Outputs: geo_prob_analysis.png, contract_analysis.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("DA\hydropower_ml_ready.csv").dropna(subset=["cost_overrun"])

# ── 1. Geo prob encoded vs cost overrun ──────────────────────────────────────
# ── 1. Geo prob encoded vs cost overrun (Dynamic Grid) ───────────────────────
geo_cols_to_plot = []
if "geo_prob_encoded" in df.columns: geo_cols_to_plot.append("geo_prob_encoded")
if "has_geo_prob" in df.columns:     geo_cols_to_plot.append("has_geo_prob")
if "prob_count" in df.columns:       geo_cols_to_plot.append("prob_count")

if geo_cols_to_plot:
    fig, axes = plt.subplots(1, len(geo_cols_to_plot), figsize=(len(geo_cols_to_plot) * 5, 5))
    if len(geo_cols_to_plot) == 1:
        axes = [axes]
        
    ax_idx = 0
    
    # 1. Handle geo_prob_encoded as a string categorical/discrete factor
    if "geo_prob_encoded" in df.columns:
        ax = axes[ax_idx]
        # Treat as string/categorical so it doesn't map continuously on an integer axis
        df_geo = df.dropna(subset=["geo_prob_encoded", "cost_overrun"]).copy()
        df_geo["geo_prob_str"] = df_geo["geo_prob_encoded"].astype(int).astype(str)
        
        sns.boxplot(x="geo_prob_str", y="cost_overrun", data=df_geo, ax=ax, palette="Blues")
        ax.set_yscale("symlog")
        ax.set_xlabel("Geo Problem Bitmask Code (Categorical)")
        ax.set_ylabel("cost_overrun (%)")
        ax.set_title("Geo Code vs Cost Overrun")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax_idx += 1

    # 2. Violin Plot for binary flag
    if "has_geo_prob" in df.columns:
        ax = axes[ax_idx]
        sns.violinplot(x="has_geo_prob", y="cost_overrun", data=df,
                       palette=["steelblue", "tomato"], ax=ax, inner="box")
        ax.set_yscale("symlog")
        ax.set_xticklabels(["No problem (0)", "Has problem (1)"])
        ax.set_title("No Problem vs Has Problem")
        ax.set_ylabel("cost_overrun (%)" if ax_idx == 0 else "")
        ax_idx += 1

    # 3. Problem count trend lines
    if "prob_count" in df.columns:
        ax = axes[ax_idx]
        ax.scatter(df["prob_count"], df["cost_overrun"], color="teal", s=50, alpha=0.7, edgecolors="white")
        mean_per_count = df.groupby("prob_count")["cost_overrun"].mean()
        ax.plot(mean_per_count.index, mean_per_count.values, "o-", color="tomato", linewidth=2, markersize=8, label="group mean")
        ax.set_yscale("symlog")
        ax.set_xlabel("Number of distinct problems")
        ax.set_ylabel("cost_overrun (%)" if ax_idx == 0 else "")
        ax.set_title("Problem Count vs Cost Overrun")
        ax.legend()
        ax_idx += 1

    plt.suptitle("Geological / Social Problem Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig("geo_prob_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

# ── 2. Contract type vs cost overrun ─────────────────────────────────────────
if "contract_encoded" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot per contract type
    groups = df.groupby("contract_encoded")["cost_overrun"]
    labels = [str(int(k)) for k in groups.groups.keys()]
    data   = [v.values for v in groups]

    bp = axes[0].boxplot(data, labels=labels, patch_artist=True,
                         medianprops=dict(color="tomato", linewidth=2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_xlabel("contract_encoded")
    axes[0].set_ylabel("cost_overrun (%)")
    axes[0].set_title("Cost Overrun by Contract Type Code")

    # Mean cost overrun per contract type
    mean_cost = df.groupby("contract_encoded")["cost_overrun"].mean().sort_values()
    axes[1].barh([str(int(k)) for k in mean_cost.index], mean_cost.values,
                 color="steelblue", edgecolor="white")
    axes[1].axvline(df["cost_overrun"].mean(), color="tomato",
                    linestyle="--", label=f"overall mean={df['cost_overrun'].mean():.1f}")
    axes[1].set_xlabel("mean cost_overrun (%)")
    axes[1].set_ylabel("contract_encoded")
    axes[1].set_title("Mean Cost Overrun by Contract Type")
    axes[1].legend(fontsize=8)

    # Code legend annotation
    legend_text = "Codes: 1=EPC  2=IR  3=SPLIT  4=BILATERAL\n5=BOOT  6=MP  7=DEPT  8=TURNKEY\nCombinations = concatenated (e.g. 12=EPC+IR)"
    fig.text(0.5, -0.05, legend_text, ha="center", fontsize=8,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.suptitle("Contract Type vs Cost Overrun", fontsize=13)
    plt.tight_layout()
    plt.savefig("contract_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

print("Saved: geo_prob_analysis.png, contract_analysis.png")
