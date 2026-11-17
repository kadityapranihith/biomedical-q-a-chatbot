"""
compare_retrievals.py
=====================
Reads all retrieval evaluation result JSON files, builds a comparison
CSV, and produces publication-quality plots.

Methods compared
----------------
1. BM25
2. BioBERT
3. Hybrid BM25+BioBERT (alpha=0.5)
4. Hybrid BM25+BioBERT (tuned alpha)
5. MedCPT
6. BM25 + MedCPT Hybrid

Usage
-----
    python -m evaluation.compare_retrievals
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent
EVAL_DIR   = BASE_DIR / "evaluation"
OUTPUT_CSV = EVAL_DIR / "retrieval_comparison.csv"
PLOTS_DIR  = EVAL_DIR / "comparison_plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PALETTE  (colourblind-friendly + modern)
# ============================================================

COLORS = [
    "#4361EE",   # BM25               - vivid blue
    "#F72585",   # BioBERT            - hot pink
    "#7209B7",   # Hybrid BM25+BioBERT- purple
    "#3A0CA3",   # Hybrid Tuned       - deep indigo
    "#4CC9F0",   # MedCPT             - sky cyan
    "#F4A261",   # BM25+MedCPT Hybrid - warm orange
]

# ============================================================
# METHOD DEFINITIONS
# ============================================================

METHODS = [
    {
        "file":        "bm25_results.json",
        "label":       "BM25",
        "recall1_key": "recall@1",
        "recall5_key": "recall@5",
        "recall10_key":"recall@10",
        "mrr_key":     "mrr",
        "lat_key":     "average_latency_ms",
    },
    {
        "file":        "biobert_results.json",
        "label":       "BioBERT",
        "recall1_key": "recall_at_1",
        "recall5_key": "recall_at_5",
        "recall10_key":"recall_at_10",
        "mrr_key":     "mrr",
        "lat_key":     "average_latency_ms",
    },
    {
        "file":        "hybrid_results.json",
        "label":       "Hybrid BM25+BioBERT (a=0.5)",
        "recall1_key": "recall@1",
        "recall5_key": "recall@5",
        "recall10_key":"recall@10",
        "mrr_key":     "mrr",
        "lat_key":     "average_latency_ms",
    },
    {
        "file":        "hybrid_tuned_results.json",
        "label":       "Hybrid BM25+BioBERT (Tuned)",
        "recall1_key": "recall@1",
        "recall5_key": "recall@5",
        "recall10_key":"recall@10",
        "mrr_key":     "mrr",
        "lat_key":     "average_latency_ms",
    },
    {
        "file":        "medcpt_results.json",
        "label":       "MedCPT",
        "recall1_key": "recall_at_1",
        "recall5_key": "recall_at_5",
        "recall10_key":"recall_at_10",
        "mrr_key":     "mrr",
        "lat_key":     "average_latency_ms",
    },
    {
        "file":        "bm25_medcpt_results.json",
        "label":       "BM25+MedCPT Hybrid",
        "recall1_key": "recall@1",
        "recall5_key": "recall@5",
        "recall10_key":"recall@10",
        "mrr_key":     "mrr",
        "lat_key":     "average_latency_ms",
    },
]

SHORT_LABELS = [
    "BM25",
    "BioBERT",
    "Hybrid\nBM25+BioBERT\n(a=0.5)",
    "Hybrid\nBM25+BioBERT\n(Tuned)",
    "MedCPT",
    "BM25+MedCPT\nHybrid",
]

# ============================================================
# LOAD DATA
# ============================================================

rows = []

for i, m in enumerate(METHODS):
    path = EVAL_DIR / m["file"]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    metrics = data["metrics"]

    r1  = metrics.get(m["recall1_key"],  0.0)
    r5  = metrics.get(m["recall5_key"],  0.0)
    r10 = metrics.get(m["recall10_key"], 0.0)
    mrr = metrics.get(m["mrr_key"],      0.0)
    lat = metrics.get(m["lat_key"],      0.0)

    row = {
        "Method":           m["label"],
        "File":             m["file"],
        "Corpus Size":      data.get("corpus_size", ""),
        "Test Size":        data.get("test_size",   ""),
        "Recall@1":         round(r1,  4),
        "Recall@5":         round(r5,  4),
        "Recall@10":        round(r10, 4),
        "MRR":              round(mrr, 4),
        "Avg Latency (ms)": round(lat, 2),
    }

    rows.append(row)

df = pd.DataFrame(rows)

# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(OUTPUT_CSV, index=False)
print(f"CSV saved to: {OUTPUT_CSV}")
print()
print(df.to_string(index=False))

# ============================================================
# HELPER
# ============================================================

def _style(ax, title, ylabel, ylim=(0, 1.05)):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="grey")
    ax.set_axisbelow(True)

# ============================================================
# FIGURE 1 - Recall@K grouped bar chart
# ============================================================

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor("#F8F9FD")
ax.set_facecolor("#F8F9FD")

n   = len(rows)
x   = np.arange(n)
w   = 0.25

r1_vals  = df["Recall@1"].tolist()
r5_vals  = df["Recall@5"].tolist()
r10_vals = df["Recall@10"].tolist()

bars1  = ax.bar(x - w, r1_vals,  width=w, label="Recall@1",
                color="#4361EE", edgecolor="white", linewidth=0.8, zorder=3)
bars5  = ax.bar(x,     r5_vals,  width=w, label="Recall@5",
                color="#4CC9F0", edgecolor="white", linewidth=0.8, zorder=3)
bars10 = ax.bar(x + w, r10_vals, width=w, label="Recall@10",
                color="#F4A261", edgecolor="white", linewidth=0.8, zorder=3)

for bars in [bars1, bars5, bars10]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom",
                fontsize=7, fontweight="bold", color="#333")

ax.set_xticks(x)
ax.set_xticklabels(SHORT_LABELS, fontsize=8.5)
ax.set_ylabel("Recall", fontsize=11)
ax.set_title("Recall@K Comparison Across Retrieval Methods",
             fontsize=14, fontweight="bold", pad=10)
ax.set_ylim(0, 1.15)
ax.legend(loc="lower right", fontsize=10, framealpha=0.7)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4, color="grey")
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "recall_at_k.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved recall_at_k.png")

# ============================================================
# FIGURE 2 - MRR bar chart
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#F8F9FD")
ax.set_facecolor("#F8F9FD")

mrr_vals = df["MRR"].tolist()
x_mrr = np.arange(n)
bars = ax.bar(x_mrr, mrr_vals, color=COLORS, width=0.55,
              edgecolor="white", linewidth=0.8, zorder=3)
ax.set_xticks(x_mrr)
ax.set_xticklabels(SHORT_LABELS, fontsize=8.5)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold")
_style(ax, "Mean Reciprocal Rank (MRR) by Retrieval Method", "MRR")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "mrr_comparison.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved mrr_comparison.png")

# ============================================================
# FIGURE 3 - Latency bar chart
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#F8F9FD")
ax.set_facecolor("#F8F9FD")

lat_vals = df["Avg Latency (ms)"].tolist()
x_lat = np.arange(n)
bars = ax.bar(x_lat, lat_vals, color=COLORS, width=0.55,
              edgecolor="white", linewidth=0.8, zorder=3)
ax.set_xticks(x_lat)
ax.set_xticklabels(SHORT_LABELS, fontsize=8.5)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4,
            f"{bar.get_height():.1f} ms",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold")
_style(ax,
       "Average Query Latency by Retrieval Method",
       "Avg Latency (ms)",
       ylim=(0, max(lat_vals) * 1.25))
plt.tight_layout()
plt.savefig(PLOTS_DIR / "latency_comparison.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved latency_comparison.png")

# ============================================================
# FIGURE 4 - Radar / spider chart
# ============================================================

categories = ["Recall@1", "Recall@5", "Recall@10", "MRR", "Speed\n(inv-lat)"]
n_cat = len(categories)
angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
angles += angles[:1]

max_lat = max(df["Avg Latency (ms)"].tolist())
speed_vals = [1 - (v / max_lat) for v in df["Avg Latency (ms)"].tolist()]

fig, ax = plt.subplots(figsize=(8, 8),
                       subplot_kw=dict(polar=True))
fig.patch.set_facecolor("#F8F9FD")
ax.set_facecolor("#F8F9FD")

for i, row in df.iterrows():
    values = [
        row["Recall@1"],
        row["Recall@5"],
        row["Recall@10"],
        row["MRR"],
        speed_vals[i],
    ]
    values += values[:1]
    ax.plot(angles, values, color=COLORS[i], linewidth=2,
            label=SHORT_LABELS[i].replace("\n", " "))
    ax.fill(angles, values, color=COLORS[i], alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8)
ax.spines["polar"].set_visible(False)
ax.grid(color="grey", linestyle="--", alpha=0.4)
ax.set_title("Retrieval Methods - Multi-Metric Radar",
             fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.15),
          fontsize=8.5, framealpha=0.7)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "radar_chart.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved radar_chart.png")

# ============================================================
# FIGURE 5 - MRR vs Latency scatter (trade-off)
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor("#F8F9FD")
ax.set_facecolor("#F8F9FD")

for i, row in df.iterrows():
    ax.scatter(row["Avg Latency (ms)"], row["MRR"],
               color=COLORS[i], s=180, zorder=5,
               edgecolors="white", linewidths=1.2)
    ax.annotate(
        SHORT_LABELS[i].replace("\n", " "),
        xy=(row["Avg Latency (ms)"], row["MRR"]),
        xytext=(8, 4),
        textcoords="offset points",
        fontsize=8.5, fontweight="bold", color=COLORS[i],
    )

ax.set_xlabel("Avg Latency (ms)", fontsize=11)
ax.set_ylabel("MRR", fontsize=11)
ax.set_title("Quality vs. Speed Trade-off  (MRR vs Average Latency)",
             fontsize=14, fontweight="bold", pad=10)
ax.set_ylim(0, 1.05)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(linestyle="--", alpha=0.4, color="grey")
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "mrr_vs_latency.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved mrr_vs_latency.png")

# ============================================================
# FIGURE 6 - Heatmap of all metrics
# ============================================================

metric_cols = ["Recall@1", "Recall@5", "Recall@10", "MRR"]
heat_data   = df[metric_cols].values
heat_labels = [s.replace("\n", " ") for s in SHORT_LABELS]

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#F8F9FD")
ax.set_facecolor("#F8F9FD")

im = ax.imshow(heat_data, cmap="YlGnBu", aspect="auto",
               vmin=0.0, vmax=1.0)

ax.set_xticks(range(len(metric_cols)))
ax.set_xticklabels(metric_cols, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(heat_labels)))
ax.set_yticklabels(heat_labels, fontsize=9)

for r in range(len(heat_labels)):
    for c in range(len(metric_cols)):
        val = heat_data[r, c]
        color = "white" if val > 0.65 else "#222"
        ax.text(c, r, f"{val:.3f}",
                ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
cbar.set_label("Score", fontsize=10)
ax.set_title("Retrieval Metrics Heatmap",
             fontsize=14, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "metrics_heatmap.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved metrics_heatmap.png")

# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("All outputs saved:")
print(f"  CSV  -> {OUTPUT_CSV}")
print(f"  PNGs -> {PLOTS_DIR}")
print("=" * 60)
