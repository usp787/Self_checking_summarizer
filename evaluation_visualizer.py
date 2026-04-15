"""
evaluation_visualizer.py
Author: Yangyie
CS6140 Final Project — Self-Checking Summarizer

Generates comparison plots from the project's results JSON files.
Run from the repo root:
    python evaluation_visualizer.py

Output: evaluation_plots.png
"""

import json
import os
import glob
import re
import csv
import matplotlib.pyplot as plt
import numpy as np

# ── Load results ──────────────────────────────────────────────────────────────

RESULTS_DIR = "results"

def load_metrics(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path) as f:
        return json.load(f)

baseline = load_metrics("baseline_qwen25_7b_a100_100samples_faithful_metrics_summary.json")

def _parse_range_from_name(path):
    name = os.path.basename(path)
    m = re.search(r"_(\d+)to(\d+)_", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _select_non_overlapping_ranges(ranges):
    """
    Prefer granular slice files and skip broad overlapping aggregate ranges
    (e.g., keep 0-50, 50-100, ... and skip 0-400).
    """
    ordered = sorted(ranges, key=lambda x: (x[1] - x[0], x[0], x[1]))
    selected = []
    for r in ordered:
        overlaps = any(not (r[1] <= s[0] or r[0] >= s[1]) for s in selected)
        if not overlaps:
            selected.append(r)
    return sorted(selected)


def load_dc_method_from_files(prefix):
    """
    Build method metrics from real experiment files in results/.
    - ROUGE and lengths: aggregate per-sample CSVs, dedup by sample_id.
    - BERTScore: weighted average from matching slice metrics_summary JSONs.
    """
    per_sample_glob = os.path.join(RESULTS_DIR, f"{prefix}_*to*_per_sample_metrics.csv")
    per_sample_files = glob.glob(per_sample_glob)
    if not per_sample_files:
        raise FileNotFoundError(f"No per-sample files found for {prefix}")

    ranges = [r for r in (_parse_range_from_name(p) for p in per_sample_files) if r is not None]
    selected_ranges = _select_non_overlapping_ranges(ranges)
    selected_files = [
        os.path.join(RESULTS_DIR, f"{prefix}_{start}to{end}_per_sample_metrics.csv")
        for start, end in selected_ranges
        if os.path.exists(os.path.join(RESULTS_DIR, f"{prefix}_{start}to{end}_per_sample_metrics.csv"))
    ]

    # Deduplicate sample rows across files
    sample_rows = {}
    for file_path in selected_files:
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_rows[int(row["sample_id"])] = row

    if not sample_rows:
        raise RuntimeError(f"No sample rows loaded for {prefix}")

    rows = list(sample_rows.values())
    rouge1 = np.array([float(r["rouge1_f1"]) for r in rows])
    rouge2 = np.array([float(r["rouge2_f1"]) for r in rows])
    rougeL = np.array([float(r["rougeL_f1"]) for r in rows])
    gen_len = np.array([float(r["gen_length"]) for r in rows])

    # Weighted mean BERTScore from selected summary files
    bert_num = 0.0
    bert_den = 0.0
    for start, end in selected_ranges:
        summary_path = os.path.join(RESULTS_DIR, f"{prefix}_{start}to{end}_metrics_summary.json")
        if not os.path.exists(summary_path):
            continue
        summary = load_metrics(os.path.basename(summary_path))
        n = int(summary.get("num_samples", 0))
        bert = float(summary.get("bertscore_f1_mean", 0.0))
        bert_num += bert * n
        bert_den += n

    if bert_den == 0:
        raise RuntimeError(f"No metrics_summary files found for {prefix}")

    return {
        "rouge1_f1_mean": float(np.mean(rouge1)),
        "rouge2_f1_mean": float(np.mean(rouge2)),
        "rougeL_f1_mean": float(np.mean(rougeL)),
        "bertscore_f1_mean": float(bert_num / bert_den),
        "gen_length_mean": float(np.mean(gen_len)),
    }


dc_results = {
    "MapReduce": load_dc_method_from_files("mapreduce_qwen25_7b_hpc"),
    "Map-Refine": load_dc_method_from_files("refine_qwen25_7b_hpc"),
}

# ── Plot setup ────────────────────────────────────────────────────────────────

FIG_BG    = "#ffffff"
AX_BG     = "#ffffff"
ACCENT    = "#00a6fb"
GREEN     = "#38b000"
YELLOW    = "#ffbe0b"
PURPLE    = "#8338ec"
GRAY      = "#6b7280"
ORANGE    = "#fb5607"

plt.rcParams.update({
    "figure.facecolor":  FIG_BG,
    "axes.facecolor":    AX_BG,
    "axes.edgecolor":    "#d1d5db",
    "axes.labelcolor":   "#111827",
    "axes.titlecolor":   "#111827",
    "xtick.color":       "#374151",
    "ytick.color":       "#374151",
    "text.color":        "#111827",
    "font.family":       "DejaVu Sans",
    "grid.color":        "#e5e7eb",
    "grid.linewidth":    0.8,
})

methods   = ["Baseline"] + list(dc_results.keys())
colors    = [GRAY, ACCENT, GREEN]
ref_len   = baseline["ref_length_mean"]

# ── Helper: bar chart ─────────────────────────────────────────────────────────

def bar_chart(ax, metric_key, title, ylabel, highlight_direction="up"):
    """highlight_direction: 'up' if higher is better, 'down' if lower is better"""
    values = [baseline[metric_key]] + [dc_results[m][metric_key] for m in dc_results]
    bars = ax.bar(methods, values, color=colors, width=0.55, zorder=3)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9, color="white", fontweight="bold"
        )

    # Highlight best performer
    if highlight_direction == "up":
        best_idx = int(np.argmax(values))
    else:
        best_idx = int(np.argmin(values))

    bars[best_idx].set_edgecolor(YELLOW)
    bars[best_idx].set_linewidth(4)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, max(values) * 1.18)
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Baseline reference line
    ax.axhline(values[0], color=GRAY, linewidth=1, linestyle="--", alpha=0.5, zorder=2)


def draw_rouge_plot(ax):
    metric_groups = {
        "ROUGE-1": ("rouge1_f1_mean", "#ff7f11"),
        "ROUGE-2": ("rouge2_f1_mean", "#e63946"),
        "ROUGE-L": ("rougeL_f1_mean", "#00b4d8"),
    }

    x = np.arange(len(methods))
    width = 0.22

    for gi, (label, (key, color)) in enumerate(metric_groups.items()):
        vals = [baseline[key]] + [dc_results[m][key] for m in dc_results]
        offset = (gi - 1) * width
        rects = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.9, zorder=3)
        for rect, val in zip(rects, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.003,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#111827"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    label_bg = {
        "Baseline": "#6b7280",      # gray
        "MapReduce": "#2563eb",     # blue
        "Map-Refine": "#16a34a",    # green
    }
    for tick in ax.get_xticklabels():
        txt = tick.get_text()
        tick.set_color("white")
        tick.set_bbox({
            "facecolor": label_bg.get(txt, "#374151"),
            "edgecolor": "none",
            "boxstyle": "round,pad=0.22",
        })
    ax.set_title("ROUGE Scores", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("F1 Score", fontsize=10)
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#d1d5db")
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Grey baseline reference line for each ROUGE metric
    ax.axhline(baseline["rouge1_f1_mean"], color=GRAY, linewidth=1.0, linestyle="--", alpha=0.45)
    ax.axhline(baseline["rouge2_f1_mean"], color=GRAY, linewidth=1.0, linestyle=":", alpha=0.45)
    ax.axhline(baseline["rougeL_f1_mean"], color=GRAY, linewidth=1.0, linestyle="-.", alpha=0.45)


def draw_length_plot(ax):
    gen_lengths = [baseline["gen_length_mean"]] + [dc_results[m]["gen_length_mean"] for m in dc_results]
    coverage = [l / ref_len * 100 for l in gen_lengths]

    bars = ax.bar(methods, coverage, color=colors, width=0.55, zorder=3)
    best_idx = int(np.argmax(coverage))
    bars[best_idx].set_edgecolor(YELLOW)
    bars[best_idx].set_linewidth(4)

    ax.axhline(
        100,
        color=ORANGE,
        linewidth=1.6,
        linestyle="--",
        alpha=0.8,
        zorder=2,
        label=f"Reference (100%, {ref_len:.0f} words)",
    )

    for bar, cov, length in zip(bars, coverage, gen_lengths):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{cov:.1f}%\n({length:.0f}w)",
            ha="center", va="bottom", fontsize=9, color="#111827", fontweight="bold"
        )

    ax.set_title("Length Coverage", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("% of Reference Length", fontsize=10)
    ax.set_ylim(0, 120)
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#d1d5db")
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(
    "Self-Checking Summarizer — Baseline vs. Divide-and-Conquer Pipeline",
    fontsize=15, fontweight="bold", color="#111827", y=1.02
)

# ── Plot 1: ROUGE scores ──────────────────────────────────────────────────────
draw_rouge_plot(axes[0])

# ── Plot 2: BERTScore ─────────────────────────────────────────────────────────
bar_chart(axes[1], "bertscore_f1_mean", "BERTScore F1", "F1 Score", "up")

# ── Plot 3: Length coverage ───────────────────────────────────────────────────
draw_length_plot(axes[2])

# ── Caption ───────────────────────────────────────────────────────────────────

fig.text(
    0.5, -0.04,
    "Yellow border = best performer per metric  |  Dashed line = baseline reference  |  "
    "D&C improves ROUGE-1, BERTScore, and length coverage; ROUGE-2/L trade off for paraphrase flexibility",
    ha="center", fontsize=9, color="#4b5563", style="italic"
)

plt.tight_layout()
output_path = "evaluation_plots.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
print(f"Saved: {output_path}")

# Standalone plots (3 additional files)
fig_rouge, ax_rouge = plt.subplots(figsize=(8, 6))
draw_rouge_plot(ax_rouge)
fig_rouge.tight_layout()
rouge_out = "evaluation_plot_rouge.png"
fig_rouge.savefig(rouge_out, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
print(f"Saved: {rouge_out}")

fig_bert, ax_bert = plt.subplots(figsize=(8, 6))
bar_chart(ax_bert, "bertscore_f1_mean", "BERTScore F1", "F1 Score", "up")
fig_bert.tight_layout()
bert_out = "evaluation_plot_bertscore.png"
fig_bert.savefig(bert_out, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
print(f"Saved: {bert_out}")

fig_len, ax_len = plt.subplots(figsize=(8, 6))
draw_length_plot(ax_len)
fig_len.tight_layout()
len_out = "evaluation_plot_length_coverage.png"
fig_len.savefig(len_out, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
print(f"Saved: {len_out}")

plt.show()
