"""
create_visuals.py

Generates figures for the Internal Latent Space Analysis section.

Usage:
    python create_visuals.py

Expects:
    ./results_anli_roberta/audit_results.json

Outputs (./results_anli_roberta/figures/):
    las_distribution.png   — LAS histogram with UDR threshold marked
    csi_distribution.png   — CSI histogram separating signal from geometric noise
"""

import json
import os

import matplotlib
matplotlib.use("Agg")   # headless-safe; remove if running interactively
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results_anli_roberta"
AUDIT_PATH  = os.path.join(RESULTS_DIR, "audit_results.json")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})
PRIMARY   = "#2C5F8A"
SECONDARY = "#E07B39"
GREY      = "#888888"

# ── load data ─────────────────────────────────────────────────────────────────
with open(AUDIT_PATH) as f:
    audit = json.load(f)

las_scores      = audit["las_scores"]
csi_scores_all  = audit["csi_scores_all"]
csi_valid_flags = audit["csi_valid_flags"]
valid_csi       = [s for s, v in zip(csi_scores_all, csi_valid_flags) if v]


# ── 1. LAS distribution ───────────────────────────────────────────────────────
def plot_las_distribution():
    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    arr = np.array(las_scores)
    ax.hist(arr, bins=20, color=PRIMARY, edgecolor="white", linewidth=0.6, alpha=0.9)
    ax.axvline(0.30, color=SECONDARY, linewidth=1.8, linestyle="--",
               label="UDR threshold (0.30)")
    ax.axvline(arr.mean(), color=GREY, linewidth=1.4, linestyle=":",
               label=f"Mean LAS ({arr.mean():.3f})")

    ax.set_xlabel("Latent Alignment Score (LAS)")
    ax.set_ylabel("Sample Count")
    ax.set_title("Distribution of Latent Alignment Scores\n"
                 "(decision state vs. human rationale, n=500)")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "las_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── 2. CSI distribution (valid attacks only) ──────────────────────────────────
def plot_csi_distribution():
    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    arr = np.array(valid_csi)
    n   = len(arr)

    counts, edges = np.histogram(arr, bins=20)
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        color = SECONDARY if lo < 0 else PRIMARY
        ax.bar(lo, c, width=(hi - lo), color=color, edgecolor="white",
               linewidth=0.5, alpha=0.88, align="edge")

    ax.axvline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.35)
    ax.axvline(arr.mean(),     color=GREY, linewidth=1.4, linestyle=":")
    ax.axvline(np.median(arr), color=GREY, linewidth=1.4, linestyle="--")

    ax.legend(
        handles=[
            Patch(facecolor=PRIMARY,   label="Positive CSI (latent delta)"),
            Patch(facecolor=SECONDARY, label="Negative CSI (geometric noise)"),
            plt.Line2D([0], [0], color=GREY, linestyle=":",
                       label=f"Mean ({arr.mean():.3f})"),
            plt.Line2D([0], [0], color=GREY, linestyle="--",
                       label=f"Median ({np.median(arr):.3f})"),
        ],
        frameon=False, fontsize=8.5, loc="upper right",
    )

    ax.set_xlabel("Causal Sensitivity Index (CSI)")
    ax.set_ylabel("Sample Count")
    ax.set_title(f"CSI Distribution — Valid TextFooler Attacks (n={n})\n"
                 f"{(arr < 0).mean():.1%} negative "
                 f"(perturbation incidentally raised alignment)")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "csi_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_las_distribution()
    plot_csi_distribution()
    print(f"\nAll figures saved to {FIG_DIR}/")
