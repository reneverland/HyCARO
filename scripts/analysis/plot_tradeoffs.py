"""Two-dimensional ContextBench trade-off plots (replaces the 3-D Pareto figure).

Values are the ContextBench summaries reported in the complete comparison table
(line F1, mean packed context tokens, p50 selector latency) at an 8K cap.
No frontier is drawn: selector timings were not all recorded under a verified
common boundary, so panel (b) is descriptive only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# name: (group, line F1, context tokens, p50 latency ms)
METHODS = {
    "BM25": ("retrieval", 0.081, 6776, 14.9),
    "Dense": ("retrieval", 0.095, 6755, 11.4),
    "RRF": ("retrieval", 0.095, 6759, 11.3),
    "MMR": ("retrieval", 0.090, 6756, 18.2),
    "Graph": ("retrieval", 0.097, 6180, 11.6),
    "Qwen-rerank": ("retrieval", 0.096, 6742, 1173.1),
    "Greedy+1-swap": ("optimization", 0.097, 3215, 187.7),
    "Two-unit greedy": ("optimization", 0.108, 3319, 633.7),
    "DE+repair": ("optimization", 0.103, 3277, 1356.9),
    "HyCARO": ("ours", 0.109, 4745, 40.1),
}

GROUP_STYLE = {
    "retrieval": dict(color="#7f7f7f", marker="o", label="Retrieval baselines"),
    "optimization": dict(color="#d9822b", marker="s", label="Optimization controls"),
    "ours": dict(color="#1f5fa8", marker="*", label="HyCARO"),
}

# Dense and RRF nearly coincide in both panels, so they share one label.
LABELS = {"Dense": "Dense / RRF", "RRF": None}

OFFSETS_TOKENS = {
    "BM25": (4, -8),
    "Dense": (4, -9),
    "MMR": (4, -2),
    "Graph": (-12, 5),
    "Qwen-rerank": (4, 3),
    "Greedy+1-swap": (4, -9),
    "Two-unit greedy": (-14, 6),
    "DE+repair": (4, -3),
    "HyCARO": (7, -2),
}

OFFSETS_LATENCY = {
    "BM25": (4, -8),
    "Dense": (4, -9),
    "MMR": (4, -2),
    "Graph": (4, 3),
    "Qwen-rerank": (-30, 5),
    "Greedy+1-swap": (4, -9),
    "Two-unit greedy": (-22, 6),
    "DE+repair": (-24, -9),
    "HyCARO": (7, -2),
}


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, idx, offsets, xlabel, title in [
        (axes[0], 2, OFFSETS_TOKENS, "Mean context tokens", "(a) Line F1 vs. context tokens"),
        (axes[1], 3, OFFSETS_LATENCY, "p50 selector latency (ms, log scale)", "(b) Line F1 vs. selector latency"),
    ]:
        seen = set()
        for name, row in METHODS.items():
            group, f1 = row[0], row[1]
            xval = row[idx]
            st = GROUP_STYLE[group]
            ax.scatter(
                xval,
                f1,
                color=st["color"],
                marker=st["marker"],
                s=70 if group == "ours" else 22,
                zorder=3,
                label=st["label"] if group not in seen else None,
                edgecolors="black" if group == "ours" else "none",
                linewidths=0.4,
            )
            seen.add(group)
            label = LABELS.get(name, name)
            if label:
                ax.annotate(label, (xval, f1), textcoords="offset points", xytext=offsets[name], fontsize=6.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ContextBench line F1")
        ax.set_title(title, fontsize=8)
        ax.grid(linewidth=0.4, alpha=0.5)
        ax.set_ylim(0.078, 0.112)
    axes[0].set_xlim(2900, 7300)
    axes[1].set_xscale("log")
    axes[1].set_xlim(7, 3000)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = Path(__file__).resolve().parents[2] / "figures" / "tradeoff2d.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
