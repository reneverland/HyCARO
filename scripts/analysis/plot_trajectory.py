"""Redraw the search-trajectory figure (surrogate fitness and line F1 per round).

The values are the recorded per-round means of the trajectory run plotted in the
original ``figures/surrogate_trajectory.pdf``. They were recovered from that
PDF's vector paths using its tick positions and agree with the text to three
decimals. This trajectory run predates the final selector revision, so its last
Full-HyCARO F1 (about 0.104) differs from the main-table value (0.109).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROUNDS = ["Init", "1", "2", "3"]

FITNESS = {
    "Seed-only": [0.6808, 0.6808, 0.6808, 0.6808],
    "Full HyCARO": [0.6808, 0.7130, 0.7230, 0.7284],
    "Pairwise proposals": [0.6808, 0.7123, 0.7223, 0.7274],
}

LINE_F1 = {
    "Seed-only": [0.0876, 0.0876, 0.0876, 0.0876],
    "Full HyCARO": [0.0876, 0.1005, 0.0998, 0.1040],
    "Pairwise proposals": [0.0876, 0.0987, 0.1004, 0.1017],
}

STYLE = {
    "Seed-only": dict(color="#7f7f7f", linestyle="--", marker=None),
    "Full HyCARO": dict(color="#1f5fa8", linestyle="-", marker="o"),
    "Pairwise proposals": dict(color="#d9822b", linestyle="-", marker="s"),
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
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    x = range(len(ROUNDS))
    panels = [
        (axes[0], FITNESS, "Surrogate score $F$", "(a) Surrogate fitness"),
        (axes[1], LINE_F1, "Line F1", "(b) ContextBench line F1"),
    ]
    for ax, data, ylabel, title in panels:
        for name, values in data.items():
            st = STYLE[name]
            ax.plot(
                x,
                values,
                label=name,
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                markersize=4,
                linewidth=1.4,
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels(ROUNDS)
        ax.set_xlabel("Search round")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    axes[1].annotate(
        f"{LINE_F1['Full HyCARO'][-1]:.3f}",
        (3, LINE_F1["Full HyCARO"][-1]),
        textcoords="offset points",
        xytext=(-4, 5),
        ha="right",
        fontsize=7,
        color=STYLE["Full HyCARO"]["color"],
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = Path(__file__).resolve().parents[2] / "figures" / "surrogate_trajectory_v2.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
