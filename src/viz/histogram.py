"""
viz/histogram.py
----------------
Measurement histogram plots for teleportation results.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence

from src.core.teleport import TeleportResult

C_IDEAL = "#1D9E75"
C_NOISY = "#D85A30"
C_BOB   = "#3B8BD4"


def plot_histogram(
    results: Sequence[TeleportResult],
    title: str = "Measurement histogram — Bob's qubit",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of Bob's measurement outcome probabilities.

    Overlays the ideal uniform distribution as a dashed line.
    Annotates each bar with its probability value.

    Parameters
    ----------
    results : list[TeleportResult]
        One or more results to display side-by-side.
    title : str
    save_path : str, optional
    """
    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    for ax, res in zip(axes[0], results):
        counts = res.counts
        if not counts:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#aaaaaa")
            continue

        labels = sorted(counts.keys())
        total  = sum(counts.values())
        probs  = [counts[k] / total for k in labels]
        color  = C_IDEAL if res.noise_model == "ideal" else C_NOISY

        bars = ax.bar(labels, probs, color=color, alpha=0.85,
                      edgecolor="white", linewidth=1.4)

        for bar, prob in zip(bars, probs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{prob:.1%}",
                ha="center", va="bottom", fontsize=9, fontweight="500"
            )

        # Ideal uniform reference
        if len(labels) > 1:
            ideal = 1.0 / len(labels)
            ax.axhline(ideal, color="#888888", linestyle="--",
                       linewidth=1.2, label=f"Uniform ({ideal:.0%})")
            ax.legend(fontsize=8)

        ax.set_title(
            f"{res.backend}\n"
            f"Noise: {res.noise_model}\n"
            f"F = {res.fidelity:.4f}   PST = {res.pst:.4f}",
            fontsize=9
        )
        ax.set_xlabel("Outcome", fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, min(1.0, max(probs) * 1.35))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Histogram saved → {save_path}")
    return fig


def plot_counts_comparison(
    results_a: Sequence[TeleportResult],
    results_b: Sequence[TeleportResult],
    label_a: str = "Qiskit",
    label_b: str = "PennyLane",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side histograms comparing two frameworks across noise levels.
    """
    n = min(len(results_a), len(results_b))
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)
    fig.suptitle(f"Histogram comparison: {label_a} vs {label_b}",
                 fontsize=12, fontweight="bold")

    for col in range(n):
        for row, (results, lbl, color) in enumerate([
            (results_a, label_a, "#6352D8"),
            (results_b, label_b, C_IDEAL),
        ]):
            ax  = axes[row][col]
            res = results[col]
            counts = res.counts
            if not counts:
                continue
            labels = sorted(counts.keys())
            total  = sum(counts.values())
            probs  = [counts[k] / total for k in labels]

            ax.bar(labels, probs, color=color, alpha=0.85, edgecolor="white")
            ax.set_title(f"{lbl}\n{res.noise_model}\nF={res.fidelity:.4f}",
                         fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_ylabel("P", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
