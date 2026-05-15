"""
viz/fidelity_plot.py
--------------------
Fidelity analysis plots: noise sweeps, framework comparisons,
per-state fidelity bars.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence

from src.core.teleport import TeleportResult

C_QISKIT    = "#6352D8"
C_PENNYLANE = "#1D9E75"
C_CLASSICAL = "#ddaa00"
C_IDEAL_LINE = "#aaaaaa"


def plot_fidelity_vs_noise(
    noise_rates: Sequence[float],
    fidelities_qiskit: Optional[Sequence[float]] = None,
    fidelities_pennylane: Optional[Sequence[float]] = None,
    std_qiskit: Optional[Sequence[float]] = None,
    std_pennylane: Optional[Sequence[float]] = None,
    title: str = "Teleportation fidelity vs. depolarizing noise",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Line plot of average fidelity as a function of error rate.

    Shows:
    - Qiskit (AerSimulator) curve
    - PennyLane (default.mixed) curve
    - Classical limit F=2/3 reference line
    - Shaded confidence bands (±1σ) if std arrays provided
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    rates = np.array(noise_rates)

    if fidelities_qiskit is not None:
        fq = np.array(fidelities_qiskit)
        ax.plot(rates, fq, color=C_QISKIT, linewidth=2,
                marker="o", markersize=5, label="Qiskit / AerSimulator")
        if std_qiskit is not None:
            sq = np.array(std_qiskit)
            ax.fill_between(rates, fq - sq, fq + sq,
                            color=C_QISKIT, alpha=0.15)

    if fidelities_pennylane is not None:
        fp = np.array(fidelities_pennylane)
        ax.plot(rates, fp, color=C_PENNYLANE, linewidth=2,
                marker="s", markersize=5, linestyle="--",
                label="PennyLane / default.mixed")
        if std_pennylane is not None:
            sp = np.array(std_pennylane)
            ax.fill_between(rates, fp - sp, fp + sp,
                            color=C_PENNYLANE, alpha=0.15)

    # Reference lines
    ax.axhline(1.0, color=C_IDEAL_LINE, linestyle=":", linewidth=1,
               label="Ideal (F=1)")
    ax.axhline(2/3, color=C_CLASSICAL, linestyle=":", linewidth=1.2,
               label="Classical limit (F=2/3)")

    # Shade below classical limit
    ax.axhspan(0, 2/3, alpha=0.04, color="#ff0000")
    ax.text(rates[-1] * 0.97, 2/3 + 0.015, "classical limit",
            ha="right", va="bottom", fontsize=8, color="#aa8800")

    ax.set_xlabel("Depolarizing error rate  p", fontsize=11)
    ax.set_ylabel("Average fidelity  F", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(rates[0], rates[-1])
    ax.legend(fontsize=9, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Fidelity plot saved → {save_path}")
    return fig


def plot_framework_comparison(
    qiskit_results: Sequence[TeleportResult],
    pennylane_results: Sequence[TeleportResult],
    noise_labels: Optional[list[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing Qiskit vs PennyLane across noise configs.

    Shows fidelity, PST, and entropy for each configuration.
    """
    assert len(qiskit_results) == len(pennylane_results)

    n      = len(qiskit_results)
    labels = noise_labels or [r.noise_model for r in qiskit_results]
    x, w   = np.arange(n), 0.35

    metrics = {
        "Fidelity  F": (
            [r.fidelity for r in qiskit_results],
            [r.fidelity for r in pennylane_results],
        ),
        "PST": (
            [r.pst for r in qiskit_results],
            [r.pst for r in pennylane_results],
        ),
        "Entropy  S(ρ)": (
            [r.entropy for r in qiskit_results],
            [r.entropy for r in pennylane_results],
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Qiskit vs PennyLane — Teleportation metrics",
                 fontsize=12, fontweight="bold", y=1.02)

    for ax, (metric, (qv, pv)) in zip(axes, metrics.items()):
        bq = ax.bar(x - w/2, qv, w, label="Qiskit",
                    color=C_QISKIT, alpha=0.85)
        bp = ax.bar(x + w/2, pv, w, label="PennyLane",
                    color=C_PENNYLANE, alpha=0.85)

        for bars in [bq, bp]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom",
                        fontsize=7.5, rotation=40)

        ax.set_title(metric, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
        ax.set_ylim(0, 1.18)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Comparison chart saved → {save_path}")
    return fig


def plot_fidelity_per_state(
    results_qiskit: Sequence[TeleportResult],
    results_pl: Sequence[TeleportResult],
    state_labels: list[str],
    noise_label: str = "ideal",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of fidelity for each of the canonical input states.
    Useful for verifying protocol correctness across |0⟩, |1⟩, |+⟩, |-⟩, etc.
    """
    x, w = np.arange(len(state_labels)), 0.35
    fq = [r.fidelity for r in results_qiskit]
    fp = [r.fidelity for r in results_pl]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, fq, w, label="Qiskit",    color=C_QISKIT,    alpha=0.85)
    ax.bar(x + w/2, fp, w, label="PennyLane", color=C_PENNYLANE, alpha=0.85)
    ax.axhline(1.0, color=C_IDEAL_LINE, linestyle=":", linewidth=1)
    ax.axhline(2/3, color=C_CLASSICAL, linestyle=":", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(state_labels, fontsize=10)
    ax.set_ylabel("Fidelity F", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title(f"Fidelity per input state — {noise_label}", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
