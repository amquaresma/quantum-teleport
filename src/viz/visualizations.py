"""
viz/visualizations.py
---------------------
All visualization functions for the quantum teleportation project.

Contents:
    - plot_bloch_sphere_animation()  → animated Bloch sphere (input vs output)
    - plot_histogram()               → measurement count histograms
    - plot_fidelity_vs_noise()       → fidelity curves across noise levels
    - plot_framework_comparison()    → Qiskit vs PennyLane side-by-side
    - draw_teleport_circuit()        → circuit diagram using Qiskit

Usage
-----
    from src.viz.visualizations import (
        plot_bloch_sphere_animation,
        plot_histogram,
        plot_fidelity_vs_noise,
        plot_framework_comparison,
    )
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from typing import Optional, Sequence

from src.core.teleport import QubitState, TeleportResult



# Color palette (consistent across all plots)

COLORS = {
    "alice":    "#E8593C",   # coral-red  (Alice / input)
    "bob":      "#3B8BD4",   # blue       (Bob / output)
    "ideal":    "#1D9E75",   # teal       (ideal / noiseless)
    "noisy":    "#D85A30",   # orange-red (noisy)
    "qiskit":   "#6352D8",   # purple     (Qiskit)
    "pennylane":"#1D9E75",   # teal       (PennyLane)
    "grid":     "#cccccc",
    "bg":       "#f9f9f9",
}

STYLE = {
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
}


# 1. Bloch Sphere Animation

def plot_bloch_sphere_animation(
    result: TeleportResult,
    n_frames: int = 60,
    interval_ms: int = 50,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    Animated Bloch sphere showing Alice's input state morphing into Bob's
    output state over the course of the teleportation.

    Parameters
    ----------
    result : TeleportResult
        Teleportation result containing input and output states.
    n_frames : int
        Number of animation frames (smoothness).
    interval_ms : int
        Milliseconds between frames.
    save_path : str, optional
        If provided, save animation as GIF to this path.

    Returns
    -------
    FuncAnimation
    """
    fig = plt.figure(figsize=(10, 5), facecolor="white")
    fig.suptitle(
        f"Quantum Teleportation — Bloch Sphere\n"
        f"Fidelity: {result.fidelity:.4f}  |  "
        f"Noise: {result.noise_model}  |  "
        f"Backend: {result.backend}",
        fontsize=11, y=1.01
    )

    ax_in  = fig.add_subplot(121, projection="3d")
    ax_out = fig.add_subplot(122, projection="3d")

    in_bv  = result.input_state.bloch_vector
    out_bv = result.output_state.bloch_vector

    def setup_sphere(ax: plt.Axes, title: str, color: str) -> None:
        """Draw the Bloch sphere wireframe and axes."""
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_box_aspect([1, 1, 1])

        # Wireframe sphere
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color="#dddddd", alpha=0.25, linewidth=0.4)

        # Axes lines
        for (x1, y1, z1), (x2, y2, z2) in [
            ((-1.2,0,0),(1.2,0,0)),
            ((0,-1.2,0),(0,1.2,0)),
            ((0,0,-1.2),(0,0,1.2)),
        ]:
            ax.plot([x1,x2],[y1,y2],[z1,z2], color="#aaaaaa", linewidth=0.7)

        # Pole labels
        for pos, label in [((0,0,1.35),"|0⟩"), ((0,0,-1.35),"|1⟩"),
                            ((1.4,0,0),"|+⟩"),  ((-1.4,0,0),"|-⟩")]:
            ax.text(*pos, label, fontsize=8, ha="center", va="center",
                    color="#666666")

    setup_sphere(ax_in,  f"Alice's state\nθ={result.input_state.theta:.3f}, φ={result.input_state.phi:.3f}", COLORS["alice"])
    setup_sphere(ax_out, f"Bob's state\nθ={result.output_state.theta:.3f}, φ={result.output_state.phi:.3f}", COLORS["bob"])

    # Static arrow for input (doesn't change)
    ax_in.quiver(0, 0, 0, *in_bv, color=COLORS["alice"], linewidth=2.5,
                 arrow_length_ratio=0.12)
    ax_in.scatter(*in_bv, color=COLORS["alice"], s=60, zorder=5)

    # Animated arrow for output (morphs from input to output)
    arrow_obj = [None]
    dot_obj   = [None]
    trail_x, trail_y, trail_z = [], [], []
    trail_obj = [None]

    def animate(frame: int):
        t = frame / max(n_frames - 1, 1)
        # Slerp between input and output Bloch vectors
        current = _slerp(in_bv, out_bv, t)

        # Remove previous arrow
        if arrow_obj[0] is not None:
            arrow_obj[0].remove()
        if dot_obj[0] is not None:
            dot_obj[0].remove()
        if trail_obj[0] is not None:
            trail_obj[0].remove()

        arrow_obj[0] = ax_out.quiver(
            0, 0, 0, *current,
            color=COLORS["bob"], linewidth=2.5, arrow_length_ratio=0.12
        )
        dot_obj[0] = ax_out.scatter(*current, color=COLORS["bob"], s=60, zorder=5)

        # Trail
        trail_x.append(current[0])
        trail_y.append(current[1])
        trail_z.append(current[2])
        trail_obj[0], = ax_out.plot(
            trail_x, trail_y, trail_z,
            color=COLORS["bob"], alpha=0.3, linewidth=1
        )

        return arrow_obj[0], dot_obj[0]

    anim = FuncAnimation(
        fig, animate,
        frames=n_frames,
        interval=interval_ms,
        blit=False,
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval_ms)
        print(f"Animation saved → {save_path}")

    plt.tight_layout()
    return anim


# 2. Measurement Histogram

def plot_histogram(
    results: list[TeleportResult],
    title: str = "Measurement histogram — Bob's qubit",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of Bob's measurement counts for one or more TeleportResults.

    Parameters
    ----------
    results : list[TeleportResult]
        One or more results to compare side-by-side.
    title : str
        Figure title.
    save_path : str, optional
        Save figure to path.

    Returns
    -------
    matplotlib Figure
    """
    with plt.rc_context(STYLE):
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

        for ax, res in zip(axes[0], results):
            counts = res.counts
            labels = sorted(counts.keys())
            values = [counts[k] for k in labels]
            total  = sum(values)
            probs  = [v / total for v in values]

            color = COLORS["ideal"] if res.noise_model == "ideal" else COLORS["noisy"]
            bars = ax.bar(labels, probs, color=color, alpha=0.85, edgecolor="white",
                          linewidth=1.2)

            # Annotate bars with probability
            for bar, prob in zip(bars, probs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{prob:.2%}",
                    ha="center", va="bottom", fontsize=9
                )

            # Ideal uniform line
            if len(labels) > 1:
                ideal_prob = 1.0 / len(labels)
                ax.axhline(ideal_prob, color="#888888", linestyle="--",
                           linewidth=1, label=f"Ideal ({ideal_prob:.0%})")
                ax.legend(fontsize=8)

            ax.set_title(
                f"{res.backend}\n"
                f"Noise: {res.noise_model}\n"
                f"F={res.fidelity:.4f}  PST={res.pst:.4f}",
                fontsize=9
            )
            ax.set_xlabel("Measurement outcome", fontsize=9)
            ax.set_ylabel("Probability", fontsize=9)
            ax.set_ylim(0, min(1.0, max(probs) * 1.3))

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Histogram saved → {save_path}")
        return fig


# 3. Fidelity vs. Noise

def plot_fidelity_vs_noise(
    noise_rates: Sequence[float],
    fidelities_qiskit: Optional[Sequence[float]] = None,
    fidelities_pennylane: Optional[Sequence[float]] = None,
    std_qiskit: Optional[Sequence[float]] = None,
    std_pennylane: Optional[Sequence[float]] = None,
    title: str = "Teleportation fidelity vs. depolarizing noise rate",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Line plot of fidelity as a function of noise (error rate).

    Can overlay Qiskit and PennyLane curves for comparison.

    Parameters
    ----------
    noise_rates : array-like
        X-axis values (error rates, e.g. np.linspace(0, 0.15, 20)).
    fidelities_qiskit : array-like, optional
        Fidelity values for Qiskit.
    fidelities_pennylane : array-like, optional
        Fidelity values for PennyLane.
    std_qiskit, std_pennylane : array-like, optional
        Standard deviations for shaded confidence bands.
    title : str
        Figure title.
    save_path : str, optional
        Save to path.

    Returns
    -------
    matplotlib Figure
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        rates = np.array(noise_rates)

        if fidelities_qiskit is not None:
            fq = np.array(fidelities_qiskit)
            ax.plot(rates, fq, color=COLORS["qiskit"], linewidth=2,
                    marker="o", markersize=5, label="Qiskit / AerSimulator")
            if std_qiskit is not None:
                sq = np.array(std_qiskit)
                ax.fill_between(rates, fq - sq, fq + sq,
                                color=COLORS["qiskit"], alpha=0.15)

        if fidelities_pennylane is not None:
            fp = np.array(fidelities_pennylane)
            ax.plot(rates, fp, color=COLORS["pennylane"], linewidth=2,
                    marker="s", markersize=5, linestyle="--",
                    label="PennyLane / default.mixed")
            if std_pennylane is not None:
                sp = np.array(std_pennylane)
                ax.fill_between(rates, fp - sp, fp + sp,
                                color=COLORS["pennylane"], alpha=0.15)

        # Classical limit (F=2/3) and ideal (F=1)
        ax.axhline(1.0, color="#aaaaaa", linestyle=":", linewidth=1,
                   label="Ideal fidelity (F=1)")
        ax.axhline(2/3, color="#ddaa00", linestyle=":", linewidth=1,
                   label="Classical limit (F=2/3)")

        ax.set_xlabel("Depolarizing error rate  p", fontsize=11)
        ax.set_ylabel("Average teleportation fidelity  F", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(rates[0], rates[-1])
        ax.legend(fontsize=9, loc="lower left")

        # Shade the "worse than classical" region
        ax.axhspan(0, 2/3, alpha=0.04, color="#ff0000",
                   label="_nolegend_")
        ax.text(
            rates[-1] * 0.98, 2/3 + 0.01,
            "classical limit", ha="right", va="bottom",
            fontsize=8, color="#aa8800"
        )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Fidelity plot saved → {save_path}")
        return fig


# 4. Framework comparison dashboard

def plot_framework_comparison(
    qiskit_results: list[TeleportResult],
    pennylane_results: list[TeleportResult],
    noise_labels: Optional[list[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of Qiskit vs PennyLane across noise levels.

    Shows fidelity, PST, and entropy for each noise configuration.

    Parameters
    ----------
    qiskit_results : list[TeleportResult]
        Results from QiskitTeleporter at various noise levels.
    pennylane_results : list[TeleportResult]
        Results from PennyLaneTeleporter at the same noise levels.
    noise_labels : list[str], optional
        X-axis labels. Defaults to noise_model strings from results.
    save_path : str, optional
        Save to path.
    """
    assert len(qiskit_results) == len(pennylane_results), \
        "Must provide equal number of results for each framework."

    n = len(qiskit_results)
    labels = noise_labels or [r.noise_model for r in qiskit_results]
    x = np.arange(n)
    w = 0.35

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

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(
            "Qiskit vs PennyLane — Teleportation metrics across noise levels",
            fontsize=12, fontweight="bold", y=1.03
        )

        for ax, (metric_name, (qvals, pvals)) in zip(axes, metrics.items()):
            bars_q = ax.bar(x - w/2, qvals, w, label="Qiskit",
                            color=COLORS["qiskit"], alpha=0.85)
            bars_p = ax.bar(x + w/2, pvals, w, label="PennyLane",
                            color=COLORS["pennylane"], alpha=0.85)

            for bars in [bars_q, bars_p]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.01,
                        f"{height:.3f}",
                        ha="center", va="bottom", fontsize=7.5, rotation=45
                    )

            ax.set_title(metric_name, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
            ax.set_ylim(0, 1.15)
            ax.legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Comparison plot saved → {save_path}")
        return fig


# 5. Circuit diagram

def draw_teleport_circuit(
    state: Optional[QubitState] = None,
    output: str = "mpl",
    save_path: Optional[str] = None,
) -> object:
    """
    Draw the quantum teleportation circuit using Qiskit.

    Parameters
    ----------
    state : QubitState, optional
        Input state. Defaults to |+⟩ (θ=π/2, φ=0).
    output : str
        "mpl" for matplotlib figure, "text" for ASCII art.
    save_path : str, optional
        Save matplotlib figure to path.

    Returns
    -------
    Matplotlib Figure or string (depending on output format).
    """
    from src.qiskit_impl.circuit import QiskitTeleporter

    if state is None:
        state = QubitState(theta=np.pi / 2, phi=0.0, label="|+⟩")

    teleporter = QiskitTeleporter(shots=1024)
    qc = teleporter.build_circuit(state)

    if output == "mpl":
        fig = qc.draw(output="mpl", style="iqp", fold=60)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Circuit diagram saved → {save_path}")
        return fig

    return qc.draw(output="text")


# Utility: spherical linear interpolation (slerp) for Bloch vectors

def _slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two unit vectors.
    Falls back to linear interpolation if vectors are near-parallel.
    """
    v0 = v0 / (np.linalg.norm(v0) + 1e-12)
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    omega = np.arccos(dot)
    if np.abs(omega) < 1e-6:
        # Nearly identical vectors — linear interpolation
        result = (1 - t) * v0 + t * v1
    else:
        result = (np.sin((1 - t) * omega) / np.sin(omega)) * v0 + \
                 (np.sin(t * omega) / np.sin(omega)) * v1
    norm = np.linalg.norm(result)
    return result / norm if norm > 1e-12 else result
