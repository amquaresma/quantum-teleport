"""
viz/bloch.py
------------
Standalone Bloch sphere visualization module.

Functions
---------
- plot_bloch_single()     → one Bloch sphere for a QubitState
- plot_bloch_comparison() → two spheres: input vs output
- animate_teleport()      → animated transition input → output
- plot_bloch_trajectory() → path of a state through a noise channel
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Sequence

from src.core.teleport import QubitState, TeleportResult


# Colors

C_ALICE = "#E8593C"
C_BOB   = "#3B8BD4"
C_IDEAL = "#1D9E75"
C_WIRE  = "#aaaaaa"
C_GRID  = "#dddddd"


# Helpers

def _draw_sphere(ax: plt.Axes, title: str = "") -> None:
    """Draw Bloch sphere wireframe, axes, and pole labels on a 3D Axes."""
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X", fontsize=8, labelpad=2)
    ax.set_ylabel("Y", fontsize=8, labelpad=2)
    ax.set_zlabel("Z", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7)

    # Wireframe sphere
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    ax.plot_wireframe(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones(40), np.cos(v)),
        color=C_GRID, alpha=0.2, linewidth=0.4
    )
    # Axis lines
    for start, end in [
        ([-1.2,0,0],[1.2,0,0]),
        ([0,-1.2,0],[0,1.2,0]),
        ([0,0,-1.2],[0,0,1.2]),
    ]:
        ax.plot(*zip(start, end), color=C_WIRE, linewidth=0.8)

    # Labels
    for pos, lbl in [
        ([0,0,1.4],  "|0⟩"),
        ([0,0,-1.4], "|1⟩"),
        ([1.45,0,0], "|+⟩"),
        ([-1.45,0,0],"|-⟩"),
        ([0,1.45,0], "|i⟩"),
    ]:
        ax.text(*pos, lbl, fontsize=8, ha="center", va="center", color="#555555")


def _draw_vector(
    ax: plt.Axes,
    bv: np.ndarray,
    color: str,
    label: str = "",
    alpha: float = 1.0,
) -> None:
    """Draw a Bloch vector arrow and tip dot."""
    ax.quiver(
        0, 0, 0, bv[0], bv[1], bv[2],
        color=color, linewidth=2.5, alpha=alpha,
        arrow_length_ratio=0.12
    )
    ax.scatter(*bv, color=color, s=55, zorder=5, alpha=alpha)
    if label:
        offset = bv * 1.18
        ax.text(*offset, label, fontsize=8, color=color, ha="center")


def _slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit vectors."""
    v0 = v0 / (np.linalg.norm(v0) + 1e-12)
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    dot = float(np.clip(np.dot(v0, v1), -1.0, 1.0))
    omega = np.arccos(dot)
    if abs(omega) < 1e-6:
        result = (1 - t) * v0 + t * v1
    else:
        result = (
            np.sin((1-t)*omega) / np.sin(omega) * v0
            + np.sin(t*omega) / np.sin(omega) * v1
        )
    n = np.linalg.norm(result)
    return result / n if n > 1e-12 else result


# Single sphere

def plot_bloch_single(
    state: QubitState,
    color: str = C_ALICE,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a single Bloch sphere for a QubitState.

    Parameters
    ----------
    state : QubitState
    color : str
        Arrow color.
    save_path : str, optional
    """
    fig = plt.figure(figsize=(5, 5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    _draw_sphere(ax, title=f"{state.label}\nθ={state.theta:.3f}, φ={state.phi:.3f}")
    _draw_vector(ax, state.bloch_vector, color=color)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# Comparison: input vs output

def plot_bloch_comparison(
    result: TeleportResult,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side Bloch spheres: Alice's input vs Bob's output.

    The angle between the two vectors visually encodes fidelity loss.
    """
    fig = plt.figure(figsize=(11, 5), facecolor="white")
    fig.suptitle(
        f"Teleportation — Bloch sphere comparison\n"
        f"F={result.fidelity:.4f}  |  "
        f"S(ρ)={result.entropy:.4f}  |  "
        f"Noise: {result.noise_model}",
        fontsize=11, y=1.01
    )

    ax_in  = fig.add_subplot(121, projection="3d")
    ax_out = fig.add_subplot(122, projection="3d")

    _draw_sphere(ax_in,
        f"Alice's input\nθ={result.input_state.theta:.3f}, φ={result.input_state.phi:.3f}")
    _draw_sphere(ax_out,
        f"Bob's output\nθ={result.output_state.theta:.3f}, φ={result.output_state.phi:.3f}")

    _draw_vector(ax_in,  result.input_state.bloch_vector,  color=C_ALICE, label="Alice")
    _draw_vector(ax_out, result.output_state.bloch_vector, color=C_BOB,   label="Bob")

    # Faint "ideal" reference on Bob's sphere
    _draw_vector(ax_out, result.input_state.bloch_vector,
                 color=C_IDEAL, alpha=0.3, label="ideal")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Bloch comparison saved → {save_path}")
    return fig


# Animation

def animate_teleport(
    result: TeleportResult,
    n_frames: int = 80,
    interval_ms: int = 40,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    Animate the Bloch vector on Bob's sphere transitioning from
    Alice's state to Bob's received state.

    Parameters
    ----------
    result : TeleportResult
    n_frames : int
        Number of animation frames.
    interval_ms : int
        Milliseconds between frames.
    save_path : str, optional
        Save as GIF (requires Pillow).
    """
    in_bv  = result.input_state.bloch_vector
    out_bv = result.output_state.bloch_vector

    fig = plt.figure(figsize=(11, 5), facecolor="white")
    fig.suptitle(
        f"Quantum teleportation — Bloch sphere animation\n"
        f"F={result.fidelity:.4f}  |  Noise: {result.noise_model}",
        fontsize=11
    )

    ax_in  = fig.add_subplot(121, projection="3d")
    ax_out = fig.add_subplot(122, projection="3d")

    _draw_sphere(ax_in,  f"Alice's state (fixed)")
    _draw_sphere(ax_out, f"Bob's state (animating)")
    _draw_vector(ax_in, in_bv, color=C_ALICE)

    arrow = [None]
    dot   = [None]
    trail_x, trail_y, trail_z = [], [], []
    trail_line = [None]

    def animate(frame: int):
        t = frame / max(n_frames - 1, 1)
        cur = _slerp(in_bv, out_bv, t)

        if arrow[0]: arrow[0].remove()
        if dot[0]:   dot[0].remove()
        if trail_line[0]: trail_line[0].remove()

        arrow[0] = ax_out.quiver(
            0,0,0, *cur, color=C_BOB, linewidth=2.5, arrow_length_ratio=0.12
        )
        dot[0] = ax_out.scatter(*cur, color=C_BOB, s=55, zorder=5)

        trail_x.append(cur[0])
        trail_y.append(cur[1])
        trail_z.append(cur[2])
        trail_line[0], = ax_out.plot(
            trail_x, trail_y, trail_z, color=C_BOB, alpha=0.3, linewidth=1
        )

        return arrow[0], dot[0]

    anim = FuncAnimation(
        fig, animate, frames=n_frames, interval=interval_ms, blit=False
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval_ms)
        print(f"Bloch animation saved → {save_path}")

    plt.tight_layout()
    return anim


# Trajectory (noise channel path)


def plot_bloch_trajectory(
    states: Sequence[QubitState],
    title: str = "State trajectory under noise",
    color_start: str = C_IDEAL,
    color_end:   str = C_ALICE,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a sequence of Bloch vectors (e.g. state evolving under a noise channel)
    as a trajectory on the Bloch sphere.

    Parameters
    ----------
    states : list of QubitState
        Ordered list of states (first = initial, last = final).
    """
    fig = plt.figure(figsize=(6, 6), facecolor="white")
    ax  = fig.add_subplot(111, projection="3d")
    _draw_sphere(ax, title=title)

    bvs = np.array([s.bloch_vector for s in states])
    n   = len(bvs)

    for i in range(n - 1):
        t = i / (n - 1)
        # Interpolate color
        r  = (1-t) * int(color_start[1:3],16) + t * int(color_end[1:3],16)
        g  = (1-t) * int(color_start[3:5],16) + t * int(color_end[3:5],16)
        b  = (1-t) * int(color_start[5:7],16) + t * int(color_end[5:7],16)
        col = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        ax.plot(
            bvs[i:i+2, 0], bvs[i:i+2, 1], bvs[i:i+2, 2],
            color=col, linewidth=2
        )

    # Start and end markers
    ax.scatter(*bvs[0],  color=C_IDEAL, s=80, zorder=6, label="Initial")
    ax.scatter(*bvs[-1], color=C_ALICE, s=80, zorder=6, label="Final")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
