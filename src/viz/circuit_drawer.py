"""
viz/circuit_drawer.py
---------------------
Circuit diagram utilities for the teleportation protocol.

Wraps Qiskit's circuit.draw() with sensible defaults and
adds a PennyLane circuit drawing option.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal

from src.core.teleport import QubitState


def draw_qiskit_circuit(
    state: Optional[QubitState] = None,
    output: Literal["mpl", "text", "latex_source"] = "mpl",
    fold: int = 60,
    style: str = "iqp",
    save_path: Optional[str] = None,
) -> object:
    """
    Draw the Qiskit teleportation circuit.

    Parameters
    ----------
    state : QubitState, optional
        Input state. Defaults to |+⟩.
    output : str
        "mpl" → matplotlib Figure (best for saving/displaying).
        "text" → ASCII string (good for terminal/notebooks).
        "latex_source" → LaTeX string.
    fold : int
        Fold long circuits at this column width.
    style : str
        Qiskit diagram style: "iqp", "bw", "clifford", etc.
    save_path : str, optional
        Save path for "mpl" output.

    Returns
    -------
    matplotlib Figure or str
    """
    from src.qiskit_impl.circuit import QiskitTeleporter

    if state is None:
        state = QubitState(theta=np.pi / 2, phi=0.0, label="|+⟩")

    teleporter = QiskitTeleporter(shots=1024)
    qc = teleporter.build_circuit(state)

    if output == "mpl":
        fig = qc.draw(output="mpl", style=style, fold=fold)
        fig.suptitle(
            f"Quantum Teleportation Circuit\n"
            f"Input: θ={state.theta:.3f} rad, φ={state.phi:.3f} rad",
            fontsize=10, y=1.02
        )
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Circuit diagram saved → {save_path}")
        return fig

    return qc.draw(output=output)


def draw_pennylane_circuit(
    state: Optional[QubitState] = None,
    noise_p: float = 0.0,
    expansion_strategy: str = "device",
) -> str:
    """
    Draw the PennyLane teleportation circuit as a Unicode string.

    Parameters
    ----------
    state : QubitState, optional
        Input state. Defaults to |+⟩.
    noise_p : float
        Depolarizing noise rate for the noisy version.
    expansion_strategy : str
        PennyLane draw expansion strategy.

    Returns
    -------
    str
        ASCII/Unicode circuit diagram.
    """
    import pennylane as qml
    from src.pennylane_impl.circuit import PennyLaneTeleporter, make_depolarizing_channels

    if state is None:
        state = QubitState(theta=np.pi / 2, phi=0.0, label="|+⟩")

    noise = make_depolarizing_channels(noise_p) if noise_p > 0 else None
    teleporter = PennyLaneTeleporter(shots=None, device_name="default.mixed")
    circuit_fn = teleporter.build_density_matrix_circuit(state, noise_channels=noise)

    return qml.draw(circuit_fn, expansion_strategy=expansion_strategy)()


def draw_both(
    state: Optional[QubitState] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Draw Qiskit circuit (mpl) and PennyLane circuit (text) side by side.

    Parameters
    ----------
    state : QubitState, optional
    save_path : str, optional

    Returns
    -------
    matplotlib Figure
    """
    if state is None:
        state = QubitState(theta=np.pi / 2, phi=0.0, label="|+⟩")

    pl_text = draw_pennylane_circuit(state)

    fig = plt.figure(figsize=(14, 6), facecolor="white")
    fig.suptitle("Teleportation circuit — Qiskit vs PennyLane",
                 fontsize=12, fontweight="bold")

    # Left: PennyLane text circuit
    ax_pl = fig.add_subplot(121)
    ax_pl.axis("off")
    ax_pl.set_title("PennyLane", fontsize=10, pad=8)
    ax_pl.text(
        0.02, 0.95, pl_text,
        transform=ax_pl.transAxes,
        fontsize=8, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f4f4f4", alpha=0.8)
    )

    # Right: Qiskit mpl circuit (inlined as image)
    from src.qiskit_impl.circuit import QiskitTeleporter
    teleporter = QiskitTeleporter(shots=1)
    qc         = teleporter.build_circuit(state)
    qc_fig     = qc.draw(output="mpl", style="bw", fold=40)

    # Convert Qiskit figure to image array and embed
    import io
    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    qc_fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    img = np.array(Image.open(buf))
    plt.close(qc_fig)

    ax_qk = fig.add_subplot(122)
    ax_qk.imshow(img)
    ax_qk.axis("off")
    ax_qk.set_title("Qiskit", fontsize=10, pad=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Dual circuit diagram saved → {save_path}")
    return fig
