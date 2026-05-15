"""
experiments/run_qiskit.py
--------------------------
Standalone entry-point for the Qiskit teleportation experiment.

Runs the protocol across all three noise levels and generates:
    - Console summary
    - results/qiskit_histogram.png
    - results/qiskit_bloch.png
    - results/qiskit_bloch_animation.gif  (if --gif)

Usage
-----
    python -m experiments.run_qiskit
    python -m experiments.run_qiskit --shots 4096 --noise thermal --gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.core.teleport import QubitState
from src.qiskit_impl.circuit import QiskitTeleporter
from src.qiskit_impl.noise_models import (
    depolarizing_model,
    depolarizing_readout_model,
    thermal_relaxation_model,
)
from src.viz.bloch import plot_bloch_comparison, animate_teleport
from src.viz.histogram import plot_histogram
from src.viz.circuit_drawer import draw_qiskit_circuit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Qiskit teleportation experiment")
    p.add_argument("--shots",  type=int, default=4096)
    p.add_argument("--noise",  type=str, default="all",
                   choices=["ideal","depolarizing","readout","thermal","all"])
    p.add_argument("--output", type=str, default="results")
    p.add_argument("--gif",    action="store_true")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    teleporter = QiskitTeleporter(shots=args.shots, seed=42)

    # Test state: |+⟩
    state = QubitState(theta=np.pi / 2, phi=0.0, label="|+⟩")

    noise_map = {
        "ideal":        None,
        "depolarizing": depolarizing_model(error_rate=0.01),
        "readout":      depolarizing_readout_model(gate_error=0.005, readout_error=0.02),
        "thermal":      thermal_relaxation_model(),
    }

    selected = (
        list(noise_map.items())
        if args.noise == "all"
        else [(args.noise, noise_map[args.noise])]
    )

    results = []
    for label, nm in selected:
        print(f"\nRunning: {label}...")
        res = teleporter.run(state, noise_model=nm)
        print(res.summary())
        results.append(res)

    # Draw circuit
    print("\nDrawing circuit...")
    draw_qiskit_circuit(
        state, output="mpl",
        save_path=str(out_dir / "qiskit_circuit.png")
    )

    # Histogram
    plot_histogram(
        results,
        title=f"Qiskit — Bob's measurement ({args.shots} shots)",
        save_path=str(out_dir / "qiskit_histogram.png"),
    )

    # Bloch sphere
    for res in results:
        plot_bloch_comparison(
            res,
            save_path=str(out_dir / f"qiskit_bloch_{res.noise_model[:12]}.png"),
        )

    # Animation for first result
    if args.gif:
        print("Generating Bloch sphere animation...")
        animate_teleport(
            results[0],
            save_path=str(out_dir / "qiskit_bloch_animation.gif"),
        )

    print(f"\nDone. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
