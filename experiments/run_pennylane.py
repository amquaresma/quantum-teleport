"""
experiments/run_pennylane.py
-----------------------------
Standalone entry-point for the PennyLane teleportation experiment.

Unique to this script (vs run_qiskit.py):
    - Gradient analysis via parameter-shift rule
    - Fidelity landscape F(θ, φ)
    - Sensitivity analysis (θ vs φ)

Usage
-----
    python -m experiments.run_pennylane
    python -m experiments.run_pennylane --shots 4096 --gradients --landscape
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.core.teleport import QubitState
from src.pennylane_impl.circuit import (
    PennyLaneTeleporter,
    make_depolarizing_channels,
    make_amplitude_damping_channels,
)
from src.pennylane_impl.gradient_analysis import (
    fidelity_landscape,
    gradient_field,
    sensitivity_analysis,
    plot_fidelity_landscape,
    plot_gradient_norms,
    plot_sensitivity_vs_noise,
)
from src.viz.bloch import plot_bloch_comparison, animate_teleport
from src.viz.histogram import plot_histogram
from src.viz.circuit_drawer import draw_pennylane_circuit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PennyLane teleportation experiment")
    p.add_argument("--shots",     type=int, default=4096)
    p.add_argument("--output",    type=str, default="results")
    p.add_argument("--gradients", action="store_true",
                   help="Run gradient analysis (slow)")
    p.add_argument("--landscape", action="store_true",
                   help="Compute fidelity landscape F(θ,φ) (slow)")
    p.add_argument("--gif",       action="store_true",
                   help="Generate Bloch sphere animation")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    teleporter = PennyLaneTeleporter(
        shots=args.shots, seed=42, device_name="default.mixed"
    )

    state = QubitState(theta=np.pi / 2, phi=0.0, label="|+⟩")

    noise_configs = [
        ("ideal",             None),
        ("depolarizing_0.01", make_depolarizing_channels(p=0.01)),
        ("depolarizing_0.05", make_depolarizing_channels(p=0.05)),
        ("amplitude_damping", make_amplitude_damping_channels(gamma=0.02)),
    ]

    # ── Main runs ────────────────────────────────────────────────────────
    results = []
    for label, noise in noise_configs:
        print(f"\nRunning PennyLane: {label}...")
        res = teleporter.run(state, noise_model=noise)
        print(res.summary())
        results.append(res)

    # ── Circuit diagram ───────────────────────────────────────────────────
    print("\nPennyLane circuit diagram:")
    print(draw_pennylane_circuit(state))

    # ── Plots ────────────────────────────────────────────────────────────
    plot_histogram(
        results,
        title=f"PennyLane — Bob's measurement ({args.shots} shots)",
        save_path=str(out_dir / "pl_histogram.png"),
    )

    for res in results:
        plot_bloch_comparison(
            res,
            save_path=str(out_dir / f"pl_bloch_{res.noise_model[:12]}.png"),
        )

    if args.gif:
        animate_teleport(
            results[1],  # noisy result is more interesting
            save_path=str(out_dir / "pl_bloch_animation.gif"),
        )

    # ── Gradient analysis (parameter-shift) ─────────────────────────────
    if args.gradients:
        print("\nRunning gradient analysis (parameter-shift rule)...")

        # Gradient of fidelity at several points
        for theta in [np.pi/4, np.pi/2, 3*np.pi/4]:
            dtheta, dphi = teleporter.gradient_of_fidelity(theta, phi=0.0)
            print(f"  θ={np.degrees(theta):.0f}°  →  "
                  f"∂F/∂θ = {dtheta:.4f}   ∂F/∂φ = {dphi:.4f}")

        # Sensitivity vs noise
        print("\nSensitivity analysis...")
        noise_levels = list(np.linspace(0, 0.10, 10))
        sens = sensitivity_analysis(noise_levels, n_samples=15)
        plot_sensitivity_vs_noise(
            sens,
            save_path=str(out_dir / "pl_sensitivity.png"),
        )
        print(f"  Saved: pl_sensitivity.png")

        # Gradient field over coarse grid
        print("Computing gradient field (resolution=10)...")
        thetas, phis, dFdt, dFdp = gradient_field(noise_p=0.02, resolution=10)
        plot_gradient_norms(
            thetas, phis, dFdt, dFdp,
            save_path=str(out_dir / "pl_gradient_norms.png"),
        )
        print(f"  Saved: pl_gradient_norms.png")

    # ── Fidelity landscape ───────────────────────────────────────────────
    if args.landscape:
        print("\nComputing fidelity landscape F(θ,φ)...")
        for p in [0.0, 0.02]:
            F, thetas, phis = fidelity_landscape(noise_p=p, resolution=25)
            fname = f"pl_landscape_p{int(p*100):02d}.png"
            plot_fidelity_landscape(
                F, thetas, phis,
                title=f"F(θ,φ) — noise p={p:.2f}",
                save_path=str(out_dir / fname),
            )
            print(f"  Saved: {fname}  (mean F={F.mean():.4f})")

    print(f"\nDone. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
