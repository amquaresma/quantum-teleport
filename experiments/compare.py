"""
experiments/compare.py
-----------------------
Full benchmark: Qiskit vs PennyLane across three noise levels.

Runs:
    1. Ideal (no noise)
    2. Light depolarizing noise (p=0.005)
    3. Heavy depolarizing noise (p=0.05)
    4. Noise sweep (p = 0 → 0.15, 20 steps)

Outputs:
    - Console summary table
    - results/histogram_comparison.png
    - results/fidelity_vs_noise.png
    - results/framework_comparison.png
    - results/bloch_animation.gif

Usage
-----
    python -m experiments.compare
    python -m experiments.compare --shots 4096 --samples 30 --output results/
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from src.core.teleport import QubitState
from src.core.metrics import average_fidelity_over_bloch_sphere

from src.qiskit_impl.circuit import QiskitTeleporter
from src.qiskit_impl.noise_models import (
    depolarizing_model,
    depolarizing_readout_model,
    thermal_relaxation_model,
    make_depolarizing_sweep,
)

from src.pennylane_impl.circuit import (
    PennyLaneTeleporter,
    make_depolarizing_channels,
)

from src.viz.visualizations import (
    plot_histogram,
    plot_fidelity_vs_noise,
    plot_framework_comparison,
    plot_bloch_sphere_animation,
    draw_teleport_circuit,
)


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark quantum teleportation: Qiskit vs PennyLane"
    )
    p.add_argument("--shots",   type=int, default=4096,
                   help="Shots per experiment (default: 4096)")
    p.add_argument("--samples", type=int, default=20,
                   help="Random states for average fidelity (default: 20)")
    p.add_argument("--noise-steps", type=int, default=15,
                   help="Points in noise sweep (default: 15)")
    p.add_argument("--output",  type=str, default="results",
                   help="Output directory (default: results/)")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip generating figures")
    p.add_argument("--gif", action="store_true",
                   help="Generate Bloch sphere animation GIF")
    return p.parse_args()


# Helpers

def make_test_states() -> list[QubitState]:
    """Six canonical input states spanning the Bloch sphere."""
    return [
        QubitState(theta=0.0,         phi=0.0,          label="|0⟩"),
        QubitState(theta=np.pi,       phi=0.0,          label="|1⟩"),
        QubitState(theta=np.pi / 2,   phi=0.0,          label="|+⟩"),
        QubitState(theta=np.pi / 2,   phi=np.pi,        label="|-⟩"),
        QubitState(theta=np.pi / 2,   phi=np.pi / 2,    label="|i⟩"),
        QubitState(theta=np.pi / 4,   phi=np.pi / 3,    label="arbitrary"),
    ]


def _header(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def _table_row(label: str, qf: float, pf: float, qpst: float, ppst: float) -> None:
    print(
        f"  {label:<20} "
        f"Qiskit F={qf:.4f}  PST={qpst:.4f}  │  "
        f"PL F={pf:.4f}  PST={ppst:.4f}"
    )


# Main benchmark

def run_benchmark(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    shots   = args.shots
    rng     = np.random.default_rng(42)

    qiskit_tp = QiskitTeleporter(shots=shots, seed=42)
    pl_tp     = PennyLaneTeleporter(shots=shots, seed=42, device_name="default.mixed")

    # ── Test input states 
    test_states = make_test_states()

    # ── Noise configurations
    noise_configs = [
        {
            "label":   "Ideal",
            "qiskit":  None,
            "pl":      None,
        },
        {
            "label":   "Depolarizing p=0.005",
            "qiskit":  depolarizing_model(error_rate=0.005),
            "pl":      make_depolarizing_channels(p=0.005),
        },
        {
            "label":   "Depolarizing p=0.02",
            "qiskit":  depolarizing_model(error_rate=0.02),
            "pl":      make_depolarizing_channels(p=0.02),
        },
        {
            "label":   "Dep+Readout",
            "qiskit":  depolarizing_readout_model(gate_error=0.005, readout_error=0.02),
            "pl":      make_depolarizing_channels(p=0.005),  # closest PL equivalent
        },
        {
            "label":   "Thermal (IBM-like)",
            "qiskit":  thermal_relaxation_model(),
            "pl":      make_depolarizing_channels(p=0.01),   # approx equivalent
        },
    ]

    # Section 1 — Per-state results for each noise config
    _header("Section 1: Per-state fidelity across noise levels")

    all_qiskit_results: list = []
    all_pl_results:     list = []

    reference_state = QubitState(theta=np.pi / 2, phi=0.0, label="|+⟩")

    for cfg in noise_configs:
        label = cfg["label"]
        t0 = time.time()

        qr = qiskit_tp.run(reference_state, noise_model=cfg["qiskit"])
        pr = pl_tp.run(reference_state, noise_model=cfg["pl"])

        elapsed = time.time() - t0
        _table_row(label, qr.fidelity, pr.fidelity, qr.pst, pr.pst)
        print(f"    └─ wall time: {elapsed:.2f}s")

        all_qiskit_results.append(qr)
        all_pl_results.append(pr)

    
    # Section 2 — Average fidelity over random Bloch sphere states
    
    _header("Section 2: Average fidelity over random input states")
    print(f"  Sampling {args.samples} random states per config...\n")

    for cfg in noise_configs:
        qmean, qstd = average_fidelity_over_bloch_sphere(
            qiskit_tp, n_samples=args.samples,
            noise_model=cfg["qiskit"], rng=rng
        )
        pmean, pstd = average_fidelity_over_bloch_sphere(
            pl_tp, n_samples=args.samples,
            noise_model=cfg["pl"], rng=rng
        )
        print(
            f"  {cfg['label']:<26} "
            f"Qiskit  {qmean:.4f} ± {qstd:.4f}  │  "
            f"PL  {pmean:.4f} ± {pstd:.4f}"
        )


    # Section 3 — Noise sweep
    _header("Section 3: Fidelity vs. noise sweep")

    noise_rates = np.linspace(0.0, 0.15, args.noise_steps)
    sweep = make_depolarizing_sweep(list(noise_rates))

    fid_qiskit, fid_pl = [], []
    std_qiskit, std_pl = [], []

    for rate, nm_qiskit in sweep:
        nm_pl = make_depolarizing_channels(p=rate) if rate > 0 else None

        qmean, qstd = average_fidelity_over_bloch_sphere(
            qiskit_tp, n_samples=args.samples,
            noise_model=nm_qiskit, rng=rng
        )
        pmean, pstd = average_fidelity_over_bloch_sphere(
            pl_tp, n_samples=args.samples,
            noise_model=nm_pl, rng=rng
        )

        fid_qiskit.append(qmean); std_qiskit.append(qstd)
        fid_pl.append(pmean);     std_pl.append(pstd)

        print(
            f"  p={rate:.3f}  Qiskit {qmean:.4f}±{qstd:.4f}  "
            f"PL {pmean:.4f}±{pstd:.4f}"
        )

    # Section 4 — Circuit diagram (text)
    _header("Section 4: Teleportation circuit (ASCII)")
    print(draw_teleport_circuit(reference_state, output="text"))

    # Plots
    if not args.no_plots:
        _header("Generating plots...")

        # — Histogram comparison (ideal vs noisiest) —
        plot_histogram(
            [all_qiskit_results[0], all_qiskit_results[-1],
             all_pl_results[0],     all_pl_results[-1]],
            title="Measurement histograms — Ideal vs Noisy",
            save_path=str(out_dir / "histogram_comparison.png"),
        )
        print("  ✓ histogram_comparison.png")

        # — Fidelity vs noise sweep —
        plot_fidelity_vs_noise(
            noise_rates=noise_rates,
            fidelities_qiskit=fid_qiskit,
            fidelities_pennylane=fid_pl,
            std_qiskit=std_qiskit,
            std_pennylane=std_pl,
            save_path=str(out_dir / "fidelity_vs_noise.png"),
        )
        print("  ✓ fidelity_vs_noise.png")

        # — Framework comparison bar chart —
        plot_framework_comparison(
            qiskit_results=all_qiskit_results,
            pennylane_results=all_pl_results,
            noise_labels=[c["label"] for c in noise_configs],
            save_path=str(out_dir / "framework_comparison.png"),
        )
        print("  ✓ framework_comparison.png")

        # — Circuit diagram —
        try:
            draw_teleport_circuit(
                reference_state,
                output="mpl",
                save_path=str(out_dir / "circuit_diagram.png"),
            )
            print("  ✓ circuit_diagram.png")
        except Exception as e:
            print(f"  ⚠ Circuit diagram skipped: {e}")

        # — Bloch sphere animation (optional, slow) —
        if args.gif:
            anim = plot_bloch_sphere_animation(
                all_qiskit_results[2],  # light noise result
                save_path=str(out_dir / "bloch_animation.gif"),
            )
            print("  ✓ bloch_animation.gif")

    _header("Benchmark complete")
    print(f"  Results saved to: {out_dir.resolve()}\n")



# Entry point

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
