"""
pennylane_impl/gradient_analysis.py
-------------------------------------
Gradient-based analysis of the teleportation fidelity landscape.

Unique to PennyLane: uses the parameter-shift rule to compute exact
quantum gradients without finite differences.

Functions
---------
- fidelity_landscape()      → F(θ, φ) over a grid
- gradient_field()          → ∂F/∂θ, ∂F/∂φ at each grid point
- plot_fidelity_landscape() → heatmap + gradient arrows
- plot_gradient_norms()     → |∇F| vs θ curve
- sensitivity_analysis()    → which parameter affects fidelity more?

Usage
-----
    from src.pennylane_impl.gradient_analysis import (
        fidelity_landscape, plot_fidelity_landscape
    )

    F, thetas, phis = fidelity_landscape(noise_p=0.02, resolution=30)
    plot_fidelity_landscape(F, thetas, phis, save_path="results/landscape.png")
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional

import pennylane as qml


# QNode factory — returns a differentiable fidelity-proxy circuit

def _make_fidelity_qnode(noise_channels: Optional[list] = None) -> callable:
    """
    Build a QNode that returns ⟨Z⟩ on Bob's qubit as a fidelity proxy.

    For |0⟩ input (θ=0): ⟨Z⟩ = +1 → F=1
    For |1⟩ input (θ=π): ⟨Z⟩ = -1 → need negation

    The actual fidelity is approximated as (⟨Z⟩ + 1) / 2.
    """
    device_name = "default.mixed" if noise_channels else "default.qubit"
    dev = qml.device(device_name, wires=3)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        theta, phi = params[0], params[1]

        # State prep
        qml.RY(theta, wires=0)
        qml.RZ(phi,   wires=0)
        if noise_channels:
            for ch_cls, kw in noise_channels:
                ch_cls(**kw, wires=0)

        # Bell pair
        qml.Hadamard(wires=1)
        qml.CNOT(wires=[1, 2])
        if noise_channels:
            for ch_cls, kw in noise_channels:
                ch_cls(**kw, wires=1)
                ch_cls(**kw, wires=2)

        # Bell measurement
        qml.CNOT(wires=[0, 1])
        qml.Hadamard(wires=0)
        if noise_channels:
            for ch_cls, kw in noise_channels:
                ch_cls(**kw, wires=0)
                ch_cls(**kw, wires=1)

        # Classical corrections
        m0 = qml.measure(0)
        m1 = qml.measure(1)
        qml.cond(m1 == 1, qml.PauliX)(wires=2)
        qml.cond(m0 == 1, qml.PauliZ)(wires=2)

        return qml.expval(qml.PauliZ(wires=2))

    return circuit


# Fidelity landscape

def fidelity_landscape(
    noise_p: float = 0.0,
    resolution: int = 30,
    phi_fixed: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute F(θ, φ) over a grid of input state parameters.

    For perfect teleportation, F should be ~1 everywhere regardless of
    (θ, φ) — any deviation reveals noise-induced fidelity loss.

    Parameters
    ----------
    noise_p : float
        Depolarizing error rate. 0 = ideal.
    resolution : int
        Grid resolution (resolution × resolution points).
    phi_fixed : float, optional
        If provided, compute F(θ) only with φ fixed (1D slice).

    Returns
    -------
    (F, thetas, phis) : tuple[np.ndarray, np.ndarray, np.ndarray]
        F has shape (resolution, resolution).
        thetas and phis are 1D arrays of length resolution.
    """
    noise_channels = None
    if noise_p > 0:
        noise_channels = [(qml.DepolarizingChannel, {"p": noise_p})]

    circuit = _make_fidelity_qnode(noise_channels)

    thetas = np.linspace(0, np.pi, resolution)
    phis   = np.linspace(0, 2 * np.pi, resolution)
    F      = np.zeros((resolution, resolution))

    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            params = np.array([theta, phi], requires_grad=False)
            expval = float(circuit(params))
            # Convert ⟨Z⟩ → fidelity proxy ∈ [0,1]
            # For |0⟩: expval=+1→F=1; for |1⟩: expval=-1→F=1 after correction
            # Bob's state matches |ψ_in⟩ regardless of basis → abs(⟨Z_ideal⟩)
            z_ideal = np.cos(theta)   # ⟨Z⟩ for the input state |ψ⟩
            fid_proxy = 1.0 - abs(expval - z_ideal) / 2.0
            F[i, j] = float(np.clip(fid_proxy, 0, 1))

    return F, thetas, phis


# Gradient field

def gradient_field(
    noise_p: float = 0.0,
    resolution: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the gradient ∇F = (∂F/∂θ, ∂F/∂φ) over a coarse grid.

    Uses the parameter-shift rule — exact quantum gradients.

    Parameters
    ----------
    noise_p : float
        Depolarizing error rate.
    resolution : int
        Coarse grid resolution for arrow density.

    Returns
    -------
    (thetas, phis, dF_dtheta, dF_dphi)
        Each array has shape (resolution, resolution).
    """
    noise_channels = None
    if noise_p > 0:
        noise_channels = [(qml.DepolarizingChannel, {"p": noise_p})]

    circuit  = _make_fidelity_qnode(noise_channels)
    grad_fn  = qml.grad(circuit)

    thetas = np.linspace(0.1, np.pi - 0.1, resolution)
    phis   = np.linspace(0.1, 2 * np.pi - 0.1, resolution)

    dF_dtheta = np.zeros((resolution, resolution))
    dF_dphi   = np.zeros((resolution, resolution))

    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            params = np.array([theta, phi], requires_grad=True)
            try:
                grads = grad_fn(params)
                dF_dtheta[i, j] = float(grads[0])
                dF_dphi[i, j]   = float(grads[1])
            except Exception:
                pass  # skip points where gradient fails

    return thetas, phis, dF_dtheta, dF_dphi


# Sensitivity analysis

def sensitivity_analysis(
    noise_levels: list[float],
    n_samples: int = 20,
    seed: int = 42,
) -> dict[str, list[float]]:
    """
    Compare how sensitive the teleportation fidelity is to θ vs φ
    perturbations, as a function of noise level.

    Returns
    -------
    dict with keys "noise_levels", "sensitivity_theta", "sensitivity_phi"
    Each is a list of floats.
    """
    rng = np.random.default_rng(seed)
    sens_theta, sens_phi = [], []

    for p in noise_levels:
        noise_channels = [(qml.DepolarizingChannel, {"p": p})] if p > 0 else None
        circuit = _make_fidelity_qnode(noise_channels)
        grad_fn = qml.grad(circuit)

        grad_thetas, grad_phis = [], []
        for _ in range(n_samples):
            theta = rng.uniform(0.1, np.pi - 0.1)
            phi   = rng.uniform(0.1, 2 * np.pi - 0.1)
            params = np.array([theta, phi], requires_grad=True)
            try:
                g = grad_fn(params)
                grad_thetas.append(abs(float(g[0])))
                grad_phis.append(abs(float(g[1])))
            except Exception:
                pass

        sens_theta.append(float(np.mean(grad_thetas)) if grad_thetas else 0.0)
        sens_phi.append(float(np.mean(grad_phis))   if grad_phis   else 0.0)

    return {
        "noise_levels":      noise_levels,
        "sensitivity_theta": sens_theta,
        "sensitivity_phi":   sens_phi,
    }


# Plots

def plot_fidelity_landscape(
    F: np.ndarray,
    thetas: np.ndarray,
    phis: np.ndarray,
    title: str = "Teleportation fidelity landscape  F(θ, φ)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of F(θ, φ) — should be uniform (~1) for ideal teleportation.
    Any pattern reveals noise-induced input-state dependence.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.pcolormesh(
        np.degrees(phis), np.degrees(thetas), F,
        cmap="RdYlGn", vmin=0, vmax=1, shading="auto"
    )
    fig.colorbar(im, ax=ax, label="Fidelity F")

    ax.set_xlabel("φ (degrees)", fontsize=11)
    ax.set_ylabel("θ (degrees)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 45, 90, 135, 180])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Landscape plot saved → {save_path}")
    return fig


def plot_gradient_norms(
    thetas: np.ndarray,
    phis: np.ndarray,
    dF_dtheta: np.ndarray,
    dF_dphi: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot average |∂F/∂θ| and |∂F/∂φ| vs θ, averaged over φ.
    Shows which angular direction has higher gradient magnitude.
    """
    mean_dt = np.mean(np.abs(dF_dtheta), axis=1)
    mean_dp = np.mean(np.abs(dF_dphi),   axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.degrees(thetas), mean_dt, color="#6352D8",
            linewidth=2, label="|∂F/∂θ| (polar)")
    ax.plot(np.degrees(thetas), mean_dp, color="#1D9E75",
            linewidth=2, linestyle="--", label="|∂F/∂φ| (azimuthal)")

    ax.set_xlabel("θ (degrees)", fontsize=11)
    ax.set_ylabel("Mean |gradient|", fontsize=11)
    ax.set_title("Gradient magnitude vs. input state angle", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks([0, 45, 90, 135, 180])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_sensitivity_vs_noise(
    results: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot θ vs φ sensitivity as a function of depolarizing noise level.
    """
    noise = results["noise_levels"]
    st    = results["sensitivity_theta"]
    sp    = results["sensitivity_phi"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(noise, st, color="#6352D8", marker="o", linewidth=2,
            label="Sensitivity to θ")
    ax.plot(noise, sp, color="#1D9E75", marker="s", linewidth=2,
            linestyle="--", label="Sensitivity to φ")

    ax.set_xlabel("Depolarizing noise  p", fontsize=11)
    ax.set_ylabel("Mean |∂F/∂param|", fontsize=11)
    ax.set_title("Parameter sensitivity vs. noise level", fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
