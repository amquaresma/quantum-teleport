"""
core/metrics.py
---------------
Quantum information metrics used to evaluate teleportation quality.

All functions operate on NumPy arrays (statevectors or density matrices)
and are framework-agnostic — called by both Qiskit and PennyLane modules.

References
----------
Nielsen & Chuang, "Quantum Computation and Quantum Information", Ch. 9.
"""

from __future__ import annotations

import numpy as np
from typing import Union



# Type aliases


Statevector = np.ndarray   # shape (2,)  or (2^n,)
DensityMatrix = np.ndarray # shape (2,2) or (2^n, 2^n)


# Conversion helpers

def to_density_matrix(state: Union[Statevector, DensityMatrix]) -> DensityMatrix:
    """
    Convert a pure statevector |ψ⟩ to its density matrix ρ = |ψ⟩⟨ψ|.
    If a density matrix is passed, return it unchanged.
    """
    state = np.asarray(state, dtype=complex)
    if state.ndim == 1:
        return np.outer(state, state.conj())
    if state.ndim == 2 and state.shape[0] == state.shape[1]:
        return state
    raise ValueError(
        f"Expected shape (N,) or (N,N), got {state.shape}"
    )


def partial_trace(rho: DensityMatrix, keep: int, dims: tuple[int, int]) -> DensityMatrix:
    """
    Compute the partial trace of a bipartite density matrix.

    Parameters
    ----------
    rho : DensityMatrix
        Combined density matrix of shape (d0*d1, d0*d1).
    keep : int
        Subsystem to keep: 0 = first, 1 = second.
    dims : tuple[int, int]
        Dimensions of the two subsystems (d0, d1).

    Returns
    -------
    DensityMatrix
        Reduced density matrix of shape (d_keep, d_keep).
    """
    d0, d1 = dims
    rho = rho.reshape(d0, d1, d0, d1)
    if keep == 0:
        return np.einsum("ibjb->ij", rho)
    return np.einsum("aiba->ib", rho.reshape(d0, d1, d0, d1)).reshape(d1, d1)


# Core metric functions

def fidelity(
    state_in: Union[Statevector, DensityMatrix],
    state_out: Union[Statevector, DensityMatrix],
) -> float:
    """
    Quantum fidelity F(ρ, σ) ∈ [0, 1].

    For pure states: F = |⟨ψ|φ⟩|²
    For mixed states: F = (Tr[√(√ρ σ √ρ)])²

    Parameters
    ----------
    state_in : array-like
        Reference (input) state — statevector or density matrix.
    state_out : array-like
        Target (output/reconstructed) state.

    Returns
    -------
    float
        Fidelity value. 1.0 = perfect teleportation.
    """
    rho = to_density_matrix(state_in)
    sigma = to_density_matrix(state_out)

    # Fast path for pure states
    if _is_pure(rho) and _is_pure(sigma):
        overlap = np.abs(np.trace(rho @ sigma))
        return float(np.clip(overlap, 0.0, 1.0))

    # General Uhlmann fidelity
    sqrt_rho = _matrix_sqrt(rho)
    m = sqrt_rho @ sigma @ sqrt_rho
    sqrt_m = _matrix_sqrt(m)
    f = (np.real(np.trace(sqrt_m))) ** 2
    return float(np.clip(f, 0.0, 1.0))


def von_neumann_entropy(
    state: Union[Statevector, DensityMatrix],
    base: float = 2.0,
) -> float:
    """
    Von Neumann entropy S(ρ) = -Tr[ρ log ρ].

    Parameters
    ----------
    state : array-like
        Statevector or density matrix.
    base : float
        Logarithm base. Default 2 (bits). Use np.e for nats.

    Returns
    -------
    float
        Entropy ≥ 0. For a pure state, S = 0.
    """
    rho = to_density_matrix(state)
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if base == 2.0:
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))
    return float(-np.sum(eigenvalues * np.log(eigenvalues)) / np.log(base))


def purity(state: Union[Statevector, DensityMatrix]) -> float:
    """
    Purity γ = Tr[ρ²] ∈ [1/d, 1].

    1.0 = pure state; < 1.0 indicates mixing due to noise.
    """
    rho = to_density_matrix(state)
    return float(np.real(np.trace(rho @ rho)))


def trace_distance(
    state_a: Union[Statevector, DensityMatrix],
    state_b: Union[Statevector, DensityMatrix],
) -> float:
    """
    Trace distance T(ρ, σ) = ½ Tr|ρ - σ| ∈ [0, 1].

    Operationally: the maximum probability of distinguishing ρ from σ
    with a single measurement. T = 0 ↔ identical states.
    """
    rho = to_density_matrix(state_a)
    sigma = to_density_matrix(state_b)
    diff = rho - sigma
    eigenvalues = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigenvalues)))


def pst_from_counts(counts: dict[str, int]) -> float:
    """
    Probability of Successful Teleportation (PST) from measurement counts.

    For the standard 3-qubit teleportation circuit the "success" outcomes
    are those where Bob's qubit matches Alice's intended state — derived
    from the post-correction measurement results.

    In the ideal protocol all four Bell-measurement outcomes {00, 01, 10, 11}
    occur with equal probability (each 25 %). PST close to 1.0 means Bob
    always receives the correct state regardless of the classical bits.

    Parameters
    ----------
    counts : dict[str, int]
        Raw histogram: {"00": 512, "01": 256, ...}.

    Returns
    -------
    float
        Estimated PST ∈ [0, 1].
    """
    if not counts:
        return 0.0
    total = sum(counts.values())
    # All outcomes are valid after classical correction — treat uniform
    # distribution as ideal (PST=1). Deviation from uniform indicates error.
    n_outcomes = len(counts)
    ideal_prob = 1.0 / max(n_outcomes, 1)
    total_variation = sum(
        abs(v / total - ideal_prob) for v in counts.values()
    )
    # PST = 1 - total variation distance from uniform
    pst = 1.0 - 0.5 * total_variation
    return float(np.clip(pst, 0.0, 1.0))


def average_fidelity_over_bloch_sphere(
    teleporter,
    n_samples: int = 100,
    noise_model=None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Estimate the average teleportation fidelity over randomly sampled
    input states uniformly distributed on the Bloch sphere.

    Uses the Haar-random distribution (uniform over |ψ⟩ on S²).

    Parameters
    ----------
    teleporter : QuantumTeleporter
        Any concrete teleporter instance.
    n_samples : int
        Number of random states to sample.
    noise_model : optional
        Noise model passed to teleporter.run().
    rng : np.random.Generator, optional
        For reproducibility.

    Returns
    -------
    tuple[float, float]
        (mean_fidelity, std_fidelity)
    """
    from src.core.teleport import QubitState

    if rng is None:
        rng = np.random.default_rng(42)

    fidelities = []
    for _ in range(n_samples):
        # Haar-random pure state on Bloch sphere
        theta = np.arccos(1 - 2 * rng.random())
        phi = 2 * np.pi * rng.random()
        state = QubitState(theta=theta, phi=phi)
        result = teleporter.run(state, noise_model=noise_model)
        fidelities.append(result.fidelity)

    arr = np.array(fidelities)
    return float(arr.mean()), float(arr.std())


# Private helpers

def _is_pure(rho: DensityMatrix, tol: float = 1e-6) -> bool:
    """Check if a density matrix is pure: Tr[ρ²] ≈ 1."""
    return abs(np.real(np.trace(rho @ rho)) - 1.0) < tol


def _matrix_sqrt(m: DensityMatrix) -> DensityMatrix:
    """
    Compute the matrix square root of a Hermitian positive-semidefinite
    matrix via eigendecomposition: √M = V diag(√λ) V†.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(m)
    eigenvalues = np.maximum(eigenvalues, 0.0)  # clip numerical negatives
    sqrt_diag = np.diag(np.sqrt(eigenvalues))
    return eigenvectors @ sqrt_diag @ eigenvectors.conj().T
