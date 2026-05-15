"""
core/teleport.py
----------------
Abstract base class for the quantum teleportation protocol.
Both Qiskit and PennyLane implementations inherit from this interface,
allowing apples-to-apples comparison in experiments/compare.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# Data containers


@dataclass
class QubitState:
    """
    Represents a single-qubit pure state on the Bloch sphere.

    |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩

    Attributes
    ----------
    theta : float
        Polar angle in [0, π].
    phi : float
        Azimuthal angle in [0, 2π].
    label : str
        Human-readable name (e.g. "Alice's qubit").
    """
    theta: float = 0.0
    phi: float = 0.0
    label: str = "qubit"

    @property
    def statevector(self) -> np.ndarray:
        """Return the 2-element complex state vector."""
        alpha = np.cos(self.theta / 2)
        beta = np.exp(1j * self.phi) * np.sin(self.theta / 2)
        return np.array([alpha, beta], dtype=complex)

    @property
    def bloch_vector(self) -> np.ndarray:
        """Return the [x, y, z] Bloch vector."""
        x = np.sin(self.theta) * np.cos(self.phi)
        y = np.sin(self.theta) * np.sin(self.phi)
        z = np.cos(self.theta)
        return np.array([x, y, z])

    @classmethod
    def from_statevector(cls, sv: np.ndarray, label: str = "qubit") -> "QubitState":
        """Reconstruct a QubitState from an arbitrary 2-element statevector."""
        sv = sv / np.linalg.norm(sv)
        alpha, beta = sv[0], sv[1]
        theta = 2 * np.arccos(np.clip(np.abs(alpha), 0, 1))
        phi = np.angle(beta) - np.angle(alpha) if np.abs(beta) > 1e-10 else 0.0
        return cls(theta=float(theta), phi=float(phi % (2 * np.pi)), label=label)

    def __repr__(self) -> str:
        return (
            f"QubitState(θ={self.theta:.4f} rad, φ={self.phi:.4f} rad, "
            f"label='{self.label}')"
        )


@dataclass
class TeleportResult:
    """
    Full result bundle returned by every teleporter implementation.

    Attributes
    ----------
    input_state : QubitState
        The state Alice wanted to teleport.
    output_state : QubitState
        Bob's reconstructed state (ideal or noisy).
    fidelity : float
        Quantum fidelity F(ρ_in, ρ_out) ∈ [0, 1].
    pst : float
        Probability of successful teleportation (PST).
    counts : dict[str, int]
        Raw measurement histogram from the simulation.
    entropy : float
        Von Neumann entropy of the output state.
    noise_model : str
        Name of the noise model used, or "ideal".
    backend : str
        Backend identifier (e.g. "aer_simulator", "default.qubit").
    shots : int
        Number of shots used.
    raw_metadata : dict
        Backend-specific extra info (T1, T2, gate errors, etc.).
    """
    input_state: QubitState
    output_state: QubitState
    fidelity: float
    pst: float
    counts: dict = field(default_factory=dict)
    entropy: float = 0.0
    noise_model: str = "ideal"
    backend: str = "unknown"
    shots: int = 1024
    raw_metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"{'─' * 44}",
            f"  Backend      : {self.backend}",
            f"  Noise model  : {self.noise_model}",
            f"  Shots        : {self.shots}",
            f"  Fidelity     : {self.fidelity:.4f}",
            f"  PST          : {self.pst:.4f}",
            f"  Entropy S(ρ) : {self.entropy:.4f}",
            f"  Input state  : θ={self.input_state.theta:.3f}, φ={self.input_state.phi:.3f}",
            f"  Output state : θ={self.output_state.theta:.3f}, φ={self.output_state.phi:.3f}",
            f"{'─' * 44}",
        ]
        return "\n".join(lines)


# Abstract base class

class QuantumTeleporter(ABC):
    """
    Abstract interface for quantum teleportation protocol implementations.

    Subclasses must implement:
        - build_circuit()  → constructs the teleportation circuit
        - run()            → executes and returns a TeleportResult
        - draw_circuit()   → returns a string/figure representation

    Both QiskitTeleporter and PennyLaneTeleporter conform to this interface.
    """

    def __init__(
        self,
        shots: int = 8192,
        seed: Optional[int] = 42,
    ) -> None:
        self.shots = shots
        self.seed = seed
        self._circuit = None

    # Abstract interface

    @abstractmethod
    def build_circuit(self, state: QubitState) -> None:
        """
        Construct the teleportation circuit for the given input state.

        Parameters
        ----------
        state : QubitState
            The qubit Alice wants to teleport to Bob.
        """

    @abstractmethod
    def run(
        self,
        state: QubitState,
        noise_model: Optional[object] = None,
    ) -> TeleportResult:
        """
        Execute the teleportation protocol and return full results.

        Parameters
        ----------
        state : QubitState
            The qubit to teleport.
        noise_model : optional
            Framework-specific noise model (e.g. qiskit_aer NoiseModel).
            Pass None for an ideal (noiseless) simulation.

        Returns
        -------
        TeleportResult
        """

    @abstractmethod
    def draw_circuit(self, state: QubitState, **kwargs) -> object:
        """Return a visual or textual representation of the circuit."""

    # Shared helpers (available to all subclasses)
    

    @property
    def framework(self) -> str:
        """Return the framework name (overridden in subclasses)."""
        return self.__class__.__name__

    def sweep_noise(
        self,
        state: QubitState,
        noise_levels: list[float],
        noise_factory,
    ) -> list[TeleportResult]:
        """
        Run the protocol across multiple noise levels.

        Parameters
        ----------
        state : QubitState
            Input state to teleport at every noise level.
        noise_levels : list[float]
            Iterable of error-rate values (0 = ideal).
        noise_factory : callable
            Function (error_rate: float) → noise_model.
            Pass None-returning lambda for the ideal case.

        Returns
        -------
        list[TeleportResult]
            One result per noise level, in the same order.
        """
        results = []
        for level in noise_levels:
            nm = noise_factory(level) if level > 0 else None
            results.append(self.run(state, noise_model=nm))
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shots={self.shots}, seed={self.seed})"
