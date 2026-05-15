"""
qiskit_impl/simulator.py
-------------------------
High-level AerSimulator wrapper that accepts NoiseConfig objects
and builds the appropriate Qiskit noise model automatically.

Decouples experiment scripts from low-level Qiskit noise API.

Usage
-----
    from src.qiskit_impl.simulator import QiskitSimulator
    from src.core.noise_models import NoiseConfig

    sim = QiskitSimulator(shots=4096, seed=42)
    result = sim.run(circuit, noise=NoiseConfig.depolarizing(p=0.01))
    counts = result.get_counts()
"""

from __future__ import annotations

from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from src.core.noise_models import NoiseConfig, NoiseLevel
from src.qiskit_impl.noise_models import (
    depolarizing_model,
    depolarizing_readout_model,
    thermal_relaxation_model,
)


class QiskitSimulator:
    """
    Thin wrapper around AerSimulator with automatic noise model injection.

    Parameters
    ----------
    shots : int
        Default number of shots for each run.
    seed : int, optional
        Simulator RNG seed for reproducibility.
    method : str
        AerSimulator method: "automatic", "statevector", "density_matrix".
    """

    def __init__(
        self,
        shots: int = 4096,
        seed: Optional[int] = 42,
        method: str = "automatic",
    ) -> None:
        self.shots = shots
        self.seed = seed
        self.method = method

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        circuit: QuantumCircuit,
        noise: Optional[NoiseConfig] = None,
        shots: Optional[int] = None,
    ):
        """
        Transpile and execute a QuantumCircuit.

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit to run.
        noise : NoiseConfig, optional
            Noise configuration. Pass None or NoiseConfig.ideal() for
            noiseless simulation.
        shots : int, optional
            Override the default shot count for this run.

        Returns
        -------
        qiskit.result.Result
        """
        nm = self._build_noise_model(noise)
        backend = AerSimulator(
            method=self.method,
            noise_model=nm,
            seed_simulator=self.seed,
        )
        n_shots = shots or self.shots
        tqc = transpile(circuit, backend)
        return backend.run(tqc, shots=n_shots).result()

    def run_density_matrix(
        self,
        circuit: QuantumCircuit,
        noise: Optional[NoiseConfig] = None,
    ):
        """
        Execute circuit with the density_matrix simulator method.
        Required for extracting mixed-state density matrices under noise.

        Returns
        -------
        qiskit.result.Result
        """
        nm = self._build_noise_model(noise)
        backend = AerSimulator(
            method="density_matrix",
            noise_model=nm,
            seed_simulator=self.seed,
        )
        tqc = transpile(circuit, backend)
        return backend.run(tqc, shots=1).result()

    # ------------------------------------------------------------------
    # Noise model factory
    # ------------------------------------------------------------------

    def _build_noise_model(
        self,
        cfg: Optional[NoiseConfig],
    ) -> Optional[NoiseModel]:
        """Convert a NoiseConfig into a Qiskit NoiseModel."""
        if cfg is None or cfg.level == NoiseLevel.IDEAL:
            return None

        if cfg.level == NoiseLevel.DEPOLARIZING:
            return depolarizing_model(
                error_rate=cfg.error_rate,
                two_qubit_factor=cfg.two_qubit_factor,
            )

        if cfg.level == NoiseLevel.DEPOLARIZING_READOUT:
            return depolarizing_readout_model(
                gate_error=cfg.error_rate,
                readout_error=cfg.readout_error,
                two_qubit_factor=cfg.two_qubit_factor,
            )

        if cfg.level == NoiseLevel.THERMAL:
            return thermal_relaxation_model(
                T1_us=cfg.T1_us,
                T2_us=cfg.T2_us,
                gate_time_1q_ns=cfg.gate_time_1q_ns,
                gate_time_2q_ns=cfg.gate_time_2q_ns,
                readout_error=cfg.readout_error,
            )

        raise ValueError(
            f"NoiseLevel '{cfg.level}' is not supported by QiskitSimulator. "
            f"Supported: IDEAL, DEPOLARIZING, DEPOLARIZING_READOUT, THERMAL."
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_counts(
        self,
        circuit: QuantumCircuit,
        noise: Optional[NoiseConfig] = None,
        shots: Optional[int] = None,
    ) -> dict[str, int]:
        """Run and return counts dict directly."""
        result = self.run(circuit, noise=noise, shots=shots)
        return result.get_counts()

    def __repr__(self) -> str:
        return (
            f"QiskitSimulator(shots={self.shots}, "
            f"seed={self.seed}, method='{self.method}')"
        )
