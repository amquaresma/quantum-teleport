"""
pennylane_impl/circuit.py
--------------------------
PennyLane implementation of the quantum teleportation protocol.

Same 3-qubit protocol as the Qiskit version, built with PennyLane QNodes.
Supports:
    - Ideal simulation (default.qubit)
    - Noisy simulation (default.mixed with custom channels)
    - Gradient analysis via qml.grad / parameter-shift rule

Wire assignment:
    wire 0 → Alice's qubit (state to teleport)
    wire 1 → Alice's half of Bell pair
    wire 2 → Bob's qubit
"""

from __future__ import annotations

import numpy as np
from typing import Optional
import pennylane as qml

from src.core.teleport import QuantumTeleporter, QubitState, TeleportResult
from src.core.metrics import fidelity, von_neumann_entropy, pst_from_counts


class PennyLaneTeleporter(QuantumTeleporter):
    """
    Quantum teleportation via PennyLane.

    Parameters
    ----------
    shots : int
        Number of measurement shots per experiment.
        Pass None for analytic (exact) expectation values.
    seed : int, optional
        RNG seed for reproducibility.
    device_name : str
        PennyLane device to use.
        "default.qubit"  → ideal statevector simulation.
        "default.mixed"  → density-matrix simulation (supports noise).
    """

    def __init__(
        self,
        shots: Optional[int] = 8192,
        seed: Optional[int] = 42,
        device_name: str = "default.qubit",
    ) -> None:
        super().__init__(shots=shots, seed=seed)
        self.device_name = device_name
        self._dev: Optional[qml.Device] = None

    @property
    def framework(self) -> str:
        return "PennyLane"

    # ------------------------------------------------------------------
    # Device factory
    # ------------------------------------------------------------------

    def _make_device(self, noise_channels: Optional[list] = None) -> qml.Device:
        """
        Create the appropriate PennyLane device.

        For noisy simulations, always use "default.mixed".
        """
        device_name = self.device_name
        if noise_channels:
            device_name = "default.mixed"

        kwargs = {"wires": 3}
        if self.shots is not None:
            kwargs["shots"] = self.shots
        if self.seed is not None:
            kwargs["seed"] = self.seed

        return qml.device(device_name, **kwargs)

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def build_circuit(self, state: QubitState) -> callable:
        """
        Return a PennyLane QNode implementing the teleportation protocol.

        The returned function accepts an optional list of noise_channels
        (PennyLane channel operations) to apply after each gate when
        running on "default.mixed".

        Parameters
        ----------
        state : QubitState
            Input state to teleport.
        """
        dev = self._make_device()

        @qml.qnode(dev)
        def circuit():
            # Step 1: Prepare Alice's state on wire 0
            qml.RY(state.theta, wires=0)
            qml.RZ(state.phi, wires=0)

            # Step 2: Bell pair on wires (1, 2)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[1, 2])

            # Step 3: Bell measurement (Alice's side)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=0)

            # Step 4 & 5: Measure + conditional corrections
            # PennyLane mid-circuit measurement with conditional ops
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            qml.cond(m1 == 1, qml.PauliX)(wires=2)
            qml.cond(m0 == 1, qml.PauliZ)(wires=2)

            # Step 6: Return Bob's state
            return qml.state()

        self._circuit = circuit
        return circuit

    def build_density_matrix_circuit(
        self,
        state: QubitState,
        noise_channels: Optional[list] = None,
    ) -> callable:
        """
        Build a QNode on default.mixed that returns Bob's density matrix.
        Applies user-supplied noise channels after key operations.

        Parameters
        ----------
        state : QubitState
            Input state to teleport.
        noise_channels : list, optional
            List of (channel_class, kwargs) tuples to inject as noise.
            Example: [(qml.DepolarizingChannel, {"p": 0.01})]
        """
        dev = self._make_device(noise_channels=noise_channels)

        @qml.qnode(dev)
        def noisy_circuit():
            # State prep
            qml.RY(state.theta, wires=0)
            qml.RZ(state.phi, wires=0)
            _apply_noise(noise_channels, wires=[0])

            # Bell pair
            qml.Hadamard(wires=1)
            _apply_noise(noise_channels, wires=[1])
            qml.CNOT(wires=[1, 2])
            _apply_noise(noise_channels, wires=[1, 2])

            # Bell measurement
            qml.CNOT(wires=[0, 1])
            _apply_noise(noise_channels, wires=[0, 1])
            qml.Hadamard(wires=0)
            _apply_noise(noise_channels, wires=[0])

            # Mid-circuit measurements + corrections
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            qml.cond(m1 == 1, qml.PauliX)(wires=2)
            qml.cond(m0 == 1, qml.PauliZ)(wires=2)
            _apply_noise(noise_channels, wires=[2])

            # Bob's reduced density matrix
            return qml.density_matrix(wires=[2])

        return noisy_circuit

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        state: QubitState,
        noise_model: Optional[list] = None,
    ) -> TeleportResult:
        """
        Execute the teleportation protocol.

        Parameters
        ----------
        state : QubitState
            Input state to teleport.
        noise_model : list of (channel_class, kwargs), optional
            PennyLane noise channels. Pass None for ideal simulation.
            Example:
                [(qml.DepolarizingChannel, {"p": 0.01})]

        Returns
        -------
        TeleportResult
        """
        noise_label = "ideal" if noise_model is None else "custom_channels"
        if noise_model and hasattr(noise_model, "_label"):
            noise_label = noise_model._label

        # ── Get Bob's density matrix ────────────────────────────────────
        dm_circuit = self.build_density_matrix_circuit(state, noise_channels=noise_model)
        bob_dm = np.array(dm_circuit())

        # ── Get measurement counts via shot-based simulation ────────────
        bob_counts = self._sample_bob_counts(state, noise_model)

        # ── Reconstruct output state ─────────────────────────────────────
        eigenvalues, eigenvectors = np.linalg.eigh(bob_dm)
        dominant_sv = eigenvectors[:, np.argmax(eigenvalues)]
        output_state = QubitState.from_statevector(dominant_sv, label="Bob's qubit")

        # ── Metrics ─────────────────────────────────────────────────────
        f = fidelity(state.statevector, bob_dm)
        s = von_neumann_entropy(bob_dm)
        p = pst_from_counts(bob_counts)

        return TeleportResult(
            input_state=state,
            output_state=output_state,
            fidelity=f,
            pst=p,
            counts=bob_counts,
            entropy=s,
            noise_model=noise_label,
            backend=f"pennylane/{self.device_name}",
            shots=self.shots or 0,
            raw_metadata={
                "bob_density_matrix": bob_dm,
                "noise_channels": noise_model,
            },
        )

    def draw_circuit(self, state: QubitState, **kwargs) -> str:
        """Return a string representation of the PennyLane circuit."""
        circuit = self.build_density_matrix_circuit(state)
        return qml.draw(circuit)()

    # ------------------------------------------------------------------
    # Gradient analysis
    # ------------------------------------------------------------------

    def fidelity_vs_theta(
        self,
        phi: float = 0.0,
        theta_range: Optional[np.ndarray] = None,
        noise_model: Optional[list] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute teleportation fidelity as θ sweeps from 0 to π.

        Demonstrates that the protocol maintains fidelity across all
        input states (θ-independence is a key correctness check).

        Parameters
        ----------
        phi : float
            Fixed azimuthal angle.
        theta_range : np.ndarray, optional
            Array of θ values. Defaults to 50 points in [0, π].
        noise_model : list, optional
            PennyLane noise channels.

        Returns
        -------
        (thetas, fidelities) : tuple[np.ndarray, np.ndarray]
        """
        if theta_range is None:
            theta_range = np.linspace(0, np.pi, 50)

        fidelities = []
        for theta in theta_range:
            s = QubitState(theta=float(theta), phi=phi)
            result = self.run(s, noise_model=noise_model)
            fidelities.append(result.fidelity)

        return theta_range, np.array(fidelities)
            def gradient_of_fidelity(
        self,
        theta: float,
        phi: float = 0.0,
    ) -> tuple[float, float]:
        """
        Compute ∂F/∂θ and ∂F/∂φ using the parameter-shift rule.

        This is unique to PennyLane and demonstrates its automatic
        differentiation capabilities for quantum circuits.

        Returns
        -------
        (dF_dtheta, dF_dphi)
        """
        dev = qml.device("default.mixed", wires=3)

        @qml.qnode(dev, diff_method="parameter-shift")
        def fidelity_circuit(params):
            t, p = params[0], params[1]
            # Input state
            qml.RY(t, wires=0)
            qml.RZ(p, wires=0)
            # Bell pair
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[1, 2])
            # Bell measurement
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=0)
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            qml.cond(m1 == 1, qml.PauliX)(wires=2)
            qml.cond(m0 == 1, qml.PauliZ)(wires=2)
            # Expectation value used as fidelity proxy
            return qml.expval(qml.PauliZ(wires=2))

        grad_fn = qml.grad(fidelity_circuit)
        params = np.array([theta, phi], requires_grad=True)
        grads = grad_fn(params)
        return float(grads[0]), float(grads[1])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_bob_counts(
        self,
        state: QubitState,
        noise_channels: Optional[list],
    ) -> dict[str, int]:
        """Run a shot-based measurement on Bob's qubit and return counts."""
        shots = self.shots or 1024
        dev = qml.device(
            "default.mixed" if noise_channels else "default.qubit",
            wires=3,
            shots=shots,
            seed=self.seed,
        )

        @qml.qnode(dev)
        def measure_circuit():
            qml.RY(state.theta, wires=0)
            qml.RZ(state.phi, wires=0)
            _apply_noise(noise_channels, wires=[0])
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[1, 2])
            _apply_noise(noise_channels, wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=0)
            _apply_noise(noise_channels, wires=[0, 1])
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            qml.cond(m1 == 1, qml.PauliX)(wires=2)
            qml.cond(m0 == 1, qml.PauliZ)(wires=2)
            _apply_noise(noise_channels, wires=[2])
            return qml.sample(wires=[2])

        samples = measure_circuit()
        if samples.ndim == 0:
            samples = np.array([int(samples)])
        counts: dict[str, int] = {}
        for s in samples:
            key = str(int(s))
            counts[key] = counts.get(key, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Noise channel helpers for PennyLane
# ---------------------------------------------------------------------------

def _apply_noise(channels: Optional[list], wires: list[int]) -> None:
    """
    Apply a list of noise channels to the given wires.

    Parameters
    ----------
    channels : list of (channel_class, kwargs), optional
        Each entry is (qml.SomeChannel, {"param": value}).
    wires : list[int]
        Wires to apply each channel to.
    """
    if not channels:
        return
    for channel_cls, kwargs in channels:
        for wire in wires:
            channel_cls(**kwargs, wires=wire)


def make_depolarizing_channels(p: float) -> list:
    """
    Build a single-qubit depolarizing noise channel list.

    Parameters
    ----------
    p : float
        Depolarizing parameter ∈ [0, 3/4].

    Returns
    -------
    list of (channel_class, kwargs)
    """
    channels = [(qml.DepolarizingChannel, {"p": p})]
    channels._label = f"pl_depolarizing(p={p:.4f})"
    return channels


def make_amplitude_damping_channels(gamma: float) -> list:
    """
    Build an amplitude damping channel list (models T1 decay).

    Parameters
    ----------
    gamma : float
        Damping parameter ∈ [0, 1]. γ = 1 - exp(-t/T1).

    Returns
    -------
    list of (channel_class, kwargs)
    """
    channels = [(qml.AmplitudeDamping, {"gamma": gamma})]
    channels._label = f"pl_amplitude_damping(γ={gamma:.4f})"
    return channels


def make_phase_damping_channels(gamma: float) -> list:
    """
    Build a phase damping channel list (models T2 dephasing).

    Parameters
    ----------
    gamma : float
        Dephasing parameter ∈ [0, 1].

    Returns
    -------
    list of (channel_class, kwargs)
    """
    channels = [(qml.PhaseDamping, {"gamma": gamma})]
    channels._label = f"pl_phase_damping(γ={gamma:.4f})"
    return channels
