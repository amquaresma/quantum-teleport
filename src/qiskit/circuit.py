"""
qiskit_impl/circuit.py
----------------------
Qiskit implementation of the quantum teleportation protocol.

Circuit structure (3 qubits):
    q0 = Alice's qubit (state to teleport)
    q1 = Alice's half of the Bell pair
    q2 = Bob's qubit (receives teleported state)

Steps:
    1. Prepare |ψ⟩ on q0 via Ry(θ) Rz(φ)
    2. Create Bell pair on (q1, q2): H on q1, CNOT q1→q2
    3. Bell measurement on Alice's side (q0, q1): CNOT q0→q1, H on q0
    4. Measure q0, q1 → classical bits c0, c1
    5. Classical corrections on Bob:  if c1=1 → X on q2
                                      if c0=1 → Z on q2
    6. Optional: measure q2 to get Bob's state histogram
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import Initialize
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from src.core.teleport import QuantumTeleporter, QubitState, TeleportResult
from src.core.metrics import fidelity, von_neumann_entropy, pst_from_counts, to_density_matrix


class QiskitTeleporter(QuantumTeleporter):
    """
    Quantum teleportation via Qiskit + AerSimulator.

    Parameters
    ----------
    shots : int
        Number of measurement shots per experiment.
    seed : int, optional
        RNG seed for reproducibility.
    use_statevector : bool
        If True, use statevector simulator for exact fidelity.
        If False, use shot-based simulation (more realistic).
    """

    def __init__(
        self,
        shots: int = 8192,
        seed: Optional[int] = 42,
        use_statevector: bool = False,
    ) -> None:
        super().__init__(shots=shots, seed=seed)
        self.use_statevector = use_statevector
        self._circuit: Optional[QuantumCircuit] = None

    @property
    def framework(self) -> str:
        return "Qiskit"

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def build_circuit(self, state: QubitState) -> QuantumCircuit:
        """
        Build the full 3-qubit teleportation circuit.

        Returns a QuantumCircuit with:
          - q0, q1, q2  (quantum registers)
          - c_alice (2 classical bits for Bell measurement)
          - c_bob   (1 classical bit for Bob's final measurement)
        """
        qr = QuantumRegister(3, "q")
        c_alice = ClassicalRegister(2, "alice")
        c_bob = ClassicalRegister(1, "bob")
        qc = QuantumCircuit(qr, c_alice, c_bob)

        # ── Step 1: Prepare Alice's input state on q0 ──────────────────
        qc.ry(state.theta, qr[0])
        qc.rz(state.phi, qr[0])
        qc.barrier(label="state prep")

        # ── Step 2: Create Bell pair (q1, q2) ──────────────────────────
        qc.h(qr[1])
        qc.cx(qr[1], qr[2])
        qc.barrier(label="Bell pair")

        # ── Step 3: Bell measurement on Alice's side (q0, q1) ──────────
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])
        qc.barrier(label="Bell measure")

        # ── Step 4: Measure Alice's qubits ─────────────────────────────
        qc.measure(qr[0], c_alice[0])
        qc.measure(qr[1], c_alice[1])

        # ── Step 5: Classical corrections on Bob ───────────────────────
        with qc.if_test((c_alice[1], 1)):
            qc.x(qr[2])
        with qc.if_test((c_alice[0], 1)):
            qc.z(qr[2])
        qc.barrier(label="correction")

        # ── Step 6: Measure Bob's qubit ────────────────────────────────
        qc.measure(qr[2], c_bob[0])

        self._circuit = qc
        return qc

    def build_circuit_no_measure(self, state: QubitState) -> QuantumCircuit:
        """
        Build the circuit WITHOUT final measurement on Bob's qubit.
        Used for statevector / density-matrix extraction.
        """
        qr = QuantumRegister(3, "q")
        c_alice = ClassicalRegister(2, "alice")
        qc = QuantumCircuit(qr, c_alice)

        qc.ry(state.theta, qr[0])
        qc.rz(state.phi, qr[0])
        qc.barrier()
        qc.h(qr[1])
        qc.cx(qr[1], qr[2])
        qc.barrier()
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])
        qc.barrier()
        qc.measure(qr[0], c_alice[0])
        qc.measure(qr[1], c_alice[1])
        with qc.if_test((c_alice[1], 1)):
            qc.x(qr[2])
        with qc.if_test((c_alice[0], 1)):
            qc.z(qr[2])
        return qc

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        state: QubitState,
        noise_model: Optional[NoiseModel] = None,
    ) -> TeleportResult:
        """
        Execute teleportation and return TeleportResult.

        Parameters
        ----------
        state : QubitState
            Input state to teleport.
        noise_model : NoiseModel, optional
            Qiskit Aer noise model. Pass None for ideal simulation.
        """
        noise_label = "ideal" if noise_model is None else getattr(noise_model, "_label", "custom")

        # ── Run shot-based simulation ───────────────────────────────────
        qc = self.build_circuit(state)
        backend = AerSimulator(
            noise_model=noise_model,
            seed_simulator=self.seed,
        )
        tqc = transpile(qc, backend)
        job = backend.run(tqc, shots=self.shots)
        result = job.result()
        counts_raw = result.get_counts()

        # Extract only Bob's register counts (last classical bit)
        bob_counts = self._extract_bob_counts(counts_raw)

        # ── Get Bob's output state via density matrix simulation ────────
        output_dm = self._get_bob_density_matrix(state, noise_model)
        output_sv = _dm_to_bloch(output_dm)
        output_state = QubitState.from_statevector(output_sv, label="Bob's qubit")

        # ── Compute metrics ─────────────────────────────────────────────
        input_sv = state.statevector
        f = fidelity(input_sv, output_dm)
        s = von_neumann_entropy(output_dm)
        p = pst_from_counts(bob_counts)

        return TeleportResult(
            input_state=state,
            output_state=output_state,
            fidelity=f,
            pst=p,
            counts=bob_counts,
            entropy=s,
            noise_model=noise_label,
            backend="aer_simulator",
            shots=self.shots,
            raw_metadata={
                "full_counts": counts_raw,
                "noise_model_obj": noise_model,
            },
        )

    def draw_circuit(self, state: QubitState, output: str = "text", **kwargs) -> object:
        """
        Draw the teleportation circuit.

        Parameters
        ----------
        state : QubitState
            Input state (needed to build the circuit first).
        output : str
            Qiskit draw output format: "text", "mpl", "latex_source".
        """
        qc = self.build_circuit(state)
        return qc.draw(output=output, **kwargs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_bob_density_matrix(
        self,
        state: QubitState,
        noise_model: Optional[NoiseModel],
    ) -> np.ndarray:
        """
        Obtain Bob's reduced density matrix by running the circuit
        with the density_matrix simulator and tracing out Alice's qubits.
        """
        qc_nm = self.build_circuit_no_measure(state)

        backend = AerSimulator(
            method="density_matrix",
            noise_model=noise_model,
            seed_simulator=self.seed,
        )
        # Save the full density matrix at the end
        qc_save = qc_nm.copy()
        qc_save.save_density_matrix()

        tqc = transpile(qc_save, backend)
        job = backend.run(tqc, shots=1)
        result = job.result()

        full_dm = result.data()["density_matrix"].data  # (8×8) for 3 qubits

        # Partial trace: keep only Bob's qubit (index 2)
        # qiskit ordering is little-endian: qubit 0 is the rightmost
        # Bob is q[2] → leftmost in tensor product
        bob_dm = _partial_trace_keep_last(full_dm, n_qubits=3)
        return bob_dm

    @staticmethod
    def _extract_bob_counts(counts: dict) -> dict[str, int]:
        """
        Extract Bob's bit from the combined measurement result string.
        Qiskit formats counts as "c_bob c_alice": e.g. "1 01".
        """
        bob_counts: dict[str, int] = {}
        for bitstring, count in counts.items():
            # Split on space; first token is the last-measured register (bob)
            parts = bitstring.split()
            bob_bit = parts[0] if parts else bitstring[-1]
            bob_counts[bob_bit] = bob_counts.get(bob_bit, 0) + count
        return bob_counts


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _partial_trace_keep_last(dm: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Partial trace over all qubits except the last one (Bob's qubit).
    Works for any n_qubits. Assumes all qubits are 2-dimensional.
    """
    d = 2 ** n_qubits
    assert dm.shape == (d, d), f"Expected ({d},{d}), got {dm.shape}"

    # Trace out qubits 0 … n-2, keeping qubit n-1
    rho = dm
    for _ in range(n_qubits - 1):
        current_d = rho.shape[0]
        d_keep = current_d // 2
        rho = rho.reshape(2, d_keep, 2, d_keep)
        rho = rho[0, :, 0, :] + rho[1, :, 1, :]
    return rho


def _dm_to_bloch(dm: np.ndarray) -> np.ndarray:
    """
    Extract the statevector of the dominant eigenvector of a density matrix.
    For pure states this is exact; for mixed states it returns the
    closest pure state (maximum eigenvalue eigenvector).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(dm)
    dominant = eigenvectors[:, np.argmax(eigenvalues)]
    return dominant / np.linalg.norm(dominant)
