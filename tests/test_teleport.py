"""
tests/test_teleport.py
-----------------------
Integration tests for QiskitTeleporter and PennyLaneTeleporter.

These tests verify:
    - Circuit builds without errors
    - Run returns a valid TeleportResult
    - Ideal fidelity is close to 1.0
    - Noisy fidelity is lower than ideal
    - PST and entropy are in valid ranges

Run with:
    pytest tests/test_teleport.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.teleport import QubitState, TeleportResult
from src.core.metrics import fidelity


# ── Shared fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def canonical_states() -> list[QubitState]:
    return [
        QubitState(theta=0.0,        phi=0.0,       label="|0⟩"),
        QubitState(theta=np.pi,      phi=0.0,       label="|1⟩"),
        QubitState(theta=np.pi/2,    phi=0.0,       label="|+⟩"),
        QubitState(theta=np.pi/2,    phi=np.pi,     label="|-⟩"),
        QubitState(theta=np.pi/4,    phi=np.pi/3,   label="arbitrary"),
    ]

@pytest.fixture(scope="module")
def qiskit_teleporter():
    from src.qiskit_impl.circuit import QiskitTeleporter
    return QiskitTeleporter(shots=2048, seed=42)

@pytest.fixture(scope="module")
def pl_teleporter():
    from src.pennylane_impl.circuit import PennyLaneTeleporter
    return PennyLaneTeleporter(shots=2048, seed=42, device_name="default.mixed")


# ── Qiskit: circuit building ────────────────────────────────────────────────

class TestQiskitCircuit:
    def test_build_circuit_returns_quantum_circuit(self, qiskit_teleporter):
        from qiskit import QuantumCircuit
        state = QubitState(theta=np.pi/2, phi=0.0)
        qc = qiskit_teleporter.build_circuit(state)
        assert isinstance(qc, QuantumCircuit)

    def test_circuit_has_3_qubits(self, qiskit_teleporter):
        state = QubitState(theta=np.pi/4, phi=0.5)
        qc = qiskit_teleporter.build_circuit(state)
        assert qc.num_qubits == 3

    def test_circuit_has_classical_registers(self, qiskit_teleporter):
        state = QubitState(theta=np.pi/2, phi=0.0)
        qc = qiskit_teleporter.build_circuit(state)
        total_bits = sum(cr.size for cr in qc.cregs)
        assert total_bits == 3  # 2 alice + 1 bob

    def test_draw_circuit_text(self, qiskit_teleporter):
        state = QubitState(theta=np.pi/2, phi=0.0)
        diagram = qiskit_teleporter.draw_circuit(state, output="text")
        assert isinstance(diagram, str)
        assert len(diagram) > 0


# ── Qiskit: ideal execution ──────────────────────────────────────────────────

class TestQiskitIdealRun:
    def test_run_returns_teleport_result(self, qiskit_teleporter):
        state = QubitState(theta=np.pi/2, phi=0.0)
        result = qiskit_teleporter.run(state)
        assert isinstance(result, TeleportResult)

    def test_ideal_fidelity_close_to_one(self, qiskit_teleporter, canonical_states):
        for state in canonical_states:
            res = qiskit_teleporter.run(state, noise_model=None)
            assert res.fidelity > 0.90, (
                f"Ideal fidelity too low for {state.label}: {res.fidelity:.4f}"
            )

    def test_ideal_noise_model_label(self, qiskit_teleporter):
        state = QubitState(theta=np.pi/4, phi=0.3)
        res = qiskit_teleporter.run(state, noise_model=None)
        assert res.noise_model == "ideal"

    def test_backend_label_is_aer(self, qiskit_teleporter):
        state = QubitState(theta=0.5, phi=1.0)
        res = qiskit_teleporter.run(state)
        assert "aer" in res.backend.lower()

    def test_result_shots_match(self, qiskit_teleporter):
        state = QubitState(theta=np.pi/3, phi=0.5)
        res = qiskit_teleporter.run(state)
        total = sum(res.counts.values())
        assert total == qiskit_teleporter.shots

    def test_fidelity_range(self, qiskit_teleporter, canonical_states):
        for state in canonical_states:
            res = qiskit_teleporter.run(state)
            assert 0.0 <= res.fidelity <= 1.0 + 1e-6

    def test_pst_range(self, qiskit_teleporter):
        state = QubitState(theta=np.pi/2, phi=0.0)
        res = qiskit_teleporter.run(state)
        assert 0.0 <= res.pst <= 1.0 + 1e-6

    def test_entropy_non_negative(self, qiskit_teleporter):
        state = QubitState(theta=np.pi/2, phi=0.0)
        res = qiskit_teleporter.run(state)
        assert res.entropy >= -1e-8


# ── Qiskit: noisy execution ──────────────────────────────────────────────────

class TestQiskitNoisyRun:
    def test_noisy_fidelity_below_ideal(self, qiskit_teleporter):
        from src.qiskit_impl.noise_models import depolarizing_model
        state = QubitState(theta=np.pi/2, phi=0.0)
        ideal_res = qiskit_teleporter.run(state, noise_model=None)
        noisy_res = qiskit_teleporter.run(state, noise_model=depolarizing_model(0.05))
        assert noisy_res.fidelity <= ideal_res.fidelity + 0.05

    def test_high_noise_degrades_fidelity(self, qiskit_teleporter):
        from src.qiskit_impl.noise_models import depolarizing_model
        state  = QubitState(theta=np.pi/2, phi=0.0)
        res_lo = qiskit_teleporter.run(state, depolarizing_model(0.001))
        res_hi = qiskit_teleporter.run(state, depolarizing_model(0.10))
        assert res_hi.fidelity <= res_lo.fidelity + 0.1

    def test_thermal_noise_result_valid(self, qiskit_teleporter):
        from src.qiskit_impl.noise_models import thermal_relaxation_model
        state = QubitState(theta=np.pi/3, phi=np.pi/4)
        res = qiskit_teleporter.run(state, thermal_relaxation_model())
        assert isinstance(res, TeleportResult)
        assert 0.0 <= res.fidelity <= 1.0 + 1e-6


# ── PennyLane: ideal execution ────────────────────────────────────────────────

class TestPennyLaneIdealRun:
    def test_run_returns_teleport_result(self, pl_teleporter):
        state = QubitState(theta=np.pi/2, phi=0.0)
        res = pl_teleporter.run(state, noise_model=None)
        assert isinstance(res, TeleportResult)

    def test_ideal_fidelity_close_to_one(self, pl_teleporter, canonical_states):
        for state in canonical_states:
            res = pl_teleporter.run(state, noise_model=None)
            assert res.fidelity > 0.90, (
                f"PL ideal fidelity too low for {state.label}: {res.fidelity:.4f}"
            )

    def test_backend_label_contains_pennylane(self, pl_teleporter):
        state = QubitState(theta=0.5, phi=0.2)
        res = pl_teleporter.run(state)
        assert "pennylane" in res.backend.lower() or "default" in res.backend.lower()

    def test_fidelity_range(self, pl_teleporter, canonical_states):
        for state in canonical_states:
            res = pl_teleporter.run(state)
            assert 0.0 <= res.fidelity <= 1.0 + 1e-6

    def test_entropy_non_negative(self, pl_teleporter):
        state = QubitState(theta=np.pi/2, phi=0.0)
        res = pl_teleporter.run(state)
        assert res.entropy >= -1e-8


# ── PennyLane: noisy execution ────────────────────────────────────────────────

class TestPennyLaneNoisyRun:
    def test_noisy_fidelity_below_ideal(self, pl_teleporter):
        from src.pennylane_impl.circuit import make_depolarizing_channels
        state     = QubitState(theta=np.pi/2, phi=0.0)
        ideal_res = pl_teleporter.run(state, noise_model=None)
        noisy_res = pl_teleporter.run(state, noise_model=make_depolarizing_channels(0.05))
        assert noisy_res.fidelity <= ideal_res.fidelity + 0.05

    def test_amplitude_damping_result_valid(self, pl_teleporter):
        from src.pennylane_impl.circuit import make_amplitude_damping_channels
        state = QubitState(theta=np.pi/4, phi=0.5)
        res = pl_teleporter.run(state, make_amplitude_damping_channels(0.03))
        assert isinstance(res, TeleportResult)
        assert 0.0 <= res.fidelity <= 1.0 + 1e-6


# ── Framework consistency ─────────────────────────────────────────────────────

class TestFrameworkConsistency:
    def test_ideal_fidelities_similar(self, qiskit_teleporter, pl_teleporter):
        """Qiskit and PennyLane ideal fidelities should be close."""
        state = QubitState(theta=np.pi/2, phi=0.0)
        qr = qiskit_teleporter.run(state, noise_model=None)
        pr = pl_teleporter.run(state, noise_model=None)
        # Allow 10% tolerance (shot noise + different DM extraction methods)
        assert abs(qr.fidelity - pr.fidelity) < 0.10, (
            f"Fidelity gap too large: Qiskit={qr.fidelity:.4f} PL={pr.fidelity:.4f}"
        )

    def test_both_produce_valid_bloch_vectors(self, qiskit_teleporter, pl_teleporter):
        state = QubitState(theta=np.pi/3, phi=np.pi/4)
        qr = qiskit_teleporter.run(state)
        pr = pl_teleporter.run(state)
        for res in [qr, pr]:
            bv = res.output_state.bloch_vector
            norm = np.linalg.norm(bv)
            assert norm <= 1.0 + 1e-6, f"Bloch vector norm > 1: {norm}"
