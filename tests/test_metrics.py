"""
tests/test_metrics.py
---------------------
Unit tests for core/metrics.py

Run with:
    pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest
from src.core.metrics import (
    fidelity,
    von_neumann_entropy,
    purity,
    trace_distance,
    pst_from_counts,
    to_density_matrix,
    _is_pure,
)
from src.core.teleport import QubitState


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def ket_0():
    return np.array([1.0, 0.0], dtype=complex)

@pytest.fixture
def ket_1():
    return np.array([0.0, 1.0], dtype=complex)

@pytest.fixture
def ket_plus():
    return np.array([1, 1], dtype=complex) / np.sqrt(2)

@pytest.fixture
def ket_minus():
    return np.array([1, -1], dtype=complex) / np.sqrt(2)

@pytest.fixture
def maximally_mixed():
    return np.eye(2, dtype=complex) / 2


# ── QubitState ─────────────────────────────────────────────────────────────

class TestQubitState:
    def test_ket_zero_statevector(self):
        s = QubitState(theta=0.0, phi=0.0)
        sv = s.statevector
        assert np.allclose(sv, [1.0, 0.0], atol=1e-8)

    def test_ket_one_statevector(self):
        s = QubitState(theta=np.pi, phi=0.0)
        sv = s.statevector
        assert np.allclose(np.abs(sv), [0.0, 1.0], atol=1e-8)

    def test_ket_plus_statevector(self):
        s = QubitState(theta=np.pi / 2, phi=0.0)
        sv = s.statevector
        expected = np.array([1, 1]) / np.sqrt(2)
        assert np.allclose(np.abs(sv), np.abs(expected), atol=1e-8)

    def test_bloch_vector_z_for_ket0(self):
        s = QubitState(theta=0.0, phi=0.0)
        bv = s.bloch_vector
        assert np.allclose(bv, [0, 0, 1], atol=1e-8)

    def test_bloch_vector_neg_z_for_ket1(self):
        s = QubitState(theta=np.pi, phi=0.0)
        bv = s.bloch_vector
        assert np.allclose(bv, [0, 0, -1], atol=1e-8)

    def test_bloch_vector_x_for_ket_plus(self):
        s = QubitState(theta=np.pi / 2, phi=0.0)
        bv = s.bloch_vector
        assert np.allclose(bv, [1, 0, 0], atol=1e-8)

    def test_from_statevector_roundtrip(self):
        original = QubitState(theta=0.7, phi=1.2)
        sv = original.statevector
        reconstructed = QubitState.from_statevector(sv)
        assert np.allclose(reconstructed.theta, original.theta, atol=1e-6)
        assert np.allclose(reconstructed.phi,   original.phi,   atol=1e-6)

    def test_normalization(self):
        for theta in np.linspace(0, np.pi, 10):
            for phi in np.linspace(0, 2 * np.pi, 10):
                s = QubitState(theta=theta, phi=phi)
                assert abs(np.linalg.norm(s.statevector) - 1.0) < 1e-10
                assert abs(np.linalg.norm(s.bloch_vector) - 1.0) < 1e-10


# ── Fidelity ───────────────────────────────────────────────────────────────

class TestFidelity:
    def test_identical_pure_states(self, ket_0):
        assert fidelity(ket_0, ket_0) == pytest.approx(1.0, abs=1e-8)

    def test_orthogonal_pure_states(self, ket_0, ket_1):
        assert fidelity(ket_0, ket_1) == pytest.approx(0.0, abs=1e-8)

    def test_plus_minus_orthogonal(self, ket_plus, ket_minus):
        assert fidelity(ket_plus, ket_minus) == pytest.approx(0.0, abs=1e-8)

    def test_overlap_ket0_ket_plus(self, ket_0, ket_plus):
        # |⟨0|+⟩|² = 0.5
        assert fidelity(ket_0, ket_plus) == pytest.approx(0.5, abs=1e-6)

    def test_fidelity_symmetric(self, ket_0, ket_plus):
        assert fidelity(ket_0, ket_plus) == pytest.approx(
            fidelity(ket_plus, ket_0), abs=1e-8
        )

    def test_fidelity_mixed_state_bounds(self, ket_0, maximally_mixed):
        f = fidelity(ket_0, maximally_mixed)
        assert 0.0 <= f <= 1.0

    def test_fidelity_pure_vs_own_dm(self, ket_plus):
        dm = to_density_matrix(ket_plus)
        assert fidelity(ket_plus, dm) == pytest.approx(1.0, abs=1e-8)

    def test_fidelity_range(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            sv1 = rng.normal(size=2) + 1j * rng.normal(size=2)
            sv2 = rng.normal(size=2) + 1j * rng.normal(size=2)
            sv1 /= np.linalg.norm(sv1)
            sv2 /= np.linalg.norm(sv2)
            f = fidelity(sv1, sv2)
            assert 0.0 <= f <= 1.0 + 1e-8


# ── Von Neumann entropy ────────────────────────────────────────────────────

class TestVonNeumannEntropy:
    def test_pure_state_zero_entropy(self, ket_0):
        assert von_neumann_entropy(ket_0) == pytest.approx(0.0, abs=1e-8)

    def test_maximally_mixed_one_bit(self, maximally_mixed):
        # S(I/2) = 1 bit
        assert von_neumann_entropy(maximally_mixed) == pytest.approx(1.0, abs=1e-8)

    def test_entropy_non_negative(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            sv = rng.normal(size=2) + 1j * rng.normal(size=2)
            sv /= np.linalg.norm(sv)
            assert von_neumann_entropy(sv) >= -1e-10

    def test_entropy_base_e(self, maximally_mixed):
        s_nats = von_neumann_entropy(maximally_mixed, base=np.e)
        assert s_nats == pytest.approx(np.log(2), abs=1e-8)


# ── Purity ─────────────────────────────────────────────────────────────────

class TestPurity:
    def test_pure_state_purity_one(self, ket_plus):
        assert purity(ket_plus) == pytest.approx(1.0, abs=1e-8)

    def test_maximally_mixed_purity_half(self, maximally_mixed):
        assert purity(maximally_mixed) == pytest.approx(0.5, abs=1e-8)


# ── Trace distance ─────────────────────────────────────────────────────────

class TestTraceDistance:
    def test_identical_states_zero(self, ket_0):
        assert trace_distance(ket_0, ket_0) == pytest.approx(0.0, abs=1e-8)

    def test_orthogonal_states_one(self, ket_0, ket_1):
        assert trace_distance(ket_0, ket_1) == pytest.approx(1.0, abs=1e-8)

    def test_range(self, ket_0, ket_plus, maximally_mixed):
        for a, b in [(ket_0, ket_plus), (ket_0, maximally_mixed)]:
            td = trace_distance(a, b)
            assert 0.0 <= td <= 1.0 + 1e-8


# ── PST from counts ────────────────────────────────────────────────────────

class TestPSTFromCounts:
    def test_uniform_counts_pst_one(self):
        counts = {"0": 500, "1": 500}
        pst = pst_from_counts(counts)
        assert pst == pytest.approx(1.0, abs=1e-8)

    def test_all_zero_pst_zero(self):
        counts = {"0": 1000, "1": 0}
        pst = pst_from_counts(counts)
        assert pst < 0.5

    def test_empty_counts(self):
        assert pst_from_counts({}) == 0.0

    def test_pst_range(self):
        import random
        random.seed(42)
        for _ in range(20):
            a = random.randint(1, 1000)
            b = random.randint(1, 1000)
            pst = pst_from_counts({"0": a, "1": b})
            assert 0.0 <= pst <= 1.0 + 1e-8
