"""
qiskit_impl/noise_models.py
----------------------------
Ready-made noise model factory functions for Qiskit AerSimulator.

Three escalating levels of realism:

    Level 1 — Depolarizing
        Simple Pauli channel applied after each gate.
        Good for a quick sanity check on decoherence effects.

    Level 2 — Depolarizing + Readout error
        Adds classical bit-flip probability during measurement.
        Models imperfect detectors.

    Level 3 — Thermal relaxation (T1 / T2)
        Uses real IBM hardware T1/T2 values.
        Most physically accurate model available without real hardware.

Usage
-----
    from src.qiskit_impl.noise_models import (
        depolarizing_model,
        depolarizing_readout_model,
        thermal_relaxation_model,
        NOISE_REGISTRY,
    )

    nm = depolarizing_model(error_rate=0.01)
    result = teleporter.run(state, noise_model=nm)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)
import numpy as np


# Level 1 — Depolarizing channel

def depolarizing_model(
    error_rate: float = 0.01,
    two_qubit_factor: float = 10.0,
) -> NoiseModel:
    """
    Uniform depolarizing noise on all single- and two-qubit gates.

    Parameters
    ----------
    error_rate : float
        Single-qubit gate error probability ∈ [0, 1/2].
    two_qubit_factor : float
        Multiplier for two-qubit gate error (CNOT is noisier than single-qubit).
        Default 10× is consistent with typical superconducting hardware.

    Returns
    -------
    NoiseModel
    """
    nm = NoiseModel()
    nm._label = f"depolarizing(p={error_rate:.4f})"

    # Single-qubit gates
    error_1q = depolarizing_error(error_rate, 1)
    nm.add_all_qubit_quantum_error(error_1q, ["h", "x", "z", "ry", "rz", "id"])

    # Two-qubit gates (CNOT)
    error_2q = depolarizing_error(
        min(error_rate * two_qubit_factor, 0.99), 2
    )
    nm.add_all_qubit_quantum_error(error_2q, ["cx"])

    return nm


# Level 2 — Depolarizing + Readout error

def depolarizing_readout_model(
    gate_error: float = 0.005,
    readout_error: float = 0.02,
    two_qubit_factor: float = 10.0,
) -> NoiseModel:
    """
    Depolarizing gate noise plus classical measurement (readout) errors.

    Parameters
    ----------
    gate_error : float
        Single-qubit gate error probability.
    readout_error : float
        Probability of flipping a measured bit: P(1|0) = P(0|1).
    two_qubit_factor : float
        CNOT error multiplier.

    Returns
    -------
    NoiseModel
    """
    nm = depolarizing_model(error_rate=gate_error, two_qubit_factor=two_qubit_factor)
    nm._label = (
        f"depolarizing+readout(gate={gate_error:.4f}, ro={readout_error:.4f})"
    )

    # Symmetric readout error matrix [[P(0|0), P(1|0)], [P(0|1), P(1|1)]]
    ro_matrix = [
        [1 - readout_error, readout_error],
        [readout_error, 1 - readout_error],
    ]
    ro_error = ReadoutError(ro_matrix)
    nm.add_all_qubit_readout_error(ro_error)

    return nm


# Level 3 — Thermal relaxation (T1 / T2)

# Representative values for IBM 5-qubit devices (ibm_nairobi, ~2023)
_IBM_DEFAULT_PARAMS = {
    "T1_us": 100.0,   # T1 relaxation time in microseconds
    "T2_us": 80.0,    # T2 dephasing time in microseconds
    "gate_time_1q_ns": 50.0,   # single-qubit gate time in nanoseconds
    "gate_time_2q_ns": 300.0,  # CNOT gate time in nanoseconds
}


def thermal_relaxation_model(
    T1_us: float = _IBM_DEFAULT_PARAMS["T1_us"],
    T2_us: float = _IBM_DEFAULT_PARAMS["T2_us"],
    gate_time_1q_ns: float = _IBM_DEFAULT_PARAMS["gate_time_1q_ns"],
    gate_time_2q_ns: float = _IBM_DEFAULT_PARAMS["gate_time_2q_ns"],
    readout_error: float = 0.02,
    scale: float = 1.0,
) -> NoiseModel:
    """
    Thermal relaxation noise using T1/T2 hardware parameters.

    Models amplitude damping (T1) and pure dephasing (T2) as they occur
    on real superconducting qubit hardware.

    Parameters
    ----------
    T1_us : float
        Energy relaxation time in microseconds.
    T2_us : float
        Dephasing time in microseconds. Must satisfy T2 ≤ 2*T1.
    gate_time_1q_ns : float
        Duration of a single-qubit gate in nanoseconds.
    gate_time_2q_ns : float
        Duration of a CNOT gate in nanoseconds.
    readout_error : float
        Symmetric readout bit-flip probability.
    scale : float
        Multiply all gate times by this factor to simulate
        longer-running (noisier) circuits. scale=1 = realistic,
        scale=10 = ten times longer (more decoherence).

    Returns
    -------
    NoiseModel
    """
    # Clamp T2 to physical limit
    T2_us = min(T2_us, 2 * T1_us)

    nm = NoiseModel()
    nm._label = (
        f"thermal(T1={T1_us}µs, T2={T2_us}µs, scale={scale})"
    )

    t1_ns = T1_us * 1_000
    t2_ns = T2_us * 1_000
    gt_1q = gate_time_1q_ns * scale
    gt_2q = gate_time_2q_ns * scale

    # Single-qubit thermal error
    err_1q = thermal_relaxation_error(t1_ns, t2_ns, gt_1q)
    nm.add_all_qubit_quantum_error(
        err_1q, ["h", "x", "z", "ry", "rz", "id"]
    )

    # Two-qubit thermal error (applied to each qubit in the CNOT)
    err_2q = thermal_relaxation_error(t1_ns, t2_ns, gt_2q)
    err_cx = err_2q.expand(err_2q)
    nm.add_all_qubit_quantum_error(err_cx, ["cx"])

    # Readout error
    if readout_error > 0:
        ro_matrix = [
            [1 - readout_error, readout_error],
            [readout_error, 1 - readout_error],
        ]
        nm.add_all_qubit_readout_error(ReadoutError(ro_matrix))

    return nm


# ---------------------------------------------------------------------------
# Noise sweep helper
# ---------------------------------------------------------------------------

def make_depolarizing_sweep(
    rates: list[float],
) -> list[tuple[float, NoiseModel | None]]:
    """
    Build a list of (rate, noise_model) pairs for a noise sweep.

    rate=0 produces None (ideal simulation).

    Parameters
    ----------
    rates : list[float]
        Error rates to sweep over.

    Returns
    -------
    list of (rate, noise_model | None)
    """
    result = []
    for r in rates:
        nm = None if r == 0.0 else depolarizing_model(error_rate=r)
        result.append((r, nm))
    return result


# Registry — look up models by name

NOISE_REGISTRY: dict[str, Callable[..., NoiseModel]] = {
    "depolarizing": depolarizing_model,
    "depolarizing_readout": depolarizing_readout_model,
    "thermal": thermal_relaxation_model,
}


def get_noise_model(name: str, **kwargs) -> NoiseModel:
    """
    Retrieve and instantiate a noise model by registry name.

    Parameters
    ----------
    name : str
        One of: "depolarizing", "depolarizing_readout", "thermal".
    **kwargs
        Passed to the noise model factory function.

    Raises
    ------
    KeyError
        If name is not in the registry.
    """
    if name not in NOISE_REGISTRY:
        raise KeyError(
            f"Unknown noise model '{name}'. "
            f"Available: {list(NOISE_REGISTRY.keys())}"
        )
    return NOISE_REGISTRY[name](**kwargs)
