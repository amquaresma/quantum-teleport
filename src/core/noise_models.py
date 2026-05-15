"""
core/noise_models.py
--------------------
Framework-agnostic noise model descriptors.

These dataclasses carry noise parameters in a neutral format.
Each framework adapter (qiskit_impl, pennylane_impl) reads them
and builds the appropriate native noise objects.

Usage
-----
    from src.core.noise_models import NoiseConfig, NoiseLevel

    cfg = NoiseConfig(level=NoiseLevel.DEPOLARIZING, error_rate=0.01)
    # pass to QiskitTeleporter or PennyLaneTeleporter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NoiseLevel(str, Enum):
    """Enumeration of supported noise model types."""
    IDEAL               = "ideal"
    DEPOLARIZING        = "depolarizing"
    DEPOLARIZING_READOUT = "depolarizing_readout"
    THERMAL             = "thermal"
    AMPLITUDE_DAMPING   = "amplitude_damping"
    PHASE_DAMPING       = "phase_damping"


@dataclass
class NoiseConfig:
    """
    Neutral noise configuration passed to any teleporter implementation.

    Attributes
    ----------
    level : NoiseLevel
        The type of noise model.
    error_rate : float
        Primary error parameter (gate error probability for depolarizing,
        damping parameter γ for amplitude/phase damping).
    readout_error : float
        Measurement bit-flip probability (used in DEPOLARIZING_READOUT).
    T1_us : float
        Energy relaxation time in microseconds (used in THERMAL).
    T2_us : float
        Dephasing time in microseconds (used in THERMAL).
    gate_time_1q_ns : float
        Single-qubit gate time in nanoseconds (used in THERMAL).
    gate_time_2q_ns : float
        Two-qubit gate time in nanoseconds (used in THERMAL).
    two_qubit_factor : float
        Multiplier for two-qubit gate error vs single-qubit.
    label : str
        Human-readable label override. Auto-generated if empty.
    """
    level: NoiseLevel = NoiseLevel.IDEAL
    error_rate: float = 0.0
    readout_error: float = 0.02
    T1_us: float = 100.0
    T2_us: float = 80.0
    gate_time_1q_ns: float = 50.0
    gate_time_2q_ns: float = 300.0
    two_qubit_factor: float = 10.0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self._auto_label()

    def _auto_label(self) -> str:
        if self.level == NoiseLevel.IDEAL:
            return "ideal"
        if self.level == NoiseLevel.DEPOLARIZING:
            return f"depolarizing(p={self.error_rate:.4f})"
        if self.level == NoiseLevel.DEPOLARIZING_READOUT:
            return f"dep+readout(p={self.error_rate:.4f}, ro={self.readout_error:.4f})"
        if self.level == NoiseLevel.THERMAL:
            return f"thermal(T1={self.T1_us}µs, T2={self.T2_us}µs)"
        if self.level == NoiseLevel.AMPLITUDE_DAMPING:
            return f"amplitude_damping(γ={self.error_rate:.4f})"
        if self.level == NoiseLevel.PHASE_DAMPING:
            return f"phase_damping(γ={self.error_rate:.4f})"
        return str(self.level)

    @classmethod
    def ideal(cls) -> "NoiseConfig":
        return cls(level=NoiseLevel.IDEAL)

    @classmethod
    def depolarizing(cls, p: float) -> "NoiseConfig":
        return cls(level=NoiseLevel.DEPOLARIZING, error_rate=p)

    @classmethod
    def depolarizing_readout(cls, p: float = 0.005, ro: float = 0.02) -> "NoiseConfig":
        return cls(level=NoiseLevel.DEPOLARIZING_READOUT,
                   error_rate=p, readout_error=ro)

    @classmethod
    def thermal(
        cls,
        T1_us: float = 100.0,
        T2_us: float = 80.0,
        gate_1q_ns: float = 50.0,
        gate_2q_ns: float = 300.0,
    ) -> "NoiseConfig":
        return cls(
            level=NoiseLevel.THERMAL,
            T1_us=T1_us, T2_us=T2_us,
            gate_time_1q_ns=gate_1q_ns,
            gate_time_2q_ns=gate_2q_ns,
        )


# Pre-built sweep presets


STANDARD_SWEEP = [
    NoiseConfig.ideal(),
    NoiseConfig.depolarizing(p=0.001),
    NoiseConfig.depolarizing(p=0.005),
    NoiseConfig.depolarizing(p=0.01),
    NoiseConfig.depolarizing(p=0.02),
    NoiseConfig.depolarizing(p=0.05),
    NoiseConfig.depolarizing(p=0.10),
    NoiseConfig.depolarizing_readout(p=0.005, ro=0.02),
    NoiseConfig.thermal(),
]
