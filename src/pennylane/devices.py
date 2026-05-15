"""
pennylane_impl/devices.py
--------------------------
PennyLane device factory with automatic selection based on NoiseConfig.

Abstracts away device creation so circuit code stays framework-agnostic
at the device level.

Usage
-----
    from src.pennylane_impl.devices import make_device, DeviceConfig

    dev = make_device(DeviceConfig(name="default.qubit", shots=4096))
    dev = make_device(DeviceConfig(name="default.mixed", shots=4096))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pennylane as qml

from src.core.noise_models import NoiseConfig, NoiseLevel


# Device configuration

@dataclass
class DeviceConfig:
    """
    Configuration for a PennyLane device.

    Attributes
    ----------
    name : str
        PennyLane device name.
        "default.qubit"  → ideal statevector (fastest).
        "default.mixed"  → density matrix, supports noise channels.
        "lightning.qubit"→ C++ accelerated statevector (no noise).
    wires : int
        Number of qubits.
    shots : int or None
        Shot count. None = analytic (exact) mode.
    seed : int or None
        RNG seed for reproducibility.
    diff_method : str
        Differentiation method for QNodes.
        "best", "parameter-shift", "backprop", "adjoint".
    """
    name: str = "default.qubit"
    wires: int = 3
    shots: Optional[int] = 4096
    seed: Optional[int] = 42
    diff_method: str = "best"

    @classmethod
    def from_noise_config(cls, noise: Optional[NoiseConfig], **kwargs) -> "DeviceConfig":
        """
        Choose the appropriate device based on the noise configuration.

        - Ideal / no noise → default.qubit (fast).
        - Any noise        → default.mixed  (density matrix).
        """
        needs_mixed = (
            noise is not None
            and noise.level != NoiseLevel.IDEAL
        )
        name = "default.mixed" if needs_mixed else "default.qubit"
        return cls(name=name, **kwargs)


# Device factory

def make_device(cfg: DeviceConfig) -> qml.Device:
    """
    Instantiate a PennyLane device from a DeviceConfig.

    Parameters
    ----------
    cfg : DeviceConfig
        Device configuration.

    Returns
    -------
    qml.Device
    """
    kwargs = {"wires": cfg.wires}

    if cfg.shots is not None:
        kwargs["shots"] = cfg.shots

    if cfg.seed is not None and cfg.name in ("default.qubit", "default.mixed"):
        kwargs["seed"] = cfg.seed

    return qml.device(cfg.name, **kwargs)


def noise_config_to_channels(noise: Optional[NoiseConfig]) -> Optional[list]:
    """
    Convert a NoiseConfig into a list of PennyLane channel descriptors.

    Returns None for ideal simulation.

    Each entry is (channel_class, kwargs_dict) — consumed by
    _apply_noise() in pennylane_impl/circuit.py.
    """
    if noise is None or noise.level == NoiseLevel.IDEAL:
        return None

    if noise.level == NoiseLevel.DEPOLARIZING:
        channels = [(qml.DepolarizingChannel, {"p": noise.error_rate})]
        channels._label = noise.label
        return channels

    if noise.level == NoiseLevel.DEPOLARIZING_READOUT:
        # PennyLane doesn't have a native readout error channel;
        # approximate with depolarizing + bit-flip
        channels = [
            (qml.DepolarizingChannel, {"p": noise.error_rate}),
            (qml.BitFlip,             {"p": noise.readout_error}),
        ]
        channels._label = noise.label
        return channels

    if noise.level == NoiseLevel.AMPLITUDE_DAMPING:
        channels = [(qml.AmplitudeDamping, {"gamma": noise.error_rate})]
        channels._label = noise.label
        return channels

    if noise.level == NoiseLevel.PHASE_DAMPING:
        channels = [(qml.PhaseDamping, {"gamma": noise.error_rate})]
        channels._label = noise.label
        return channels

    if noise.level == NoiseLevel.THERMAL:
        # Approximate thermal relaxation:
        # T1 decay → amplitude damping with γ ≈ 1 - exp(-t/T1)
        import numpy as np
        t_gate_us = noise.gate_time_1q_ns / 1000.0
        gamma_t1  = 1.0 - np.exp(-t_gate_us / noise.T1_us)
        gamma_t2  = 1.0 - np.exp(-t_gate_us / noise.T2_us)
        channels  = [
            (qml.AmplitudeDamping, {"gamma": float(np.clip(gamma_t1, 0, 1))}),
            (qml.PhaseDamping,     {"gamma": float(np.clip(gamma_t2, 0, 1))}),
        ]
        channels._label = noise.label
        return channels

    raise ValueError(f"Unsupported NoiseLevel for PennyLane: {noise.level}")


# Device registry

DEVICE_REGISTRY = {
    "ideal":        DeviceConfig(name="default.qubit"),
    "mixed":        DeviceConfig(name="default.mixed"),
    "lightning":    DeviceConfig(name="lightning.qubit"),
    "analytic":     DeviceConfig(name="default.qubit",  shots=None),
}


def get_device(name: str, **kwargs) -> qml.Device:
    """
    Look up a device by registry name and instantiate it.

    Parameters
    ----------
    name : str
        One of: "ideal", "mixed", "lightning", "analytic".
    **kwargs
        Override DeviceConfig fields (shots, seed, wires, etc.).
    """
    if name not in DEVICE_REGISTRY:
        raise KeyError(
            f"Unknown device '{name}'. "
            f"Available: {list(DEVICE_REGISTRY.keys())}"
        )
    cfg = DEVICE_REGISTRY[name]
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return make_device(cfg)
