"""
qiskit_impl/ibm_runner.py
--------------------------
Runner for executing the teleportation circuit on real IBM Quantum hardware
via IBM Quantum Cloud (qiskit-ibm-runtime).

Requirements
------------
    pip install qiskit-ibm-runtime

Setup
-----
    1. Create a free account at https://quantum.ibm.com
    2. Copy your API token from the dashboard
    3. Set it as an environment variable:
           export IBM_QUANTUM_TOKEN="your_token_here"
       Or pass it directly to IBMRunner(token="...").

Usage
-----
    from src.qiskit_impl.ibm_runner import IBMRunner
    from src.core.teleport import QubitState
    import numpy as np

    runner = IBMRunner()                       # reads token from env
    result = runner.run_teleport(
        QubitState(theta=np.pi/2, phi=0.0),
        backend_name="ibm_nairobi",            # or "least_busy"
        shots=1024,
    )
    print(result.summary())
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np

from src.core.teleport import QubitState, TeleportResult
from src.core.metrics import fidelity, von_neumann_entropy, pst_from_counts

# Guard import — qiskit-ibm-runtime is optional
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit_ibm_runtime import Session
    from qiskit import transpile
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False


class IBMRunner:
    """
    Runs the quantum teleportation protocol on real IBM Quantum hardware.

    Parameters
    ----------
    token : str, optional
        IBM Quantum API token. Falls back to the IBM_QUANTUM_TOKEN
        environment variable if not provided.
    instance : str
        IBM Quantum instance string (hub/group/project).
        Default is the open free-tier instance.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        instance: str = "ibm-q/open/main",
    ) -> None:
        if not IBM_RUNTIME_AVAILABLE:
            raise ImportError(
                "qiskit-ibm-runtime is not installed.\n"
                "Install it with:  pip install qiskit-ibm-runtime"
            )

        self.token    = token or os.environ.get("IBM_QUANTUM_TOKEN", "")
        self.instance = instance
        self._service: Optional[QiskitRuntimeService] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> QiskitRuntimeService:
        """Authenticate and return the IBM Quantum service object."""
        if not self.token:
            raise ValueError(
                "IBM Quantum token not found.\n"
                "Set the IBM_QUANTUM_TOKEN environment variable or pass "
                "token= to IBMRunner()."
            )
        if self._service is None:
            self._service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=self.token,
                instance=self.instance,
            )
        return self._service

    def list_backends(self, operational: bool = True) -> list[str]:
        """Return names of available backends."""
        service = self.connect()
        backends = service.backends(operational=operational)
        return [b.name for b in backends]

    def least_busy_backend(self, min_qubits: int = 5) -> str:
        """Return the name of the least-busy operational backend."""
        service = self.connect()
        backend = service.least_busy(
            operational=True,
            min_num_qubits=min_qubits,
        )
        print(f"Least busy backend: {backend.name}")
        return backend.name

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_teleport(
        self,
        state: QubitState,
        backend_name: str = "least_busy",
        shots: int = 1024,
        optimization_level: int = 1,
    ) -> TeleportResult:
        """
        Run the quantum teleportation protocol on IBM hardware.

        Parameters
        ----------
        state : QubitState
            The qubit state Alice wants to teleport.
        backend_name : str
            Name of the IBM backend, or "least_busy" to auto-select.
        shots : int
            Number of measurement shots.
        optimization_level : int
            Qiskit transpiler optimization level (0–3).

        Returns
        -------
        TeleportResult
        """
        from src.qiskit_impl.circuit import QiskitTeleporter

        service = self.connect()

        if backend_name == "least_busy":
            backend_name = self.least_busy_backend()

        backend = service.backend(backend_name)
        print(f"Submitting job to: {backend.name} ({backend.num_qubits} qubits)")

        # Build circuit
        teleporter = QiskitTeleporter(shots=shots)
        qc = teleporter.build_circuit(state)

        # Transpile for the target backend
        tqc = transpile(
            qc,
            backend=backend,
            optimization_level=optimization_level,
        )
        print(f"Circuit depth after transpilation: {tqc.depth()}")

        # Run via Session + Sampler
        with Session(backend=backend) as session:
            sampler = Sampler(session=session)
            job = sampler.run([tqc], shots=shots)
            print(f"Job ID: {job.job_id()}")
            print("Waiting for results (this may take several minutes)...")
            pub_result = job.result()[0]

        # Extract counts
        counts_raw = pub_result.data.c_bob.get_counts()
        bob_counts = {str(k): v for k, v in counts_raw.items()}

        # Build approximate output state from counts
        p0 = bob_counts.get("0", 0) / shots
        p1 = bob_counts.get("1", 0) / shots
        theta_out = 2 * np.arccos(np.sqrt(max(p0, 0)))
        output_state = QubitState(
            theta=float(theta_out),
            phi=state.phi,
            label="Bob's qubit (hardware)"
        )

        # Metrics
        f_val  = fidelity(state.statevector, output_state.statevector)
        s_val  = von_neumann_entropy(output_state.statevector)
        pst    = pst_from_counts(bob_counts)

        return TeleportResult(
            input_state=state,
            output_state=output_state,
            fidelity=f_val,
            pst=pst,
            counts=bob_counts,
            entropy=s_val,
            noise_model="ibm_hardware",
            backend=backend_name,
            shots=shots,
            raw_metadata={
                "backend_name":       backend_name,
                "circuit_depth":      tqc.depth(),
                "optimization_level": optimization_level,
            },
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def backend_properties(self, backend_name: str) -> dict:
        """
        Return key hardware properties for a given backend.

        Returns T1, T2, gate errors, and readout errors — useful for
        configuring the thermal relaxation noise model in simulation.
        """
        service  = self.connect()
        backend  = service.backend(backend_name)
        props    = backend.properties()

        result = {
            "name":        backend_name,
            "num_qubits":  backend.num_qubits,
            "T1_us":       [],
            "T2_us":       [],
            "readout_err": [],
        }

        for qubit in range(backend.num_qubits):
            result["T1_us"].append(props.t1(qubit) * 1e6)
            result["T2_us"].append(props.t2(qubit) * 1e6)
            result["readout_err"].append(props.readout_error(qubit))

        return result

    def __repr__(self) -> str:
        connected = self._service is not None
        return f"IBMRunner(instance='{self.instance}', connected={connected})"
