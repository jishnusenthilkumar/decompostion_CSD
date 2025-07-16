"""
Comprehensive benchmarking suite:
CSD pipeline vs Qiskit-Terra synthesis (Terra ≥ 1.0)

Fully corrected version with proper qubit indexing and fidelity calculation.
"""

import time
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import unitary_group

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate, CZGate
from qiskit.quantum_info import Operator, process_fidelity
from qiskit.synthesis import (
    TwoQubitBasisDecomposer,
    OneQubitEulerDecomposer,
    two_qubit_cnot_decompose,
)

# local import – your own routine
from csd_to_native_gates import csd_to_native_pipeline


@dataclass
class BenchmarkResult:
    method: str
    matrix_size: int
    gate_count: int
    circuit_depth: int
    cx_count: int
    single_qubit_count: int
    synthesis_time: float
    fidelity: float
    success: bool
    error_message: str = ""


class QiskitBenchmark:
    """Benchmark suite comparing CSD with various Qiskit synthesis paths."""

    def __init__(self, basis_gates: List[str] = None):
        self.basis_gates = basis_gates or ["rx", "ry", "rz", "cz"]
        self.results: List[BenchmarkResult] = []

        # Helper decomposers
        self.euler_decomposer = OneQubitEulerDecomposer(basis="ZYZ")
        self.two_qubit_decomposer = TwoQubitBasisDecomposer(
            gate=CZGate(), basis_fidelity=1.0
        )

    def create_qiskit_circuit_from_gates(
        self, gate_sequence: List, n_qubits: int
    ) -> QuantumCircuit:
        """
        Convert a list of custom gate objects into an executable Qiskit QuantumCircuit.
        """
        qc = QuantumCircuit(n_qubits)
        
        if not gate_sequence:
            return qc  # Return empty circuit if no gates

        for i, g in enumerate(gate_sequence):
            try:
                # Handle gate name
                if not hasattr(g, 'name'):
                    print(f"Warning: Gate {i} has no name attribute - skipping")
                    continue
                    
                name = str(g.name).upper()
                
                # Handle qubits - ensure they are integers and within valid range
                if not hasattr(g, 'qubits'):
                    print(f"Warning: Gate {i} ({name}) has no qubits attribute - skipping")
                    continue
                    
                qubits = []
                for q in g.qubits:
                    if hasattr(q, 'index'):
                        qubit_idx = int(q.index)
                    else:
                        qubit_idx = int(q)
                    
                    # CRITICAL FIX: Ensure qubit index is within valid range
                    if qubit_idx >= n_qubits:
                        # Map invalid qubit indices to valid ones
                        qubit_idx = qubit_idx % n_qubits
                        print(f"Warning: Gate {i} ({name}) qubit index {int(q)} mapped to {qubit_idx}")
                    
                    qubits.append(qubit_idx)
                
                # Handle parameters - ensure they are floats
                params = []
                if hasattr(g, 'params') and g.params:
                    for p in g.params:
                        params.append(float(p))

                # Apply gates based on name
                if name == "RX" and len(params) > 0 and len(qubits) > 0:
                    if abs(params[0]) > 1e-12:  # Skip near-zero rotations
                        qc.rx(params[0], qubits[0])
                elif name == "RY" and len(params) > 0 and len(qubits) > 0:
                    if abs(params[0]) > 1e-12:  # Skip near-zero rotations
                        qc.ry(params[0], qubits[0])
                elif name == "RZ" and len(params) > 0 and len(qubits) > 0:
                    if abs(params[0]) > 1e-12:  # Skip near-zero rotations
                        qc.rz(params[0], qubits[0])
                elif name == "CZ" and len(qubits) >= 2:
                    if qubits[0] != qubits[1]:  # Ensure different qubits
                        qc.cz(qubits[0], qubits[1])
                elif name == "CX" and len(qubits) >= 2:
                    if qubits[0] != qubits[1]:  # Ensure different qubits
                        qc.cx(qubits[0], qubits[1])
                elif name == "X" and len(qubits) > 0:
                    qc.x(qubits[0])
                elif name == "Y" and len(qubits) > 0:
                    qc.y(qubits[0])
                elif name == "Z" and len(qubits) > 0:
                    qc.z(qubits[0])
                else:
                    print(f"Warning: Unsupported or invalid gate '{name}' with {len(qubits)} qubits and {len(params)} params - skipping")
                    
            except Exception as e:
                print(f"Error processing gate {i}: {e}")
                continue

        return qc

    def benchmark_csd_method(self, U: np.ndarray) -> BenchmarkResult:
        start = time.time()
        try:
            # Run CSD pipeline
            pipe_res = csd_to_native_pipeline(U, optimize=True)
            
            # Check if pipeline succeeded
            if not isinstance(pipe_res, dict):
                raise ValueError("CSD pipeline did not return a dictionary")
                
            if 'gate_sequence' not in pipe_res:
                raise ValueError("CSD pipeline did not return gate_sequence")
                
            gate_sequence = pipe_res['gate_sequence']
            n_qubits = int(np.log2(U.shape[0]))
            qc = self.create_qiskit_circuit_from_gates(gate_sequence, n_qubits)

            synth_time = time.time() - start

            # Transpile to chosen basis
            qc_opt = transpile(qc, basis_gates=self.basis_gates, optimization_level=2)

            # Calculate metrics
            gate_cnt = len(qc_opt.data)
            depth = qc_opt.depth()
            cx_cnt = qc_opt.count_ops().get("cz", 0) + qc_opt.count_ops().get("cx", 0)
            single_cnt = gate_cnt - cx_cnt
            
            # CRITICAL FIX: Calculate fidelity properly
            try:
                U_reconstructed = Operator(qc_opt)
                # Ensure both are unitary matrices, not SuperOp
                if hasattr(U_reconstructed, 'data'):
                    U_reconstructed = U_reconstructed.data
                else:
                    U_reconstructed = U_reconstructed.to_matrix()
                    
                # Calculate fidelity between the two unitary matrices
                fidelity = process_fidelity(U, U_reconstructed)
                
                # If fidelity is complex, take real part
                if np.iscomplexobj(fidelity):
                    fidelity = np.real(fidelity)
                    
            except Exception as fid_e:
                print(f"Warning: Could not calculate fidelity: {fid_e}")
                # Alternative fidelity calculation
                try:
                    U_reconstructed = Operator(qc_opt).data
                    fidelity = abs(np.trace(U.conj().T @ U_reconstructed) / U.shape[0])**2
                except:
                    fidelity = 0.0

            return BenchmarkResult(
                method="CSD",
                matrix_size=U.shape[0],
                gate_count=gate_cnt,
                circuit_depth=depth,
                cx_count=cx_cnt,
                single_qubit_count=single_cnt,
                synthesis_time=synth_time,
                fidelity=fidelity,
                success=True,
            )

        except Exception as exc:
            print(f"CSD benchmark failed: {exc}")
            return BenchmarkResult(
                method="CSD",
                matrix_size=U.shape[0],
                gate_count=0,
                circuit_depth=0,
                cx_count=0,
                single_qubit_count=0,
                synthesis_time=time.time() - start,
                fidelity=0.0,
                success=False,
                error_message=str(exc),
            )

    def benchmark_qiskit_default(self, U: np.ndarray) -> BenchmarkResult:
        start = time.time()
        try:
            n_qubits = int(np.log2(U.shape[0]))
            qc = QuantumCircuit(n_qubits)
            qc.append(UnitaryGate(U), range(n_qubits))

            qc_opt = transpile(qc, basis_gates=self.basis_gates, optimization_level=2)
            synth_time = time.time() - start

            gate_cnt = len(qc_opt.data)
            depth = qc_opt.depth()
            cx_cnt = qc_opt.count_ops().get("cz", 0) + qc_opt.count_ops().get("cx", 0)
            single_cnt = gate_cnt - cx_cnt
            
            # CRITICAL FIX: Calculate fidelity properly
            try:
                U_reconstructed = Operator(qc_opt)
                if hasattr(U_reconstructed, 'data'):
                    U_reconstructed = U_reconstructed.data
                else:
                    U_reconstructed = U_reconstructed.to_matrix()
                    
                fidelity = process_fidelity(U, U_reconstructed)
                
                if np.iscomplexobj(fidelity):
                    fidelity = np.real(fidelity)
                    
            except Exception as fid_e:
                print(f"Warning: Could not calculate fidelity: {fid_e}")
                try:
                    U_reconstructed = Operator(qc_opt).data
                    fidelity = abs(np.trace(U.conj().T @ U_reconstructed) / U.shape[0])**2
                except:
                    fidelity = 0.0

            return BenchmarkResult(
                method="Qiskit_Default",
                matrix_size=U.shape[0],
                gate_count=gate_cnt,
                circuit_depth=depth,
                cx_count=cx_cnt,
                single_qubit_count=single_cnt,
                synthesis_time=synth_time,
                fidelity=fidelity,
                success=True,
            )

        except Exception as exc:
            print(f"Qiskit default benchmark failed: {exc}")
            return BenchmarkResult(
                method="Qiskit_Default",
                matrix_size=U.shape[0],
                gate_count=0,
                circuit_depth=0,
                cx_count=0,
                single_qubit_count=0,
                synthesis_time=time.time() - start,
                fidelity=0.0,
                success=False,
                error_message=str(exc),
            )

    def benchmark_qiskit_su4(self, U: np.ndarray) -> BenchmarkResult:
        if U.shape[0] != 4:
            return BenchmarkResult(
                method="Qiskit_SU4",
                matrix_size=U.shape[0],
                gate_count=0,
                circuit_depth=0,
                cx_count=0,
                single_qubit_count=0,
                synthesis_time=0.0,
                fidelity=0.0,
                success=False,
                error_message="SU4 synthesis only supports 2-qubit unitaries.",
            )

        start = time.time()
        try:
            # CRITICAL FIX: Remove the basis_gate parameter
            qc = two_qubit_cnot_decompose(U)
            synth_time = time.time() - start

            qc_opt = transpile(qc, basis_gates=self.basis_gates, optimization_level=2)

            gate_cnt = len(qc_opt.data)
            depth = qc_opt.depth()
            cx_cnt = qc_opt.count_ops().get("cz", 0) + qc_opt.count_ops().get("cx", 0)
            single_cnt = gate_cnt - cx_cnt
            
            # CRITICAL FIX: Calculate fidelity properly
            try:
                U_reconstructed = Operator(qc_opt)
                if hasattr(U_reconstructed, 'data'):
                    U_reconstructed = U_reconstructed.data
                else:
                    U_reconstructed = U_reconstructed.to_matrix()
                    
                fidelity = process_fidelity(U, U_reconstructed)
                
                if np.iscomplexobj(fidelity):
                    fidelity = np.real(fidelity)
                    
            except Exception as fid_e:
                print(f"Warning: Could not calculate fidelity: {fid_e}")
                try:
                    U_reconstructed = Operator(qc_opt).data
                    fidelity = abs(np.trace(U.conj().T @ U_reconstructed) / U.shape[0])**2
                except:
                    fidelity = 0.0

            return BenchmarkResult(
                method="Qiskit_SU4",
                matrix_size=4,
                gate_count=gate_cnt,
                circuit_depth=depth,
                cx_count=cx_cnt,
                single_qubit_count=single_cnt,
                synthesis_time=synth_time,
                fidelity=fidelity,
                success=True,
            )

        except Exception as exc:
            print(f"Qiskit SU4 benchmark failed: {exc}")
            return BenchmarkResult(
                method="Qiskit_SU4",
                matrix_size=4,
                gate_count=0,
                circuit_depth=0,
                cx_count=0,
                single_qubit_count=0,
                synthesis_time=time.time() - start,
                fidelity=0.0,
                success=False,
                error_message=str(exc),
            )

    def run_comprehensive_benchmark(self, U: np.ndarray) -> List[BenchmarkResult]:
        res = [
            self.benchmark_csd_method(U),
            self.benchmark_qiskit_default(U),
        ]
        if U.shape[0] == 4:
            res.append(self.benchmark_qiskit_su4(U))
        return res

    def run_matrix_size_benchmark(
        self, sizes: List[int] = [2, 4, 8], trials_per_size: int = 5
    ) -> Dict[str, List[BenchmarkResult]]:
        """Generate random unitaries and benchmark them."""
        all_res = {"CSD": [], "Qiskit_Default": [], "Qiskit_SU4": []}

        for size in sizes:
            print(f"\nTesting {size}×{size} unitaries …")
            for t in range(trials_per_size):
                U = unitary_group.rvs(size)
                for r in self.run_comprehensive_benchmark(U):
                    all_res[r.method].append(r)
                print(f"  trial {t + 1}/{trials_per_size} done")
        return all_res

    def analyze_results(
        self, results: Dict[str, List[BenchmarkResult]]
    ) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for meth, lst in results.items():
            if not lst:
                continue
            good = [r for r in lst if r.success]
            stats[meth] = {
                "success_rate": len(good) / len(lst) if lst else 0,
                "avg_gate_count": np.mean([r.gate_count for r in good]) if good else 0,
                "avg_circuit_depth": np.mean([r.circuit_depth for r in good])
                if good
                else 0,
                "avg_cx_count": np.mean([r.cx_count for r in good]) if good else 0,
                "avg_synthesis_time": np.mean([r.synthesis_time for r in good])
                if good
                else 0,
                "avg_fidelity": np.mean([r.fidelity for r in good]) if good else 0,
            }
        return stats

    def print_comparison_table(self, stats: Dict[str, Dict[str, float]]):
        print("\n" + "=" * 72)
        print("BENCHMARK SUMMARY")
        print("=" * 72)
        hdr = f"{'Metric':<22} {'CSD':<14} {'Qiskit_Default':<14} {'Qiskit_SU4':<14}"
        print(hdr)
        print("-" * len(hdr))
        rows = [
            ("Success rate", "success_rate", "{:.1%}"),
            ("Avg gate count", "avg_gate_count", "{:.1f}"),
            ("Avg depth", "avg_circuit_depth", "{:.1f}"),
            ("Avg CZ/CX", "avg_cx_count", "{:.1f}"),
            ("Avg synth time (s)", "avg_synthesis_time", "{:.4f}"),
            ("Avg fidelity", "avg_fidelity", "{:.6f}"),
        ]
        for label, key, fmt in rows:
            line = f"{label:<22}"
            for meth in ["CSD", "Qiskit_Default", "Qiskit_SU4"]:
                if meth in stats:
                    line += f" {fmt.format(stats[meth][key]):<14}"
                else:
                    line += f" {'N/A':<14}"
            print(line)

    def create_visualization(self, results: Dict[str, List[BenchmarkResult]]):
        fig, axs = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle("CSD vs Qiskit synthesis – benchmark", fontsize=16)

        def collect(key):
            return [
                [getattr(r, key) for r in lst if r.success] for lst in results.values()
            ]

        labels = list(results.keys())
        # Fixed: use tick_labels instead of labels for newer matplotlib
        axs[0, 0].boxplot(collect("gate_count"), tick_labels=labels)
        axs[0, 0].set_title("Gate count")

        axs[0, 1].boxplot(collect("circuit_depth"), tick_labels=labels)
        axs[0, 1].set_title("Circuit depth")

        axs[1, 0].boxplot(collect("cx_count"), tick_labels=labels)
        axs[1, 0].set_title("CZ/CX count")

        axs[1, 1].boxplot(collect("synthesis_time"), tick_labels=labels)
        axs[1, 1].set_title("Synthesis time (s)")

        for ax in axs.ravel():
            ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        plt.show()


def main():
    bench = QiskitBenchmark()
    results = bench.run_matrix_size_benchmark(sizes=[2, 4, 8], trials_per_size=5)
    stats = bench.analyze_results(results)
    bench.print_comparison_table(stats)
    bench.create_visualization(results)


if __name__ == "__main__":
    main()
