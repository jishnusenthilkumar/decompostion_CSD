"""
Comprehensive benchmarking suite comparing CSD-based synthesis 
with Qiskit's native unitary synthesis methods.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import unitary_group

# Qiskit imports (ensure qiskit >= 1.0)
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, process_fidelity
from qiskit.circuit.library import UnitaryGate
from qiskit.synthesis import (
    TwoQubitBasisDecomposer, 
    OneQubitEulerDecomposer,
    synth_su4_no_1q_gates,
    synth_cnot_count_full_ancilla
)
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler import PassManager

# Your custom implementation
from csd_to_native_gates import csd_to_native_pipeline

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
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
    """Benchmark suite comparing CSD with Qiskit synthesis methods."""
    
    def __init__(self, basis_gates: List[str] = None):
        self.basis_gates = basis_gates or ['rx', 'ry', 'rz', 'cz']
        self.results = []
        
        # Initialize Qiskit decomposers
        self.euler_decomposer = OneQubitEulerDecomposer(basis='ZYZ')
        self.two_qubit_decomposer = TwoQubitBasisDecomposer(
            gate=UnitaryGate(Operator.from_label('CZ').data),
            basis_fidelity=1.0
        )
        
    def create_qiskit_circuit_from_gates(self, gate_sequence: List, n_qubits: int) -> QuantumCircuit:
        """Convert your custom gate sequence to Qiskit circuit."""
        qc = QuantumCircuit(n_qubits)
        
        for gate in gate_sequence:
            if gate.name == 'RY':
                qc.ry(gate.params[0], gate.qubits[0])
            elif gate.name == 'RZ':
                qc.rz(gate.params[0], gate.qubits[0])
            elif gate.name == 'RX':
                qc.rx(gate.params[0], gate.qubits[0])
            elif gate.name == 'CZ':
                qc.cz(gate.qubits[0], gate.qubits[1])
            elif gate.name == 'X':
                qc.x(gate.qubits[0])
            elif gate.name == 'Y':
                qc.y(gate.qubits[0])
            elif gate.name == 'Z':
                qc.z(gate.qubits[0])
        
        return qc
    
    def benchmark_csd_method(self, U: np.ndarray) -> BenchmarkResult:
        """Benchmark your CSD implementation."""
        start_time = time.time()
        
        try:
            # Run your CSD pipeline
            results = csd_to_native_pipeline(U, optimize=True)
            synthesis_time = time.time() - start_time
            
            # Convert to Qiskit circuit
            n_qubits = int(np.log2(U.shape[0]))
            qc = self.create_qiskit_circuit_from_gates(results['gate_sequence'], n_qubits)
            
            # Optimize with Qiskit
            qc_optimized = transpile(qc, basis_gates=self.basis_gates, optimization_level=2)
            
            # Calculate metrics
            gate_count = len(qc_optimized.data)
            circuit_depth = qc_optimized.depth()
            
            # Count specific gate types
            cx_count = qc_optimized.count_ops().get('cz', 0) + qc_optimized.count_ops().get('cx', 0)
            single_qubit_count = gate_count - cx_count
            
            # Calculate fidelity
            U_reconstructed = Operator(qc_optimized).data
            fidelity = process_fidelity(U, U_reconstructed)
            
            return BenchmarkResult(
                method="CSD",
                matrix_size=U.shape[0],
                gate_count=gate_count,
                circuit_depth=circuit_depth,
                cx_count=cx_count,
                single_qubit_count=single_qubit_count,
                synthesis_time=synthesis_time,
                fidelity=fidelity,
                success=True
            )
            
        except Exception as e:
            return BenchmarkResult(
                method="CSD",
                matrix_size=U.shape[0],
                gate_count=0,
                circuit_depth=0,
                cx_count=0,
                single_qubit_count=0,
                synthesis_time=time.time() - start_time,
                fidelity=0.0,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_qiskit_default(self, U: np.ndarray) -> BenchmarkResult:
        """Benchmark Qiskit's default UnitaryGate synthesis."""
        start_time = time.time()
        
        try:
            n_qubits = int(np.log2(U.shape[0]))
            qc = QuantumCircuit(n_qubits)
            
            # Use Qiskit's UnitaryGate
            unitary_gate = UnitaryGate(U)
            qc.append(unitary_gate, range(n_qubits))
            
            # Transpile to basis gates
            qc_transpiled = transpile(qc, basis_gates=self.basis_gates, optimization_level=2)
            synthesis_time = time.time() - start_time
            
            # Calculate metrics
            gate_count = len(qc_transpiled.data)
            circuit_depth = qc_transpiled.depth()
            
            # Count specific gate types
            cx_count = qc_transpiled.count_ops().get('cz', 0) + qc_transpiled.count_ops().get('cx', 0)
            single_qubit_count = gate_count - cx_count
            
            # Calculate fidelity
            U_reconstructed = Operator(qc_transpiled).data
            fidelity = process_fidelity(U, U_reconstructed)
            
            return BenchmarkResult(
                method="Qiskit_Default",
                matrix_size=U.shape[0],
                gate_count=gate_count,
                circuit_depth=circuit_depth,
                cx_count=cx_count,
                single_qubit_count=single_qubit_count,
                synthesis_time=synthesis_time,
                fidelity=fidelity,
                success=True
            )
            
        except Exception as e:
            return BenchmarkResult(
                method="Qiskit_Default",
                matrix_size=U.shape[0],
                gate_count=0,
                circuit_depth=0,
                cx_count=0,
                single_qubit_count=0,
                synthesis_time=time.time() - start_time,
                fidelity=0.0,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_qiskit_su4(self, U: np.ndarray) -> BenchmarkResult:
        """Benchmark Qiskit's specialized SU(4) synthesis for 2-qubit gates."""
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
                error_message="SU4 synthesis only for 2-qubit gates"
            )
        
        start_time = time.time()
        
        try:
            # Use Qiskit's optimized SU(4) synthesis
            qc = synth_su4_no_1q_gates(U)
            synthesis_time = time.time() - start_time
            
            # Optimize circuit
            qc_optimized = transpile(qc, basis_gates=self.basis_gates, optimization_level=2)
            
            # Calculate metrics
            gate_count = len(qc_optimized.data)
            circuit_depth = qc_optimized.depth()
            
            # Count specific gate types
            cx_count = qc_optimized.count_ops().get('cz', 0) + qc_optimized.count_ops().get('cx', 0)
            single_qubit_count = gate_count - cx_count
            
            # Calculate fidelity
            U_reconstructed = Operator(qc_optimized).data
            fidelity = process_fidelity(U, U_reconstructed)
            
            return BenchmarkResult(
                method="Qiskit_SU4",
                matrix_size=U.shape[0],
                gate_count=gate_count,
                circuit_depth=circuit_depth,
                cx_count=cx_count,
                single_qubit_count=single_qubit_count,
                synthesis_time=synthesis_time,
                fidelity=fidelity,
                success=True
            )
            
        except Exception as e:
            return BenchmarkResult(
                method="Qiskit_SU4",
                matrix_size=U.shape[0],
                gate_count=0,
                circuit_depth=0,
                cx_count=0,
                single_qubit_count=0,
                synthesis_time=time.time() - start_time,
                fidelity=0.0,
                success=False,
                error_message=str(e)
            )
    
    def run_comprehensive_benchmark(self, U: np.ndarray) -> List[BenchmarkResult]:
        """Run all benchmark methods on a single unitary."""
        results = []
        
        # Test your CSD method
        results.append(self.benchmark_csd_method(U))
        
        # Test Qiskit's default synthesis
        results.append(self.benchmark_qiskit_default(U))
        
        # Test Qiskit's SU(4) synthesis for 2-qubit gates
        if U.shape[0] == 4:
            results.append(self.benchmark_qiskit_su4(U))
        
        return results
    
    def run_matrix_size_benchmark(self, sizes: List[int] = [2, 4, 8], 
                                 trials_per_size: int = 10) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmarks across different matrix sizes."""
        all_results = {
            'CSD': [],
            'Qiskit_Default': [],
            'Qiskit_SU4': []
        }
        
        for size in sizes:
            print(f"\nTesting {size}×{size} matrices ({int(np.log2(size))} qubits)...")
            
            for trial in range(trials_per_size):
                # Generate random unitary matrix
                if size == 2:
                    # Special case for single-qubit
                    U = unitary_group.rvs(2)
                else:
                    U = unitary_group.rvs(size)
                
                results = self.run_comprehensive_benchmark(U)
                
                for result in results:
                    if result.method in all_results:
                        all_results[result.method].append(result)
                
                print(f"  Trial {trial + 1}/{trials_per_size} completed")
        
        return all_results
    
    def analyze_results(self, results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Dict[str, float]]:
        """Analyze benchmark results and compute statistics."""
        analysis = {}
        
        for method, method_results in results.items():
            if not method_results:
                continue
                
            successful_results = [r for r in method_results if r.success]
            
            if not successful_results:
                analysis[method] = {
                    'success_rate': 0.0,
                    'avg_gate_count': 0.0,
                    'avg_circuit_depth': 0.0,
                    'avg_cx_count': 0.0,
                    'avg_synthesis_time': 0.0,
                    'avg_fidelity': 0.0
                }
                continue
            
            analysis[method] = {
                'success_rate': len(successful_results) / len(method_results),
                'avg_gate_count': np.mean([r.gate_count for r in successful_results]),
                'avg_circuit_depth': np.mean([r.circuit_depth for r in successful_results]),
                'avg_cx_count': np.mean([r.cx_count for r in successful_results]),
                'avg_synthesis_time': np.mean([r.synthesis_time for r in successful_results]),
                'avg_fidelity': np.mean([r.fidelity for r in successful_results]),
                'std_gate_count': np.std([r.gate_count for r in successful_results]),
                'std_circuit_depth': np.std([r.circuit_depth for r in successful_results]),
                'std_cx_count': np.std([r.cx_count for r in successful_results])
            }
        
        return analysis
    
    def print_comparison_table(self, analysis: Dict[str, Dict[str, float]]):
        """Print a formatted comparison table."""
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON RESULTS")
        print("="*80)
        
        methods = list(analysis.keys())
        if not methods:
            print("No results to display.")
            return
        
        # Print header
        print(f"{'Metric':<25} {'CSD':<15} {'Qiskit_Default':<15} {'Qiskit_SU4':<15}")
        print("-" * 80)
        
        # Print metrics
        metrics = [
            ('Success Rate', 'success_rate', '{:.1%}'),
            ('Avg Gate Count', 'avg_gate_count', '{:.1f}'),
            ('Avg Circuit Depth', 'avg_circuit_depth', '{:.1f}'),
            ('Avg CZ Count', 'avg_cx_count', '{:.1f}'),
            ('Avg Synthesis Time (s)', 'avg_synthesis_time', '{:.4f}'),
            ('Avg Fidelity', 'avg_fidelity', '{:.6f}')
        ]
        
        for metric_name, metric_key, format_str in metrics:
            row = f"{metric_name:<25}"
            for method in ['CSD', 'Qiskit_Default', 'Qiskit_SU4']:
                if method in analysis and metric_key in analysis[method]:
                    value = analysis[method][metric_key]
                    row += f" {format_str.format(value):<15}"
                else:
                    row += f" {'N/A':<15}"
            print(row)
    
    def create_visualization(self, results: Dict[str, List[BenchmarkResult]]):
        """Create visualization plots for benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CSD vs Qiskit Synthesis Benchmark Results', fontsize=16)
        
        # Prepare data for plotting
        methods = []
        gate_counts = []
        circuit_depths = []
        cx_counts = []
        synthesis_times = []
        
        for method, method_results in results.items():
            successful_results = [r for r in method_results if r.success]
            if successful_results:
                methods.append(method)
                gate_counts.append([r.gate_count for r in successful_results])
                circuit_depths.append([r.circuit_depth for r in successful_results])
                cx_counts.append([r.cx_count for r in successful_results])
                synthesis_times.append([r.synthesis_time for r in successful_results])
        
        # Plot 1: Gate Count Distribution
        axes[0, 0].boxplot(gate_counts, labels=methods)
        axes[0, 0].set_title('Gate Count Distribution')
        axes[0, 0].set_ylabel('Number of Gates')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Circuit Depth Distribution
        axes[0, 1].boxplot(circuit_depths, labels=methods)
        axes[0, 1].set_title('Circuit Depth Distribution')
        axes[0, 1].set_ylabel('Circuit Depth')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: CZ Gate Count Distribution
        axes[1, 0].boxplot(cx_counts, labels=methods)
        axes[1, 0].set_title('CZ Gate Count Distribution')
        axes[1, 0].set_ylabel('Number of CZ Gates')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Synthesis Time Distribution
        axes[1, 1].boxplot(synthesis_times, labels=methods)
        axes[1, 1].set_title('Synthesis Time Distribution')
        axes[1, 1].set_ylabel('Synthesis Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('csd_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main benchmarking function."""
    print("CSD vs Qiskit Synthesis Benchmark")
    print("=" * 50)
    
    # Initialize benchmark
    benchmark = QiskitBenchmark()
    
    # Run comprehensive benchmark
    results = benchmark.run_matrix_size_benchmark(
        sizes=[2, 4, 8], 
        trials_per_size=5
    )
    
    # Analyze results
    analysis = benchmark.analyze_results(results)
    
    # Print comparison table
    benchmark.print_comparison_table(analysis)
    
    # Create visualizations
    benchmark.create_visualization(results)
    
    # Save detailed results
    print("\nDetailed results saved to 'benchmark_results.txt'")
    with open('benchmark_results.txt', 'w') as f:
        f.write("CSD vs Qiskit Synthesis Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        
        for method, method_results in results.items():
            f.write(f"{method} Results:\n")
            f.write("-" * 30 + "\n")
            
            for i, result in enumerate(method_results):
                f.write(f"Trial {i+1}:\n")
                f.write(f"  Matrix Size: {result.matrix_size}×{result.matrix_size}\n")
                f.write(f"  Gate Count: {result.gate_count}\n")
                f.write(f"  Circuit Depth: {result.circuit_depth}\n")
                f.write(f"  CZ Count: {result.cx_count}\n")
                f.write(f"  Synthesis Time: {result.synthesis_time:.4f}s\n")
                f.write(f"  Fidelity: {result.fidelity:.6f}\n")
                f.write(f"  Success: {result.success}\n")
                if not result.success:
                    f.write(f"  Error: {result.error_message}\n")
                f.write("\n")
            f.write("\n")

if __name__ == "__main__":
    main()
