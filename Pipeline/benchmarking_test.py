from comprehensive_qiskit_benchmarking_suite import QiskitBenchmark
from scipy.stats import unitary_group

benchmark = QiskitBenchmark(basis_gates=['rx', 'ry', 'rz', 'cz'])
U = unitary_group.rvs(4)
results = benchmark.run_comprehensive_benchmark(U)

for result in results:
    print(f"{result.method}: {result.gate_count} gates, {result.circuit_depth} depth, {result.fidelity:.6f} fidelity")
