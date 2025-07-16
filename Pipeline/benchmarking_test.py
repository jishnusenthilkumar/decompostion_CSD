# In your test file, add:
from comprehensive_qiskit_benchmarking_suite import QiskitBenchmark
from scipy.stats import unitary_group

# Create custom benchmark
benchmark = QiskitBenchmark(basis_gates=['rx', 'ry', 'rz', 'cz'])

# Test specific unitary
U = unitary_group.rvs(4)  # 2-qubit random unitary
results = benchmark.run_comprehensive_benchmark(U)

# Print individual results
for result in results:
    print(f"{result.method}: {result.gate_count} gates, {result.circuit_depth} depth, {result.fidelity:.6f} fidelity")
