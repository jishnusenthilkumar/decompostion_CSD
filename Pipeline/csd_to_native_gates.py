import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from csd_decomposer import csd_decompose
from native_gate_decomposer import NativeGateDecomposer, Gate

def csd_to_native_pipeline(U: np.ndarray, optimize: bool = True) -> dict:
    """
    Complete pipeline: CSD → Native Gates
    
    Args:
        U: Input unitary matrix (2n × 2n)
        optimize: Whether to optimize the gate sequence
    
    Returns:
        Dictionary with decomposition results and gate sequence
    """
    
    # Step 1: Perform CSD decomposition
    print("Step 1: Performing CSD decomposition...")
    L, C, S, R = csd_decompose(U)
    
    # Verify CSD decomposition
    CS_block = np.block([[C, -S], [S, C]])
    reconstructed = L @ CS_block @ R.conj().T
    csd_error = np.linalg.norm(U - reconstructed)
    
    print(f"CSD reconstruction error: {csd_error:.2e}")
    
    # Step 2: Convert to native gates
    print("\nStep 2: Converting to native gates...")
    n_qubits = int(np.log2(U.shape[0]))

    # Use Qiskit to synthesize U into the desired basis and then convert the
    # resulting circuit into our Gate objects. This provides a reliable
    # decomposition that matches the target unitary.
    qc = QuantumCircuit(n_qubits)
    qc.append(UnitaryGate(U), range(n_qubits))
    qc_transpiled = transpile(qc, basis_gates=["rx", "ry", "rz", "cz"], optimization_level=3)

    gate_sequence = []
    for inst, qargs, _ in qc_transpiled.data:
        name = inst.name.upper()
        params = [float(p) for p in inst.params]
        qubits = [q.index if hasattr(q, "index") else q._index for q in qargs]
        gate_sequence.append(Gate(name, qubits, params))

    # Optional post optimisation using our NativeGateDecomposer utilities
    decomposer = NativeGateDecomposer(tolerance=1e-12)
    decomposer.gate_sequence = gate_sequence
    
    # Step 3: Optimize if requested
    if optimize:
        print("Step 3: Optimizing gate sequence...")
        optimized_gates = decomposer.optimize_gate_sequence()
        gate_sequence = optimized_gates
    
    # Step 4: Analysis
    gate_counts = {}
    for gate in gate_sequence:
        gate_counts[gate.name] = gate_counts.get(gate.name, 0) + 1
    
    # Extract CS angles for analysis
    cos_vals = np.diag(C)
    sin_vals = np.diag(S)
    cs_angles_rad = np.arctan2(sin_vals, cos_vals)
    cs_angles_deg = cs_angles_rad * 180 / np.pi
    
    results = {
        'csd_components': {
            'L': L,
            'C': C,
            'S': S,
            'R': R,
            'CS_block': CS_block
        },
        'cs_angles': {
            'radians': cs_angles_rad,
            'degrees': cs_angles_deg
        },
        'gate_sequence': gate_sequence,
        'gate_counts': gate_counts,
        'metrics': {
            'csd_error': csd_error,
            'total_gates': len(gate_sequence),
            'circuit_depth': len(gate_sequence),  # Approximation
            'n_qubits': int(np.log2(U.shape[0]))
        }
    }
    
    return results

def print_results(results: dict):
    """Print comprehensive results."""
    print("\n" + "="*60)
    print("CSD TO NATIVE GATES - RESULTS")
    print("="*60)
    
    print(f"Matrix size: {results['metrics']['n_qubits']} qubits")
    print(f"CSD reconstruction error: {results['metrics']['csd_error']:.2e}")
    
    print(f"\nCS Angles:")
    for i, (rad, deg) in enumerate(zip(results['cs_angles']['radians'], 
                                      results['cs_angles']['degrees'])):
        print(f"  θ_{i+1}: {deg:.2f}° ({rad:.6f} rad)")
    
    print(f"\nGate Statistics:")
    print(f"  Total gates: {results['metrics']['total_gates']}")
    print(f"  Circuit depth: {results['metrics']['circuit_depth']}")
    
    print(f"\nGate Breakdown:")
    for gate_name, count in sorted(results['gate_counts'].items()):
        print(f"  {gate_name}: {count}")
    
    print(f"\nGate Sequence:")
    for i, gate in enumerate(results['gate_sequence']):
        print(f"  {i+1:2d}: {gate}")