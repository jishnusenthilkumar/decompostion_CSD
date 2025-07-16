import numpy as np
from csd_decomposer import csd_decompose
from native_gate_decomposer import NativeGateDecomposer

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
    decomposer = NativeGateDecomposer(tolerance=1e-12)
    
    # Decompose the CSD components
    gate_sequence = decomposer.decompose_csd_output(L, C, S, R)
    
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