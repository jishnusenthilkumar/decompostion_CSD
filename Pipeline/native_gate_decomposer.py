import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import cmath

@dataclass
class Gate:
    """Represents a quantum gate with parameters."""
    name: str
    qubits: List[int]
    params: List[float] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = []
    
    def __str__(self):
        if self.params:
            param_str = f"({', '.join(f'{p:.6f}' for p in self.params)})"
            return f"{self.name}{param_str} q{self.qubits}"
        return f"{self.name} q{self.qubits}"

class NativeGateDecomposer:
    """
    Decomposes unitary matrices into native gate sequences.
    Supports RX, RY, RZ, X, Y, Z, CZ gates.
    """
    
    def __init__(self, tolerance: float = 1e-12):
        self.tolerance = tolerance
        self.gate_sequence = []
    
    def reset(self):
        """Reset the gate sequence."""
        self.gate_sequence = []
    
    def add_gate(self, name: str, qubits: List[int], params: List[float] = None):
        """Add a gate to the sequence."""
        self.gate_sequence.append(Gate(name, qubits, params or []))
    
    def euler_zyz_decomposition(self, U: np.ndarray, qubit: int = 0) -> List[Gate]:
        """
        ZYZ Euler decomposition for single-qubit unitaries.
        Any SU(2) matrix can be written as: U = e^(iα) Rz(λ) Ry(θ) Rz(φ)
        """
        if U.shape != (2, 2):
            raise ValueError("Euler decomposition requires 2x2 matrix")
        
        # Ensure the matrix is in SU(2) by removing global phase
        det_U = np.linalg.det(U)
        if abs(det_U) > self.tolerance:
            U = U / np.sqrt(det_U)
        
        # Extract Euler angles using standard ZYZ decomposition
        # U = [[cos(θ/2)e^(i(φ+λ)/2), -sin(θ/2)e^(i(φ-λ)/2)],
        #      [sin(θ/2)e^(-i(φ-λ)/2), cos(θ/2)e^(-i(φ+λ)/2)]]
        
        gates = []
        
        # Handle special case where U[0,1] = 0 (no Y rotation needed)
        if abs(U[0, 1]) < self.tolerance:
            # Only Z rotations needed
            if abs(U[0, 0]) > self.tolerance:
                angle = 2 * np.angle(U[0, 0])
                if abs(angle) > self.tolerance:
                    gates.append(Gate("RZ", [qubit], [angle]))
            return gates
        
        # General case: extract all three angles
        theta = 2 * np.arctan2(abs(U[0, 1]), abs(U[0, 0]))
        
        if abs(np.sin(theta/2)) > self.tolerance:
            # Compute phi and lambda
            phi_plus_lambda = np.angle(U[0, 0]) - np.angle(U[1, 1])
            phi_minus_lambda = np.angle(U[0, 1]) - np.angle(U[1, 0]) + np.pi
            
            phi = (phi_plus_lambda + phi_minus_lambda) / 2
            lambda_angle = (phi_plus_lambda - phi_minus_lambda) / 2
            
            # Add gates in ZYZ order
            if abs(lambda_angle) > self.tolerance:
                gates.append(Gate("RZ", [qubit], [lambda_angle]))
            if abs(theta) > self.tolerance:
                gates.append(Gate("RY", [qubit], [theta]))
            if abs(phi) > self.tolerance:
                gates.append(Gate("RZ", [qubit], [phi]))
        
        return gates
    
    def kak_decomposition(self, U: np.ndarray, qubits: List[int] = [0, 1]) -> List[Gate]:
        """
        KAK decomposition for two-qubit unitaries.
        U = (A1 ⊗ A2) * exp(i(α*XX + β*YY + γ*ZZ)) * (B1 ⊗ B2)
        """
        if U.shape != (4, 4):
            raise ValueError("KAK decomposition requires 4x4 matrix")
        
        # Remove global phase
        det_U = np.linalg.det(U)
        if abs(det_U) > self.tolerance:
            U = U / np.sqrt(np.sqrt(det_U))
        
        gates = []
        
        # Magic basis for two-qubit gates
        magic_basis = np.array([
            [1, 0, 0, 1j],
            [0, 1j, 1, 0],
            [0, 1j, -1, 0],
            [1, 0, 0, -1j]
        ]) / np.sqrt(2)
        
        # Transform to magic basis
        U_magic = magic_basis.conj().T @ U @ magic_basis
        
        # Singular value decomposition
        V, s, Wh = np.linalg.svd(U_magic)
        
        # Extract interaction angles
        alpha = np.angle(s[0])
        beta = np.angle(s[1])
        gamma = np.angle(s[2])
        
        # Convert back to computational basis and extract single-qubit unitaries
        # This is a simplified version - full KAK requires more complex extraction
        
        # For now, implement a basic two-qubit decomposition
        # Check if it's a product state (separable)
        if self._is_product_state(U):
            # Decompose as tensor product of single-qubit gates
            U1, U2 = self._extract_product_unitaries(U)
            gates.extend(self.euler_zyz_decomposition(U1, qubits[0]))
            gates.extend(self.euler_zyz_decomposition(U2, qubits[1]))
        else:
            # General two-qubit case - use CZ-based decomposition
            gates.extend(self._general_two_qubit_decomposition(U, qubits))
        
        return gates
    
    def _is_product_state(self, U: np.ndarray) -> bool:
        """Check if a 4x4 unitary is a product of two single-qubit unitaries."""
        # Simple heuristic: compute Schmidt rank
        U_reshaped = U.reshape(2, 2, 2, 2)
        # Flatten to matrix form for SVD
        U_flat = U_reshaped.reshape(4, 4)
        _, s, _ = np.linalg.svd(U_flat)
        # If only one significant singular value, it's likely a product
        return np.sum(s > self.tolerance) <= 2
    
    def _extract_product_unitaries(self, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract single-qubit unitaries from a product state."""
        # Simple extraction assuming U = U1 ⊗ U2
        U1 = U[:2, :2]
        U2 = U[2:, 2:]
        
        # Normalize
        U1 = U1 / np.sqrt(np.linalg.det(U1))
        U2 = U2 / np.sqrt(np.linalg.det(U2))
        
        return U1, U2
    
    def _general_two_qubit_decomposition(self, U: np.ndarray, qubits: List[int]) -> List[Gate]:
        """General two-qubit decomposition using CZ gates."""
        gates = []
        
        # This is a simplified implementation
        # A full implementation would use the KAK decomposition more rigorously
        
        # For demonstration, we'll use a basic approach:
        # 1. Apply single-qubit rotations
        # 2. Apply CZ if needed
        # 3. Apply more single-qubit rotations
        
        # Check if the matrix is close to identity
        if np.allclose(U, np.eye(4), atol=self.tolerance):
            return gates
        
        # Apply CZ and single-qubit corrections
        # This is a placeholder - real implementation would be more sophisticated
        if not np.allclose(U, np.eye(4), atol=1e-3):
            gates.append(Gate("CZ", qubits))
            
            # Add corrective single-qubit rotations (simplified)
            if abs(U[0, 0] - 1) > self.tolerance:
                angle = np.angle(U[0, 0])
                if abs(angle) > self.tolerance:
                    gates.append(Gate("RZ", [qubits[0]], [angle]))
        
        return gates
    
    def controlled_decomposition(self, U: np.ndarray, control_qubits: List[int], 
                                target_qubit: int) -> List[Gate]:
        """
        Decompose a controlled unitary using V-chain decomposition.
        """
        gates = []
        n_controls = len(control_qubits)
        
        if n_controls == 0:
            # No controls - just decompose the unitary
            if U.shape == (2, 2):
                gates.extend(self.euler_zyz_decomposition(U, target_qubit))
            return gates
        
        elif n_controls == 1:
            # Single control - controlled unitary
            if U.shape == (2, 2):
                # Find the rotation that implements the controlled operation
                # This is a simplified implementation
                gates.extend(self._controlled_single_qubit(U, control_qubits[0], target_qubit))
            return gates
        
        else:
            # Multiple controls - use V-chain decomposition
            return self._multi_controlled_decomposition(U, control_qubits, target_qubit)
    
    def _controlled_single_qubit(self, U: np.ndarray, control: int, target: int) -> List[Gate]:
        """Decompose a controlled single-qubit unitary."""
        gates = []
        
        # Use standard controlled decomposition
        # C-U = I ⊗ V1 + |1⟩⟨1| ⊗ V2, where V1*V2 = U
        
        # For simplicity, we'll use a basic decomposition
        # In practice, this would use more sophisticated methods
        
        # First, apply single-qubit rotations to target
        euler_gates = self.euler_zyz_decomposition(U, target)
        
        # Convert to controlled versions using CZ
        for gate in euler_gates:
            if gate.name == "RZ":
                # Controlled RZ can be implemented with CZ
                if abs(gate.params[0]) > self.tolerance:
                    gates.append(Gate("RZ", [target], [gate.params[0]/2]))
                    gates.append(Gate("CZ", [control, target]))
                    gates.append(Gate("RZ", [target], [-gate.params[0]/2]))
            elif gate.name == "RY":
                # Controlled RY is more complex
                if abs(gate.params[0]) > self.tolerance:
                    gates.append(Gate("RY", [target], [gate.params[0]/2]))
                    gates.append(Gate("CZ", [control, target]))
                    gates.append(Gate("RY", [target], [-gate.params[0]/2]))
                    gates.append(Gate("CZ", [control, target]))
        
        return gates
    
    def _multi_controlled_decomposition(self, U: np.ndarray, control_qubits: List[int], 
                                       target_qubit: int) -> List[Gate]:
        """V-chain decomposition for multi-controlled unitaries."""
        gates = []
        n_controls = len(control_qubits)
        
        # V-chain requires auxiliary qubits, but we'll simulate the effect
        # This is a simplified version
        
        # For demonstration, we'll use a basic approach
        # Real V-chain would use Toffoli decomposition
        
        # Apply pairwise CZ gates between controls
        for i in range(n_controls - 1):
            gates.append(Gate("CZ", [control_qubits[i], control_qubits[i+1]]))
        
        # Apply controlled operation to target
        gates.append(Gate("CZ", [control_qubits[-1], target_qubit]))
        
        # Apply corrections
        if U.shape == (2, 2):
            euler_gates = self.euler_zyz_decomposition(U, target_qubit)
            gates.extend(euler_gates)
        
        return gates
    
    def decompose_csd_output(self, L: np.ndarray, C: np.ndarray, S: np.ndarray, 
                           R: np.ndarray) -> List[Gate]:
        """
        Main function to decompose CSD output into native gates.
        
        Args:
            L, C, S, R: CSD decomposition matrices where U = L @ [[C, -S], [S, C]] @ R†
        """
        self.reset()
        n = L.shape[0]
        n_qubits = int(np.log2(n))
        half_n = n // 2
        
        # Step 1: Decompose right unitary R†
        R_dagger = R.conj().T
        self._decompose_block_diagonal(R_dagger, "R_dagger")
        
        # Step 2: Decompose CS ladder (middle part)
        self._decompose_cs_ladder(C, S)
        
        # Step 3: Decompose left unitary L
        self._decompose_block_diagonal(L, "L")
        
        return self.gate_sequence
    
    def _decompose_block_diagonal(self, U: np.ndarray, label: str = ""):
        """Decompose a block diagonal unitary matrix."""
        n = U.shape[0]
        half_n = n // 2
        
        # Extract blocks
        U_upper = U[:half_n, :half_n]
        U_lower = U[half_n:, half_n:]
        
        # Decompose upper block
        if not np.allclose(U_upper, np.eye(half_n), atol=self.tolerance):
            upper_gates = self._decompose_unitary_block(U_upper, list(range(half_n)))
            self.gate_sequence.extend(upper_gates)
        
        # Decompose lower block
        if not np.allclose(U_lower, np.eye(half_n), atol=self.tolerance):
            lower_gates = self._decompose_unitary_block(U_lower, list(range(half_n, n)))
            self.gate_sequence.extend(lower_gates)
    
    def _decompose_unitary_block(self, U: np.ndarray, qubits: List[int]) -> List[Gate]:
        """Decompose a unitary block on specified qubits."""
        gates = []
        
        if U.shape == (1, 1):
            # Scalar - just global phase
            phase = np.angle(U[0, 0])
            if abs(phase) > self.tolerance:
                gates.append(Gate("RZ", [qubits[0]], [phase]))
        
        elif U.shape == (2, 2):
            # Single qubit
            gates.extend(self.euler_zyz_decomposition(U, qubits[0]))
        
        elif U.shape == (4, 4):
            # Two qubits
            gates.extend(self.kak_decomposition(U, qubits))
        
        else:
            # Larger blocks - recursive decomposition
            # This would typically use recursive CSD or other methods
            # For simplicity, we'll approximate with pairwise decompositions
            gates.extend(self._approximate_large_unitary(U, qubits))
        
        return gates
    
    def _decompose_cs_ladder(self, C: np.ndarray, S: np.ndarray):
        """Decompose the CS ladder into multi-controlled Y rotations."""
        # Extract CS angles
        cos_vals = np.diag(C)
        sin_vals = np.diag(S)
        angles = np.arctan2(sin_vals, cos_vals)
        
        # Apply multi-controlled Y rotations
        for i, theta in enumerate(angles):
            if abs(theta) > self.tolerance:
                # Multi-controlled Y rotation
                control_qubits = list(range(i))
                target_qubit = i
                
                # Convert to native gates
                if i == 0:
                    # No controls - just RY
                    self.add_gate("RY", [target_qubit], [2 * theta])
                else:
                    # Multi-controlled case
                    mcry_gates = self.controlled_decomposition(
                        np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]]),
                        control_qubits, target_qubit
                    )
                    self.gate_sequence.extend(mcry_gates)
    
    def _approximate_large_unitary(self, U: np.ndarray, qubits: List[int]) -> List[Gate]:
        """Approximate decomposition for large unitary matrices."""
        gates = []
        
        # This is a placeholder for large matrix decomposition
        # In practice, you would use more sophisticated methods
        
        # For now, just add identity (no gates)
        return gates
    
    def optimize_gate_sequence(self) -> List[Gate]:
        """Optimize the gate sequence by combining adjacent gates."""
        optimized = []
        
        i = 0
        while i < len(self.gate_sequence):
            current_gate = self.gate_sequence[i]
            
            # Look for adjacent gates on the same qubit that can be combined
            if (i + 1 < len(self.gate_sequence) and 
                current_gate.qubits == self.gate_sequence[i + 1].qubits and
                current_gate.name.startswith('R') and 
                self.gate_sequence[i + 1].name.startswith('R')):
                
                next_gate = self.gate_sequence[i + 1]
                
                # Combine rotations around the same axis
                if current_gate.name == next_gate.name:
                    combined_angle = current_gate.params[0] + next_gate.params[0]
                    if abs(combined_angle) > self.tolerance:
                        optimized.append(Gate(current_gate.name, current_gate.qubits, [combined_angle]))
                    i += 2
                    continue
            
            # No combination possible
            optimized.append(current_gate)
            i += 1
        
        return optimized
    
    def print_gate_sequence(self):
        """Print the gate sequence in a readable format."""
        print(f"Gate sequence ({len(self.gate_sequence)} gates):")
        for i, gate in enumerate(self.gate_sequence):
            print(f"  {i+1:2d}: {gate}")
    
    def get_gate_counts(self) -> Dict[str, int]:
        """Get counts of each gate type."""
        counts = {}
        for gate in self.gate_sequence:
            counts[gate.name] = counts.get(gate.name, 0) + 1
        return counts
