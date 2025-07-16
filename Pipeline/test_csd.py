# test_csd.py

import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import norm
from csd_decomposer import csd_decompose

# Generate a random 2n×2n unitary matrix
n = 4
U = unitary_group.rvs(2*n)

# Compute the CSD
L, C, S, R = csd_decompose(U)

# Build the full CS block matrix [C, -S; S, C]
CS_block = np.block([[C, -S], [S, C]])

# Check reconstruction U ≈ L @ CS_block @ R^H
err1 = norm(U - L.dot(CS_block).dot(R.conj().T))
# Check that C^2 + S^2 ≈ I_n
err2 = norm(C.dot(C) + S.dot(S) - np.eye(n))

print(f"||U - L@CS@Rᴴ|| = {err1:.3e}")
print(f"||C^2 + S^2 - I|| = {err2:.3e}")

assert err1 < 1e-10, "Reconstruction error too large"
assert err2 < 1e-10, "C^2+S^2 not sufficiently identity"
print("CSD decomposition is validated successfully.")
#print(U)
#print(L, C, S, R)