# csd_decomposer.py

import numpy as np
from scipy.linalg import cossin

def csd_decompose(U):
    """
    Compute the Cosine–Sine Decomposition of a 2n×2n unitary matrix U.
    Returns L, C, S, R such that U ≈ L @ [[C, -S],[S, C]] @ R^H, with C^2+S^2=I.
    """
    m, n2 = U.shape
    assert m == n2 and m % 2 == 0, "U must be 2n x 2n"
    n = m // 2
    # Use SciPy's CSD routine (LAPACK) to get factors u, cs, vh
    u, cs_mat, vdh = cossin(U, p=n, q=n)
    # Extract diagonal C and S matrices
    C = cs_mat[:n, :n].copy()        # upper-left block (diagonal cosines)
    S = cs_mat[n:, :n].copy()        # lower-left block (diagonal sines)
    # Form L and R from returned factors
    L = u                            # block-diagonal [U1,0;0,U2]
    R = vdh.conj().T                 # block-diagonal [V1,0;0,V2]
    return L, C, S, R
