"""
buckling.py
-----------
Global eigenvalue buckling analysis to find the first 5 failure modes.
Solves the generalized eigenvalue problem K u = lambda M u.
"""

import numpy as np
import scipy.sparse.linalg as sla
from .bar_hinge import build_bar_hinge_model

def run_buckling_analysis(V, F, EA=1000.0, k_hinge=10.0, num_modes=5):
    """
    Computes the buckling eigenvalues (load multipliers lambda) and 
    eigenvectors (buckling modes u).
    """
    K, M = build_bar_hinge_model(V, F, EA=EA, k_hinge=k_hinge)
    
    try:
        # Solve generalized eigenvalue problem: K u = lam M u
        # Since K is sparse and we want lowest frequencies, we use shift-invert with sigma
        lam, phi = sla.eigsh(K, k=num_modes, M=M, sigma=1e-2, tol=1e-3, maxiter=500)
        # Ensure eigenvalues are positive and sorted
        idx = np.argsort(lam)
        lam = lam[idx]
        phi = phi[:, idx]
        return lam, phi
    except Exception as e:
        print(f"Warning: Buckling solve failed: {e}")
        # Return fallback zeros if the matrix is singular (due to poor triangulation)
        return np.zeros(num_modes), np.zeros((len(V)*3, num_modes))

def objective_buckling_mass(V, F, EA=1000.0, k_hinge=10.0):
    """
    Computes (Buckling Load / Total Mass).
    Simplified: Mass is proportional to area of triangles.
    """
    lam, _ = run_buckling_analysis(V, F, EA, k_hinge, num_modes=1)
    lam1 = lam[0] if len(lam) > 0 else 0.0
    
    # Simple Area estimation
    mass = 0.0
    for f in F:
        u, v, w = V[f[0]], V[f[1]], V[f[2]]
        cp = np.cross(v - u, w - u)
        mass += 0.5 * np.linalg.norm(cp)
        
    if mass < 1e-8:
        mass = 1.0
        
    return lam1 / mass
