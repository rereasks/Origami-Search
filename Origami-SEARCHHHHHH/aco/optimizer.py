"""
optimizer.py
------------
Outer optimization loop iteratively improving a parametrized 
Jitter field and hinge stiffness to maximize the buckling criteria.
"""

import numpy as np
from scipy.optimize import minimize
from .buckling import objective_buckling_mass

def optimize_structure(V_base, F, EA=1000.0):
    """
    Optimizes a global vertex jitter amplitude and a uniform hinge
    stiffness scalar to maximize (Buckling Load / Mass).
    Variables:
      x[0] : jitter amplitude (spatial sinusoidal wave)
      x[1] : hinge stiffness scalar
    """
    
    N = len(V_base)
    # Define a spatial sinusoidal jitter basis to keep things smooth
    x_min, x_max = np.min(V_base[:,0]), np.max(V_base[:,0])
    y_min, y_max = np.min(V_base[:,1]), np.max(V_base[:,1])
    Lx = max(x_max - x_min, 1e-3)
    Ly = max(y_max - y_min, 1e-3)
    
    # Basis mode: z-displacement proportional to sin(2 pi x / Lx) * cos(2 pi y / Ly)
    J = np.zeros_like(V_base)
    J[:, 2] = np.sin(2 * np.pi * (V_base[:, 0] - x_min) / Lx) * np.cos(2 * np.pi * (V_base[:, 1] - y_min) / Ly)
    
    def objective(x):
        # maximize -> minimize negative
        jitter_amp, k_h_scalar = x
        V_perturbed = V_base + jitter_amp * J
        k_hinge = max(k_h_scalar, 1e-3) # ensure positivity
        
        # Calculate buckling score
        score = objective_buckling_mass(V_perturbed, F, EA=EA, k_hinge=k_hinge)
        
        # Penalize huge stiffness
        penalty = 0.01 * k_h_scalar**2
        
        return -(score - penalty)
        
    # Initial guess
    x0 = np.array([0.05, 10.0]) # 5% jitter, 10 base hinge stiffness
    
    res = minimize(
        objective, 
        x0, 
        method='Nelder-Mead', # derivative-free is safer here
        options={'maxiter': 20, 'disp': True}
    )
    
    best_amp, best_k_hinge = res.x
    V_opt = V_base + best_amp * J
    best_score = -res.fun
    
    return V_opt, best_k_hinge, best_score
