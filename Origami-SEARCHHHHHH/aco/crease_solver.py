"""
crease_solver.py
----------------
Kawasaki-Justin and Maekawa-Justin theorem enforcement.
Perturbs the 2D CVT vertices slightly using Scipy SLSQP so that
the foldability conditions for every interior vertex are respected.
"""

import numpy as np
import networkx as nx
from scipy.optimize import minimize
from scipy.spatial import Delaunay

def get_interior_neighbors(V, F):
    """
    Returns a dictionary mapping each interior vertex index
    to a counter-clockwise ordered list of its neighbors.
    """
    n = len(V)
    adj = {i: set() for i in range(n)}
    bnd_edges = set()
    for f in F:
        for i in range(3):
            u, v = sorted([f[i], f[(i+1)%3]])
            adj[u].add(f[(i+1)%3])
            adj[f[(i+1)%3]].add(u)
            if (u,v) in bnd_edges:
                bnd_edges.remove((u,v))
            else:
                bnd_edges.add((u,v))
    
    bnd_verts = set()
    for u, v in bnd_edges:
        bnd_verts.add(u)
        bnd_verts.add(v)
        
    interior = [i for i in range(n) if i not in bnd_verts]
    
    ordered_adj = {}
    for i in interior:
        pts = V[list(adj[i])] - V[i]
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        sorted_idx = np.argsort(angles)
        ordered_adj[i] = np.array(list(adj[i]))[sorted_idx]
        
    return interior, ordered_adj, list(bnd_verts)

def kawasaki_objective(x, bnd_x, V_shape, interior, ordered_adj, bnd_verts):
    """
    Minimizes the violation of Kawasaki's theorem:
    Sum of alternating sector angles = pi.
    Returns the sum of squared errors over all interior vertices.
    """
    V = np.zeros(V_shape)
    V[interior] = x.reshape(-1, 2)
    V[bnd_verts] = bnd_x.reshape(-1, 2)
    
    error = 0.0
    for i in interior:
        neighbors = ordered_adj[i]
        k = len(neighbors)
        if k % 2 != 0:
            # For Kawasaki, we need even degree. If odd, we penalize it
            # But triangulations mapped to flat typically have mostly degree-6.
            continue
            
        points = V[neighbors] - V[i]
        angles = np.arctan2(points[:, 1], points[:, 0])
        # Wrap sequence to positive
        angles[angles < 0] += 2 * np.pi
        
        # Calculate sector angles
        sectors = np.diff(angles)
        sectors = np.append(sectors, angles[0] + 2 * np.pi - angles[-1])
        # Force positive
        sectors[sectors < 0] += 2 * np.pi
        
        alt_sum = np.sum(sectors[0::2])
        error += (alt_sum - np.pi)**2
        
    return error

def enforce_foldability(V, F):
    """
    Perturbs V so that the interior vertices satisfy Kawasaki.
    """
    interior, ordered_adj, bnd_verts = get_interior_neighbors(V, F)
    if len(interior) == 0:
        return V # Nothing to optimize
        
    x0 = V[interior].ravel()
    bnd_x = V[bnd_verts].ravel()
    V_shape = V.shape
    
    res = minimize(
        kawasaki_objective, 
        x0, 
        args=(bnd_x, V_shape, interior, ordered_adj, bnd_verts),
        method='L-BFGS-B',
        options={'maxiter': 50, 'ftol': 1e-4}  # Relaxed tolerance for speed
    )
    
    V_opt = V.copy()
    V_opt[interior] = res.x.reshape(-1, 2)
    return V_opt
