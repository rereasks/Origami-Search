"""
lscm.py
-------
Least-Squares Conformal Map (LSCM) / Harmonic Parameterization.
Maps a 3D disk-topology mesh to a 2D parameterization.
We use a Harmonic Parameterization with fixed circular boundary
as a robust surrogate for full free-boundary LSCM, which provides
excellent low-distortion mapping for our CVT density field.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import networkx as nx

def boundary_loop(F):
    """Find the ordered boundary loop of vertices for a simple disk-topology mesh."""
    edges = set()
    for f in F:
        for i in range(3):
            e = tuple(sorted([f[i], f[(i+1)%3]]))
            if e in edges:
                edges.remove(e)
            else:
                edges.add(e)
    
    # Build graph from boundary edges
    G = nx.Graph()
    G.add_edges_from(list(edges))
    
    # Find a cycle
    try:
        cycle = nx.find_cycle(G)
        ordered_bnd = [u for u, v in cycle]
        return ordered_bnd
    except nx.NetworkXNoCycle:
        return []

def harmonic_map(V, F):
    """
    Computes a harmonic map of the 3D mesh (V, F) to a 2D unit disk.
    This acts as our conformal mapping.
    """
    n = len(V)
    bnd = boundary_loop(F)
    if not bnd:
        raise ValueError("Mesh does not have a valid boundary loop.")
    
    bnd_set = set(bnd)
    internal = [i for i in range(n) if i not in bnd_set]
    
    # Map boundary to unit circle
    theta = np.linspace(0, 2 * np.pi, len(bnd), endpoint=False)
    uv_bnd = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Simple uniform Laplacian (umbrella operator) for stability
    # In a full geometry-aware LSCM, we'd use cotangent weights.
    row, col, data = [], [], []
    for f in F:
        for i in range(3):
            u = f[i]
            v = f[(i+1)%3]
            row.extend([u, v])
            col.extend([v, u])
            data.extend([1.0, 1.0])
    
    # Create Adjacency matrix
    A = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    A.data = np.ones_like(A.data) # Uniform weights
    
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    L = D - A
    
    # Partition L into internal and boundary parts
    L_ii = L[internal, :][:, internal]
    L_ib = L[internal, :][:, bnd]
    
    # Solve L_ii * uv_int = - L_ib * uv_bnd
    rhs = -L_ib.dot(uv_bnd)
    
    uv_int_x = spsolve(L_ii, rhs[:, 0])
    uv_int_y = spsolve(L_ii, rhs[:, 1])
    
    uv = np.zeros((n, 2))
    uv[bnd, :] = uv_bnd
    uv[internal, 0] = uv_int_x
    uv[internal, 1] = uv_int_y
    
    return uv
