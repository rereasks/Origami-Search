"""
mesh_gen.py
-----------
Generate target 3-D meshes for the ACO pipeline.

Surfaces
~~~~~~~~
• Saddle  – hyperbolic paraboloid  z = (x²/a²) − (y²/b²)
• NACA    – extruded NACA-0012 airfoil (finite-span wing section)

Both functions return
    V : (N, 3) float64   vertex positions
    F : (M, 3) int32     triangle faces
"""

import numpy as np
from scipy.spatial import Delaunay


# ── helpers ──────────────────────────────────────────────────────────────────

def _triangulate_grid(nx: int, ny: int) -> np.ndarray:
    """Return (2*(nx-1)*(ny-1), 3) face indices for a regular nx×ny grid."""
    idx = np.arange(nx * ny).reshape(ny, nx)
    # lower-left triangle + upper-right triangle for every quad
    i0 = idx[:-1, :-1].ravel()
    i1 = idx[:-1, 1:].ravel()
    i2 = idx[1:, :-1].ravel()
    i3 = idx[1:, 1:].ravel()
    lower = np.column_stack([i0, i1, i2])
    upper = np.column_stack([i1, i3, i2])
    return np.vstack([lower, upper]).astype(np.int32)


# ── public API ───────────────────────────────────────────────────────────────

def saddle_mesh(
    nx: int = 40,
    ny: int = 40,
    x_range: tuple = (-1.0, 1.0),
    y_range: tuple = (-1.0, 1.0),
    a: float = 1.0,
    b: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hyperbolic-paraboloid saddle surface  z = (x/a)² − (y/b)².

    Parameters
    ----------
    nx, ny  : grid resolution
    x_range, y_range : extents
    a, b    : shape parameters

    Returns
    -------
    V : (N, 3) vertex array
    F : (M, 3) face array
    """
    xs = np.linspace(*x_range, nx)
    ys = np.linspace(*y_range, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = (X / a) ** 2 - (Y / b) ** 2

    V = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)
    F = _triangulate_grid(nx, ny)
    return V, F


def _naca_0012(t: np.ndarray) -> np.ndarray:
    """
    NACA-0012 half-thickness at chord-wise positions t ∈ [0, 1].
    Returns y_t (thickness, symmetric about chord line).
    """
    return (
        0.2969 * np.sqrt(t)
        - 0.1260 * t
        - 0.3516 * t**2
        + 0.2843 * t**3
        - 0.1015 * t**4
    ) * (0.12 / 0.2)  # scale so max thickness = 12 % chord


def naca_mesh(
    n_chord: int = 40,
    n_span: int = 20,
    span: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finite-span wing section (NACA-0012 extruded along z).

    Parameters
    ----------
    n_chord : points around the airfoil section
    n_span  : points along the span
    span    : total span length

    Returns
    -------
    V : (N, 3) vertex array   (x=chord, y=thickness, z=span)
    F : (M, 3) face array
    """
    # chord-wise parameter (cosine spacing for leading/trailing-edge refinement)
    beta = np.linspace(0.0, np.pi, n_chord // 2 + 1)
    t_upper = 0.5 * (1 - np.cos(beta))          # 0 → 1 (upper surface)
    t_lower = t_upper[::-1]                      # 1 → 0 (lower surface)
    t_section = np.concatenate([t_upper, t_lower[1:-1]])

    x_upper = t_upper
    y_upper = _naca_0012(t_upper)
    x_lower = t_lower
    y_lower = -_naca_0012(t_lower)
    x_sec = np.concatenate([x_upper, x_lower[1:-1]])
    y_sec = np.concatenate([y_upper, y_lower[1:-1]])

    n_sec = len(x_sec)
    zs = np.linspace(0.0, span, n_span)

    # replicate cross-section along span
    V = np.zeros((n_sec * n_span, 3), dtype=np.float64)
    for k, z in enumerate(zs):
        sl = slice(k * n_sec, (k + 1) * n_sec)
        V[sl, 0] = x_sec
        V[sl, 1] = y_sec
        V[sl, 2] = z

    # build faces: for each span strip connect adjacent sections
    F_list = []
    for k in range(n_span - 1):
        base = k * n_sec
        for j in range(n_sec):
            j_next = (j + 1) % n_sec
            # two triangles per quad
            a0 = base + j
            a1 = base + j_next
            b0 = base + n_sec + j
            b1 = base + n_sec + j_next
            F_list.append([a0, a1, b0])
            F_list.append([a1, b1, b0])

    F = np.array(F_list, dtype=np.int32)
    return V, F
