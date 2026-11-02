"""
ACO – Aperiodic Conformal Origami Inverse Design Pipeline
==========================================================
Modules
-------
mesh_gen     : Saddle / NACA-0012 target surface generation
lscm         : Least-Squares Conformal Map (scipy sparse, no igl required)
cvt          : Centroidal Voronoi Tessellation with curvature density weighting
crease_solver: Kawasaki–Justin & Maekawa–Justin constraint enforcement
bar_hinge    : Bar-and-Hinge structural model + Newton-Raphson solver
buckling     : Eigenvalue buckling analysis (first 5 modes)
optimizer    : Outer optimisation loop (vertex jitter + hinge stiffness)
export       : .obj / .stl export, CSV energy-strain, isotropy-index plot
"""
