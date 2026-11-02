"""
ui_visualize.py
---------------
Advanced Plotly-based 3D visualization for the Streamlit dashboard.
"""

import plotly.graph_objects as go
import numpy as np

def create_interactive_mesh(V, F, buckling_modes=None):
    """
    Renders the 3D topology using Plotly. 
    Highlights "weak" and "strong" points using the buckling displacement mode.
    Draws structural creases (fold lines) distinctly.
    """
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    
    # Extract structural vulnerability from buckling mode 1 (if available)
    if buckling_modes is not None and buckling_modes.shape[1] > 0:
        # Convert flat [3*N] array into [N, 3] displacements
        mode1 = buckling_modes[:, 0].reshape(-1, 3)
        # Magnitude of displacement
        displacement = np.linalg.norm(mode1, axis=1)
        # Strong points (blue, low displacement) vs Weak points (red, high displacement)
        color_vals = displacement
        colorscale = 'RdYlBu_r' 
        colorbar_title = "Vulnerability<br>(Buckling Mode)"
    else:
        # Fallback coloring based on Z-height
        color_vals = z
        colorscale = 'Viridis'
        colorbar_title = "Z Elevation"

    # 1. Plot the Main Mesh Surface
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=F[:, 0], j=F[:, 1], k=F[:, 2],
        intensity=color_vals,
        colorscale=colorscale,
        colorbar_title=colorbar_title,
        opacity=0.9,
        flatshading=True,
        name="Origami Skin"
    )

    # 2. Extract and Plot Creases (Wireframe)
    edges = set()
    for f in F:
        for k in range(3):
            edges.add(tuple(sorted([f[k], f[(k+1)%3]])))
            
    edge_x = []
    edge_y = []
    edge_z = []
    for u, v in edges:
        edge_x.extend([x[u], x[v], None])
        edge_y.extend([y[u], y[v], None])
        edge_z.extend([z[u], z[v], None])
        
    creases = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='black', width=3),
        name="Crease Folds"
    )

    # Assemble Plot
    fig = go.Figure(data=[mesh, creases])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.01, y=0.99),
    )
    
    return fig
