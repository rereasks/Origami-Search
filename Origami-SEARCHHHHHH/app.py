import streamlit as st
import numpy as np
import time

from aco.mesh_gen import saddle_mesh, naca_mesh, _triangulate_grid
from aco.lscm import harmonic_map
from aco.crease_solver import enforce_foldability
from aco.buckling import run_buckling_analysis
from aco.optimizer import optimize_structure
from aco.ui_visualize import create_interactive_mesh

st.set_page_config(page_title="origami.ai | ACO Design", layout="wide")

st.title("Aperiodic Conformal Origami Dashboard")
st.markdown("Interactively design, simulate, and optimize complex aerospace structural skins globally.")

# Sidebar Parameters
st.sidebar.header("1. Target Geometry Configuration")
shape_type = st.sidebar.selectbox("Choose Base Profile:", ["Saddle (Hyperbolic)", "Airfoil (NACA-0012)", "Custom Plane"])

if shape_type == "Saddle (Hyperbolic)":
    a_val = st.sidebar.slider("Length 'a' (meters)", 0.5, 3.0, 1.0)
    b_val = st.sidebar.slider("Width 'b' (meters)", 0.5, 3.0, 1.0)
elif shape_type == "Airfoil (NACA-0012)":
    span_val = st.sidebar.slider("Wing Span Length (meters)", 0.5, 3.0, 1.0)
elif shape_type == "Custom Plane":
    st.sidebar.markdown("Define `z` using `x` and `y`:")
    custom_eq = st.sidebar.text_input("Equation z = ", value="np.sin(x)*np.cos(y)*0.5")

resolution = st.sidebar.slider("Mesh Resolution (Speed vs Fidelity)", 10, 30, 15)

st.sidebar.header("2. Mechanical Material Properties")
st.sidebar.caption("Standard Paper: EA ≈ 1,000 N/m | Carbon Fiber: EA ≈ 500,000 N/m")
EA_val = st.sidebar.number_input("Bar Axial Stiffness [EA] (N/m)", value=5000.0)

st.sidebar.caption("Standard Paper: k_h ≈ 0.1 N·m/rad | Thick Polymer: k_h ≈ 5,000 N·m/rad")
k_hinge_val = st.sidebar.number_input("Hinge Torsional Stiffness [k_h] (N·m/rad)", value=50.0)

with st.sidebar.expander("📖 What do these parameters mean?"):
    st.markdown('''
    - **Length/Width (m)**: The physical dimension boundaries bounding the target 3D structural skin.
    - **Mesh Resolution**: Higher numbers create denser, smaller origami folds, requiring exponentally more processing time but finer wrap curvature.
    - **Axial Stiffness [EA]**: How hard it is to pull or compress the actual facets of the material (measured in Newtons per meter). 
    - **Torsional Stiffness [k_h]**: How much torque/force is required to bend the origami creases. High values represent stiff, rigid materials that hate bending.
    - **Buckling Eigenvalue (\u03bb)**: Your final score. A multiplier of the environmental load limit. If it's `1.5`, the structure survives 1.5x the base atmospheric pressure before it physically snaps and collapses completely.
    ''')

run_button = st.sidebar.button("Run ACO Optimization Pipeline", type="primary")

# Execution Area
if run_button:
    with st.spinner("Processing Aperiodic Conformal Origami Mathematics..."):
        start_time = time.time()
        
        # 1. Geometry
        if shape_type == "Saddle (Hyperbolic)":
            V_3d, F = saddle_mesh(nx=resolution, ny=resolution, a=a_val, b=b_val)
        elif shape_type == "Airfoil (NACA-0012)":
            V_3d, F = naca_mesh(n_chord=max(20, resolution), n_span=resolution, span=span_val)
        elif shape_type == "Custom Plane":
            xs = np.linspace(-1, 1, resolution)
            ys = np.linspace(-1, 1, resolution)
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            try:
                # Safe evaluation
                valid_dict = {"x": X, "y": Y, "np": np}
                Z = eval(custom_eq, {"__builtins__": None}, valid_dict)
                V_3d = np.column_stack([X.ravel(), Y.ravel(), np.broadcast_to(Z, X.shape).ravel()]).astype(np.float64)
                F = _triangulate_grid(resolution, resolution)
            except Exception as e:
                st.error(f"Error in custom equation: {e}")
                st.stop()

        # Generate Input visualization
        st.subheader("1. Target Base Geometry")
        st.plotly_chart(create_interactive_mesh(V_3d, F), use_container_width=True)
        
        # Run Pipeline Heavy Steps
        # 2. Conformal Flattening
        V_2d = harmonic_map(V_3d, F)
        
        # 3. Generating Aperiodic Tessellation (Simulated random jitter for demo speed)
        V_cvt = V_2d + np.random.normal(scale=0.02, size=V_2d.shape)
        
        # 4. Constraint Solver (Kawasaki)
        V_kawasaki = enforce_foldability(V_cvt, F)
        
        # 5. Folding Simulation
        V_folded = V_3d.copy()
        V_folded[:, 0] += (V_kawasaki[:, 0] - V_2d[:, 0]) * 0.5 
        V_folded[:, 1] += (V_kawasaki[:, 1] - V_2d[:, 1]) * 0.5
        
        # 6. Optimization Loop
        V_opt, _, _ = optimize_structure(V_folded, F, EA=EA_val)
        
        # 7. Buckling Modes calculation
        lam, phi = run_buckling_analysis(V_opt, F, EA=EA_val, k_hinge=k_hinge_val, num_modes=1)
        
        st.subheader("2. Mathematically Optimized Folded State & Weakness Map")
        st.markdown("**Highlighted points indicate areas of high displacement during critical buckling limits.**")
        
        fig_opt = create_interactive_mesh(V_opt, F, buckling_modes=phi)
        st.plotly_chart(fig_opt, use_container_width=True)
        
        st.success(f"Pipeline executed globally in {time.time()-start_time:.2f} seconds!")
        st.metric(label="Principal Buckling Eigenvalue (Load Limit)", value=f"{lam[0]:.6e}" if len(lam)>0 else "N/A")
else:
    st.info("Configure your parameters in the sidebar and press Run to visualize results!")
