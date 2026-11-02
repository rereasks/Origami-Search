"""
visualize.py
------------
User-friendly 3D visualization of the Origami structure 
using matplotlib's 3D toolkit for easy side-by-side comparison.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_interactive_comparison(V_target, F_target, V_optimized, F_optimized):
    """
    Renders an interactive side-by-side 3D visualization of the input target 
    manifold and the optimized origami folded state.
    """
    fig = plt.figure(figsize=(14, 6))
    plt.suptitle("Aperiodic Conformal Origami Optimization Result", fontsize=16)
    
    # Left subplot: Target Geometry
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title("Input Target Surface", fontsize=14)
    ax1.plot_trisurf(V_target[:, 0], V_target[:, 1], V_target[:, 2], 
                     triangles=F_target, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Right subplot: Optimized Geometry
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("Optimized Folded State", fontsize=14)
    # Using a different colormap to highlight the creased/origami nature
    ax2.plot_trisurf(V_optimized[:, 0], V_optimized[:, 1], V_optimized[:, 2], 
                     triangles=F_optimized, cmap='plasma', edgecolor='black', linewidth=0.2, alpha=0.9)
    
    # Equalize aspect ratios for better viewing
    for ax in [ax1, ax2]:
        ax.set_box_aspect((1, 1, 0.5))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(False)
        ax.view_init(elev=30, azim=45)
        
    plt.tight_layout()
    
    print("\n>>> Launching interactive 3D Viewer...")
    print(">>> (Close the window to complete the pipeline script)")
    plt.show()

def render_comparison_image(V_target, F_target, V_optimized, F_optimized, filename="output/comparison_render.png"):
    """
    Saves the side-by-side 3D visualization as an image.
    """
    fig = plt.figure(figsize=(14, 6))
    plt.suptitle("Aperiodic Conformal Origami Result", fontsize=16)
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title("Input Target Surface")
    ax1.plot_trisurf(V_target[:, 0], V_target[:, 1], V_target[:, 2], 
                     triangles=F_target, cmap='viridis', edgecolor='none', alpha=0.8)
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("Optimized Folded State")
    ax2.plot_trisurf(V_optimized[:, 0], V_optimized[:, 1], V_optimized[:, 2], 
                     triangles=F_optimized, cmap='plasma', edgecolor='black', linewidth=0.2, alpha=0.9)
    
    for ax in [ax1, ax2]:
        ax.set_box_aspect((1, 1, 0.5))
        ax.view_init(elev=30, azim=45)
        ax.axis('off')
        
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
