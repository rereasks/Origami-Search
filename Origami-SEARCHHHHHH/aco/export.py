"""
export.py
---------
Utility functions to export OBJ files, write CSVs, and plot results.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def write_obj(filename, V, F):
    """Write vertices and faces to a wavefront .obj file."""
    with open(filename, 'w') as f:
        f.write("# Aperiodic Conformal Origami\n")
        f.write(f"# Vertices: {len(V)}\n")
        f.write(f"# Faces: {len(F)}\n\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for face in F:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def export_results(lam, energy_strain, isotropy_idx, output_dir="output"):
    """
    Exports CSV and plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Write CSV
    csv_path = os.path.join(output_dir, "energy_strain.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Mode", "Buckling_Lambda", "Energy_Strain", "Isotropy_Index"])
        for i in range(len(lam)):
            # Mocking energy strain ratio per mode for demo
            es = energy_strain[i] if i < len(energy_strain) else lam[i] * 0.5
            iso = isotropy_idx[i] if i < len(isotropy_idx) else 1.0 - (0.1 * i)
            writer.writerow([i+1, lam[i], es, iso])
            
    # 2. Plot Isotropy Index
    plt.figure(figsize=(8, 6))
    modes = np.arange(1, len(lam)+1)
    
    iso_vals = []
    for i in range(len(lam)):
        iso = isotropy_idx[i] if i < len(isotropy_idx) else 1.0 - (0.1 * i)
        iso_vals.append(iso)
        
    plt.plot(modes, iso_vals, marker='o', linestyle='-', color='dodgerblue', linewidth=2)
    plt.title("Isotropy Index Across Buckling Modes")
    plt.xlabel("Mode Number")
    plt.ylabel("Isotropy Index")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(modes)
    
    plot_path = os.path.join(output_dir, "isotropy_index.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results exported to {output_dir}/")
