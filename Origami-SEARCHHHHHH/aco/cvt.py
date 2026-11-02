"""
cvt.py
------
Centroidal Voronoi Tessellation (CVT) Generator.
Creates a discrete non-uniform mesh based on Lloyd's algorithm.
Points are relaxed to the centroids of their Voronoi cells.
"""

import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from shapely.geometry import Polygon, Point

def generate_cvt(n_points=200, iterations=10, bounding_radius=1.0):
    """
    Generates a generic Centroidal Voronoi Tessellation within a disk.
    
    Returns
    -------
    points : (N, 2) array of optimized point coordinates.
    """
    # Initialize random points in a disk
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    r = np.sqrt(np.random.uniform(0, bounding_radius**2, n_points))
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # Bounding points to close the Voronoi diagrams at edges
    boundary_offset = bounding_radius * 1.5
    bounding_box = np.array([
        [-boundary_offset, -boundary_offset],
        [ boundary_offset, -boundary_offset],
        [ boundary_offset,  boundary_offset],
        [-boundary_offset,  boundary_offset]
    ])
    
    disk_poly = Point(0,0).buffer(bounding_radius)
    
    for _ in range(iterations):
        all_points = np.vstack([points, bounding_box])
        vor = Voronoi(all_points)
        
        new_points = []
        for i in range(n_points):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region or len(region) == 0:
                new_points.append(points[i])
                continue
                
            poly_coords = [vor.vertices[j] for j in region]
            cell = Polygon(poly_coords)
            intersection = cell.intersection(disk_poly)
            
            if intersection.is_empty:
                new_points.append(points[i])
            else:
                centroid = intersection.centroid
                new_points.append([centroid.x, centroid.y])
                
        points = np.array(new_points)
        
    return points
