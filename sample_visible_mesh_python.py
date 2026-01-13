#!/usr/bin/env python3
"""
Python-based visible mesh surface sampling for DeepSDF evaluation.
Replaces the C++ SampleVisibleMeshSurface executable that depends on Pangolin.
"""

import argparse
import logging
import numpy as np
import sys
from pathlib import Path

try:
    import trimesh
except ImportError:
    print("Installing required dependency: trimesh")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "trimesh"])
    import trimesh


# Use root logger directly - parent will configure
logger = logging.getLogger()


def compute_normalization_parameters(mesh: trimesh.Trimesh, buffer: float = 1.03):
    """
    Compute normalization parameters for a mesh.

    This matches the C++ ComputeNormalizationParameters behavior (Utils.cpp:109-168)
    by filtering out unused vertices when computing the bounding box.

    Returns:
        translation: Translation vector to center mesh (offset in C++ terminology)
        scale: Scale factor to normalize to unit cube
    """
    vertices = mesh.vertices

    # Find which vertices are actually used in faces
    # This matches C++ behavior in Utils.cpp:121-142
    used_vertices = set()
    for face in mesh.faces:
        used_vertices.update(face)

    # Filter to only used vertices
    used_mask = np.array([i in used_vertices for i in range(len(vertices))])
    used_verts = vertices[used_mask]

    # Find bounding box of used vertices only
    min_vals = used_verts.min(axis=0)
    max_vals = used_verts.max(axis=0)

    # Compute center
    center = (max_vals + min_vals) / 2.0

    # Compute max distance from center
    centered_verts = used_verts - center
    max_dist = np.linalg.norm(centered_verts, axis=1).max()

    # Scale with buffer
    scale = 1.0 / (max_dist * buffer)

    # Return negative center as translation (to center the mesh)
    return -center, scale


def normalize_mesh_with_params(mesh: trimesh.Trimesh, translation: np.ndarray, scale: float):
    """Apply normalization parameters to mesh."""
    mesh.vertices = (mesh.vertices + translation) * scale


def generate_equidistant_camera_views(num_views: int, radius: float) -> np.ndarray:
    """
    Generate camera positions evenly distributed on a sphere.

    Returns:
        Array of shape (num_views, 3) containing camera positions
    """
    views = np.zeros((num_views, 3))
    offset = 2.0 / num_views
    increment = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(num_views):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y * y)
        phi = (i + 1) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        views[i] = radius * np.array([x, y, z])

    return views


def sample_visible_mesh_surface(
    mesh: trimesh.Trimesh,
    num_samples: int = 100000,
    num_views: int = 100,
    camera_distance_mult: float = 1.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample visible surface points using ray-casting to approximate
    the C++ multi-view OpenGL rendering approach.

    This matches the behavior of SampleVisibleMeshSurface.cpp which:
    1. Renders mesh from 100 viewpoints using OpenGL
    2. Extracts visible surface points from each rendered frame
    3. Accumulates visible points across all views
    4. Samples from visible triangles for evaluation

    Args:
        mesh: Input mesh (should already be normalized)
        num_samples: Target number of samples
        num_views: Number of camera views (matches C++ default)
        camera_distance_mult: Multiplier for camera distance

    Returns:
        points: Sampled points (N, 3)
        normals: Corresponding normals (N, 3)
    """
    logger.info(f"Sampling visible surface with {num_views} views...")

    # Ensure normals are computed
    if not mesh.vertex_normals.shape[0] == mesh.vertices.shape[0]:
        mesh.fix_normals()
        mesh.recompute_vertex_normals()

    # Get mesh bounds
    max_dist = np.linalg.norm(mesh.vertices, axis=1).max()
    camera_distance = max_dist * camera_distance_mult

    # Generate camera positions (matches C++ EquiDistPointsOnSphere)
    views = generate_equidistant_camera_views(num_views, camera_distance)

    # Step 1: Determine which triangles are visible from multiple views
    # This approximates the C++ OpenGL framebuffer accumulation
    logger.info("Determining visible triangles from multiple views...")
    visible_triangles = set()

    for view_idx, view_point in enumerate(views):
        if (view_idx + 1) % 20 == 0 or view_idx == 0:
            logger.info(f"Processing view {view_idx + 1}/{len(views)}...")

        # Cast rays from view point to triangle centers
        # Triangles that are hit first (without occlusion) are visible
        triangle_centers = mesh.triangles_center
        ray_origins = np.tile(view_point, (len(triangle_centers), 1))
        ray_directions = triangle_centers - view_point
        ray_directions = ray_directions / (np.linalg.norm(ray_directions, axis=1, keepdims=True) + 1e-8)

        # Cast rays - triangles that are hit are potentially visible
        # Note: trimesh.ray.intersects_id returns the triangle indices that were hit
        ray_indices, ray_locations = mesh.ray.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            return_locations=True,
            multiple_hits=False,
        )

        # Track which triangles were visible from this view
        for i, tri_idx in enumerate(ray_indices):
            # Check if this triangle was actually hit (not occluded)
            # by verifying the hit location is close to the triangle center
            expected_dist = np.linalg.norm(triangle_centers[tri_idx] - view_point)
            actual_dist = np.linalg.norm(ray_locations[i] - view_point)

            # If the hit is close to expected, the triangle is visible
            if abs(expected_dist - actual_dist) < 1e-3:
                visible_triangles.add(tri_idx)

    logger.info(f"Found {len(visible_triangles)} visible triangles out of {len(mesh.faces)} total")

    if len(visible_triangles) == 0:
        logger.warning("No visible triangles found! Using all triangles instead.")
        visible_triangles = set(range(len(mesh.faces)))

    # Step 2: Sample from visible triangles only
    # This matches C++ SampleFromSurfaceInside behavior
    visible_face_list = list(visible_triangles)
    visible_faces = mesh.faces[visible_face_list]
    visible_tri_areas = mesh.area_faces[visible_face_list]

    # Build area-weighted CDF for sampling (matches C++ CDF approach)
    area_cdf = np.cumsum(visible_tri_areas)
    area_cdf /= area_cdf[-1]

    all_points = []
    all_normals = []

    # Sample in batches
    batch_size = min(10000, num_samples)

    while len(all_points) < num_samples:
        # Sample triangles by area (matches C++ uniform_real_distribution sampling)
        n_to_sample = min(batch_size, num_samples - len(all_points))
        tri_indices_local = np.searchsorted(area_cdf, np.random.rand(n_to_sample))
        tri_indices_local = np.minimum(tri_indices_local, len(visible_tri_areas) - 1)

        # Map back to original face indices
        tri_indices = np.array([visible_face_list[i] for i in tri_indices_local])
        selected_faces = mesh.faces[tri_indices]

        # Sample points within triangles using barycentric coordinates
        # This matches C++ SamplePointFromTriangle (Utils.cpp:93-106)
        r1 = np.random.rand(n_to_sample)
        r2 = np.random.rand(n_to_sample)

        # Transform to barycentric with sqrt for uniform distribution
        sqrt_r1 = np.sqrt(r1)
        a = 1 - sqrt_r1
        b = sqrt_r1 * (1 - r2)
        c = sqrt_r1 * r2

        # Compute points using barycentric interpolation
        tri_verts = mesh.vertices[selected_faces]
        points = (
            a[:, None] * tri_verts[:, 0, :]
            + b[:, None] * tri_verts[:, 1, :]
            + c[:, None] * tri_verts[:, 2, :]
        )

        # Interpolate vertex normals
        vert_normals = mesh.vertex_normals[selected_faces]
        normals = (
            a[:, None] * vert_normals[:, 0, :]
            + b[:, None] * vert_normals[:, 1, :]
            + c[:, None] * vert_normals[:, 2, :]
        )

        # Normalize interpolated normals
        norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norm_lengths + 1e-8)

        # Filter to remove points with NaN
        valid = ~(np.isnan(points).any(axis=1) | np.isnan(normals).any(axis=1))
        all_points.append(points[valid])
        all_normals.append(normals[valid])

    points = np.vstack(all_points)[:num_samples]
    normals = np.vstack(all_normals)[:num_samples]

    logger.info(f"Generated {len(points)} visible surface samples")

    return points, normals


def save_surface_samples_to_ply(points, normals, filename):
    """Save surface samples to PLY file."""
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")

        for p, n in zip(points, normals):
            f.write(f"{p[0]} {p[1]} {p[2]} {n[0]} {n[1]} {n[2]}\n")


def save_normalization_params(translation, scale, filename):
    """
    Save normalization parameters to NPZ file.

    NOTE: Uses 'offset' key to match C++ implementation (evaluate.py expects this).
    The variable is named 'translation' internally but saved as 'offset' for compatibility.
    """
    np.savez(filename, offset=translation, scale=scale)


def main():
    parser = argparse.ArgumentParser(
        description="Sample visible mesh surface for DeepSDF evaluation (Python version)"
    )
    parser.add_argument("-m", "--mesh", required=True, help="Input mesh file")
    parser.add_argument("-o", "--output", required=True, help="Output PLY file")
    parser.add_argument("-n", "--normals", help="Output NPZ file for normalization parameters")
    parser.add_argument("-s", "--samples", type=int, default=100000, help="Number of samples")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Loading mesh from {args.mesh}")

    # Load mesh
    mesh = trimesh.load(args.mesh, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            trimesh.Trimesh(**g.geometry) for g in mesh.geometry.values()
        )

    # Compute normalization parameters
    translation, scale = compute_normalization_parameters(mesh)
    logger.info(f"Normalization: translation={translation}, scale={scale}")

    # Apply normalization
    normalize_mesh_with_params(mesh, translation, scale)

    # Sample visible surface
    points, normals = sample_visible_mesh_surface(
        mesh=mesh,
        num_samples=args.samples,
        num_views=100,
    )

    # Save to PLY
    logger.info(f"Saving {len(points)} samples to {args.output}")
    save_surface_samples_to_ply(points, normals, args.output)

    # Save normalization parameters if requested
    if args.normals:
        logger.info(f"Saving normalization params to {args.normals}")
        save_normalization_params(translation, scale, args.normals)

    logger.info("Done!")


if __name__ == "__main__":
    main()
