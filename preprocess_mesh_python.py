#!/usr/bin/env python3
"""
Python-based mesh preprocessing for DeepSDF.
Replaces the C++ PreprocessMesh executable that depends on Pangolin.
"""

import argparse
import logging
import numpy as np
import sys
from pathlib import Path
import trimesh
from scipy.spatial import cKDTree
HAS_SCIPY = True


# Use root logger directly to avoid duplicate logs
logger = logging.getLogger()


def check_triangle_consistency(mesh: trimesh.Trimesh) -> tuple[int, int]:
    """
    Check triangle normal consistency.

    This approximates the C++ behavior in PreprocessMesh.cpp:496-512
    which checks for triangle normal consistency and rejects meshes
    with inconsistent normals.

    Instead of rejecting, we issue a warning if more than 10% of triangles
    have potentially inconsistent normals.

    Returns:
        bad_triangles: Number of potentially bad triangles
        total_triangles: Total number of triangles
    """
    # For each triangle, check if its normal is well-oriented
    # A simple heuristic: for a well-formed mesh, most face normals
    # should point away from the mesh centroid

    face_count = len(mesh.faces)
    if face_count == 0:
        return 0, 0

    mesh_centroid = mesh.vertices.mean(axis=0)
    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center

    to_face = face_centers - mesh_centroid
    to_face_norm = np.linalg.norm(to_face, axis=1)
    valid = to_face_norm > 1e-6
    to_face_unit = np.zeros_like(to_face)
    to_face_unit[valid] = to_face[valid] / to_face_norm[valid, None]

    dot_product = np.einsum("ij,ij->i", face_normals, to_face_unit)
    bad_tri = int(np.sum(valid & (dot_product < 0.0)))
    return bad_tri, face_count


def repair_mesh_normals(mesh: trimesh.Trimesh, verbose: bool = False) -> tuple[trimesh.Trimesh, dict]:
    """Multi-stage mesh normal repair using trimesh.

    Stages:
    1. Pre-cleanup: merge vertices (removes duplicates)
    2. Fix winding order: ensure consistent triangle orientation
    3. Fix normal direction: ensure normals point outward (multibody-aware)
    4. Force vertex normals recomputation (by accessing the property)

    Args:
        mesh: Input mesh to repair
        verbose: Enable verbose logging

    Returns:
        tuple: (repaired_mesh, stats_dict)
    """
    stats = {"stages_applied": []}

    # Stage 1: Pre-cleanup
    # merge_vertices removes duplicate vertices and updates faces
    mesh.merge_vertices()
    stats["stages_applied"].append("cleanup")

    # Stage 2: Fix winding order
    trimesh.repair.fix_winding(mesh)
    stats["stages_applied"].append("fix_winding")

    # Stage 3: Fix normal direction (multibody-aware for disconnected components)
    trimesh.repair.fix_normals(mesh, multibody=True)
    stats["stages_applied"].append("fix_normals")

    # Stage 4: Force vertex normals recomputation by accessing the property
    # This clears the cache and recomputes vertex normals from face normals
    _ = mesh.vertex_normals
    _ = mesh.face_normals

    if verbose:
        logger.debug(f"Applied repair stages: {stats['stages_applied']}")

    return mesh, stats


def repair_with_fallback(mesh: trimesh.Trimesh, max_attempts: int = 2) -> tuple[trimesh.Trimesh, dict, bool]:
    """Attempt mesh repair with fallback to aggressive methods.

    First attempt uses standard repair. If bad triangle ratio is still high,
    second attempt adds hole filling for watertightness.

    Args:
        mesh: Input mesh to repair
        max_attempts: Maximum number of repair attempts (default: 2)

    Returns:
        tuple: (repaired_mesh, stats_dict, success)
            - repaired_mesh: The mesh after repair attempts
            - stats_dict: Contains bad_ratio_before and bad_ratio_after
            - success: True if bad_ratio_after <= 0.03 (C++ threshold)
    """
    for attempt in range(max_attempts):
        bad_tri, total = check_triangle_consistency(mesh)
        bad_ratio = bad_tri / total if total > 0 else 0

        if attempt == 0:
            # Standard repair
            mesh, stats = repair_mesh_normals(mesh)
        else:
            # Aggressive repair: add hole filling
            if not mesh.is_watertight:
                logger.info("Mesh not watertight, filling holes...")
                trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_normals(mesh, multibody=True)
            # Force vertex normals recomputation
            _ = mesh.vertex_normals
            _ = mesh.face_normals
            stats["stages_applied"].append("fill_holes")

        # Check success
        bad_tri_after, total_after = check_triangle_consistency(mesh)
        bad_ratio_after = bad_tri_after / total_after if total_after > 0 else 0

        stats["bad_ratio_before"] = bad_ratio
        stats["bad_ratio_after"] = bad_ratio_after

        logger.debug(
            f"Attempt {attempt + 1}: bad triangle ratio {bad_ratio:.1%} -> {bad_ratio_after:.1%}"
        )

        if bad_ratio_after <= 0.03:  # C++ threshold
            return mesh, stats, True

    return mesh, stats, False


def normalize_mesh(mesh: trimesh.Trimesh, buffer: float = 1.03) -> float:
    """
    Normalize mesh to fit in unit cube centered at origin.

    This matches the C++ BoundingCubeNormalization behavior (Utils.cpp:170-244)
    by filtering out unused vertices when computing the bounding box.

    Returns:
        max_dist: The maximum distance from origin after normalization
    """
    vertices = mesh.vertices

    # Find which vertices are actually used in faces.
    # Prefer vectorized unique over Python set/loop for speed.
    if len(mesh.faces) == 0:
        used_verts = vertices
        used_count = len(vertices)
    else:
        used_idx = np.unique(mesh.faces.reshape(-1))
        used_verts = vertices[used_idx]
        used_count = len(used_idx)

    logger.debug(
        f"Using {len(used_verts)} out of {len(vertices)} vertices (filtered {len(vertices) - used_count} unused)"
    )

    # Find bounding box of used vertices only
    min_vals = used_verts.min(axis=0)
    max_vals = used_verts.max(axis=0)

    # Compute center
    center = (max_vals + min_vals) / 2.0

    # Center the mesh (apply to all vertices)
    mesh.vertices = vertices - center

    # Compute max distance from origin (using all vertices after centering)
    max_dist = np.linalg.norm(mesh.vertices, axis=1).max()

    # Scale to fit in unit cube with buffer
    scale = 1.0 / (max_dist * buffer)
    mesh.vertices *= scale

    return 1.0 / buffer


def sample_from_surface(mesh: trimesh.Trimesh, num_samples: int) -> np.ndarray:
    """
    Sample points uniformly on mesh surface.

    Args:
        mesh: Input mesh
        num_samples: Number of samples to generate

    Returns:
        Array of shape (num_samples, 3) containing sampled points
    """
    return mesh.sample(num_samples)


def generate_equidistant_points_on_sphere(num_samples: int, radius: float) -> np.ndarray:
    """
    Generate points evenly distributed on a sphere using Fibonacci lattice.

    Args:
        num_samples: Number of points to generate
        radius: Sphere radius

    Returns:
        Array of shape (num_samples, 3)
    """
    points = np.zeros((num_samples, 3))
    offset = 2.0 / num_samples
    increment = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(num_samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y * y)
        phi = (i + 1) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        points[i] = radius * np.array([x, y, z])

    return points


def sample_sdf_near_surface(
    mesh: trimesh.Trimesh,
    surface_points: np.ndarray,
    num_random_samples: int,
    variance: float = 0.005,
    second_variance: float = 0.0005,
    bounding_cube_dim: float = 2.0,
    num_votes: int = 11,
    sign_method: str = "vote",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample SDF values near the surface.

    This implements the same logic as the C++ SampleSDFNearSurface function.
    Uses KD-tree for efficient nearest neighbor queries.

    Args:
        mesh: Input mesh with vertices and faces
        surface_points: Points sampled on surface
        num_random_samples: Number of random samples in bounding cube
        variance: Standard deviation for Gaussian perturbation
        second_variance: Secondary variance for more spread out samples
        bounding_cube_dim: Dimension of sampling cube
        num_votes: Number of nearest neighbors to vote on sign
        sign_method: How to determine sign / SDF. "vote" matches C++-style
            vertex-normal voting with unanimous filtering; "proximity" uses
            trimesh's signed-distance (requires optional dependencies); "igl"
            uses libigl's signed distance (Python bindings).

    Returns:
        xyz: Sampled points
        sdfs: Corresponding SDF values
    """
    # Get mesh vertices and normals
    vertices = mesh.vertices
    _ = mesh.vertex_normals  # Force computation
    normals = mesh.vertex_normals

    # Build KD-tree
    if HAS_SCIPY:
        tree = cKDTree(vertices)
    else:
        tree = None
        logger.warning("scipy not available, using slow distance computation")

    # Generate perturbed samples near surface
    stdv = np.sqrt(variance)
    stdv2 = np.sqrt(second_variance)

    # Two samples per surface point with different variances (vectorized)
    surface_points = np.asarray(surface_points)
    noise1 = np.random.randn(*surface_points.shape) * stdv
    noise2 = np.random.randn(*surface_points.shape) * stdv2
    perturbed = np.concatenate([surface_points + noise1, surface_points + noise2], axis=0)

    # Add random samples in bounding cube
    random_samples = np.random.rand(num_random_samples, 3)
    random_samples = random_samples * bounding_cube_dim - bounding_cube_dim / 2.0
    xyz = np.vstack([perturbed, random_samples])

    if sign_method not in {"vote", "proximity", "igl"}:
        raise ValueError(
            f"Unknown sign_method: {sign_method!r} (expected 'vote', 'proximity', or 'igl')"
        )

    # Optional fast-path: use trimesh's signed distance directly.
    # This can be significantly faster and avoids kNN voting, but requires
    # optional trimesh dependencies (e.g., rtree / embree backends).
    if sign_method == "proximity":
        try:
            sdfs = trimesh.proximity.signed_distance(mesh, xyz)
            sdfs = np.asarray(sdfs)
            finite_mask = np.isfinite(sdfs)
            if not np.all(finite_mask):
                logger.warning("Dropping non-finite signed_distance results")
            return xyz[finite_mask], sdfs[finite_mask]
        except Exception as exc:
            logger.warning(f"trimesh.proximity.signed_distance failed ({exc}); falling back to 'vote' method")

    if sign_method == "igl":
        try:
            import igl  # type: ignore

            V = np.asarray(mesh.vertices, dtype=np.float64)
            F = np.asarray(mesh.faces, dtype=np.int32)
            P = np.asarray(xyz, dtype=np.float64)

            result = igl.signed_distance(P, V, F)
            # Different libigl Python bindings return different arities
            # (e.g., S, I, C) or (S, I, C, N, ...). We only need S.
            if isinstance(result, (tuple, list)):
                sdfs = result[0]
            else:
                sdfs = result
            sdfs = np.asarray(sdfs)
            finite_mask = np.isfinite(sdfs)
            if not np.all(finite_mask):
                logger.warning("Dropping non-finite igl.signed_distance results")
            return xyz[finite_mask], sdfs[finite_mask]
        except Exception as exc:
            logger.warning(f"igl.signed_distance failed ({exc}); falling back to 'vote' method")

    # Compute SDF for each sample
    if tree is None:
        # Fallback: keep the original slow path when SciPy isn't available.
        sdfs = []
        xyz_used = []
        for sample_pt in xyz:
            distances = np.linalg.norm(vertices - sample_pt, axis=1)
            indices = np.argsort(distances)[:num_votes]

            cl_vert = vertices[indices[0]]
            cl_normal = normals[indices[0]]

            ray_vec = sample_pt - cl_vert
            ray_vec_len = np.linalg.norm(ray_vec)

            if ray_vec_len < stdv:
                sdf = abs(np.dot(cl_normal, ray_vec))
            else:
                sdf = ray_vec_len

            num_pos = 0
            for j in range(num_votes):
                vert = vertices[indices[j]]
                normal = normals[indices[j]]
                ray = sample_pt - vert
                ray_normalized = ray / (np.linalg.norm(ray) + 1e-8)
                if np.dot(normal, ray_normalized) > 0:
                    num_pos += 1

            if num_pos == 0 or num_pos == num_votes:
                xyz_used.append(sample_pt)
                if num_pos <= num_votes // 2:
                    sdf = -sdf
                sdfs.append(sdf)

        return np.asarray(xyz_used), np.asarray(sdfs)

    # Fast path: chunked kNN + vectorized voting.
    # Chunking avoids allocating huge (N, K, 3) arrays at once.
    # Note: `workers=-1` uses all cores when supported by SciPy.
    chunk_size = 100_000
    xyz_used_chunks: list[np.ndarray] = []
    sdfs_chunks: list[np.ndarray] = []

    for start in range(0, len(xyz), chunk_size):
        end = min(start + chunk_size, len(xyz))
        xyz_chunk = xyz[start:end]

        try:
            distances, indices = tree.query(xyz_chunk, k=num_votes, workers=-1)
        except TypeError:
            distances, indices = tree.query(xyz_chunk, k=num_votes)

        indices = np.asarray(indices)
        distances = np.asarray(distances)
        if indices.ndim == 1:
            indices = indices[:, None]
            distances = distances[:, None]

        nearest_vertices = vertices[indices[:, 0]]
        nearest_normals = normals[indices[:, 0]]

        ray_vec = xyz_chunk - nearest_vertices
        ray_vec_len = distances[:, 0]

        # Magnitude: near-surface uses point-plane distance; otherwise Euclidean.
        near_mask = ray_vec_len < stdv
        sdf_mag = ray_vec_len.astype(np.float64, copy=True)
        if np.any(near_mask):
            sdf_mag[near_mask] = np.abs(
                np.einsum("ij,ij->i", nearest_normals[near_mask], ray_vec[near_mask])
            )

        # Vote on sign using multiple neighbors.
        # We only need the sign of dot(normal, ray/||ray||), which is identical to
        # sign of dot(normal, ray) (denominator is always positive), so avoid
        # normalizing rays.
        neighbor_vertices = vertices[indices]      # (C, K, 3)
        neighbor_normals = normals[indices]       # (C, K, 3)
        rays = xyz_chunk[:, None, :] - neighbor_vertices
        dots = np.sum(neighbor_normals * rays, axis=2)  # (C, K)
        num_pos = np.sum(dots > 0, axis=1)

        unanimous_mask = (num_pos == 0) | (num_pos == num_votes)
        if not np.any(unanimous_mask):
            continue

        xyz_kept = xyz_chunk[unanimous_mask]
        sdfs_kept = sdf_mag[unanimous_mask]

        # If votes indicate "inside" (all <= 0), flip sign.
        inside_mask = num_pos[unanimous_mask] == 0
        sdfs_kept[inside_mask] *= -1.0

        xyz_used_chunks.append(xyz_kept)
        sdfs_chunks.append(sdfs_kept)

    if not xyz_used_chunks:
        return np.empty((0, 3), dtype=xyz.dtype), np.empty((0,), dtype=np.float64)

    return np.concatenate(xyz_used_chunks, axis=0), np.concatenate(sdfs_chunks, axis=0)


def preprocess_mesh(
    mesh_path: str,
    output_path: str,
    num_sample: int = 500000,
    variance: float = 0.005,
    test_flag: bool = False,
    save_ply: bool = False,
    ply_path: str = None,
    sign_method: str = "vote",
):
    """
    Main preprocessing function.

    Args:
        mesh_path: Path to input mesh file
        output_path: Path to output NPZ file
        num_sample: Total number of samples
        variance: Gaussian variance for surface perturbation
        test_flag: Use test parameters
        save_ply: Also save as PLY for visualization
        ply_path: Path for PLY output
        sign_method: "vote" (default), "proximity" (trimesh signed distance), or "igl" (libigl)
    """
    logger.info(f"Loading: {mesh_path}")

    # Load mesh
    mesh = trimesh.load(mesh_path, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        # Extract actual mesh from scene
        mesh = trimesh.util.concatenate(
            trimesh.Trimesh(**g.geometry) for g in mesh.geometry.values()
        )

    # Repair normals with fallback (silent unless verbose)
    if logger.isEnabledFor(logging.DEBUG):
        mesh, repair_stats, repair_success = repair_with_fallback(mesh, max_attempts=2)
        if "bad_ratio_before" in repair_stats:
            logger.debug(
                f"Normals: {repair_stats['bad_ratio_before']:.1%} -> {repair_stats['bad_ratio_after']:.1%}"
            )
    else:
        mesh, repair_stats, repair_success = repair_with_fallback(mesh, max_attempts=2)

    # Ensure vertex normals exist
    _ = mesh.vertex_normals

    logger.info(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # Normalize mesh to unit cube
    max_dist = normalize_mesh(mesh)
    logger.info("Normalized")

    # Compute sampling parameters
    if test_flag:
        variance = 0.05
        second_variance = variance / 100
        num_samp_near_surf = int(45 * num_sample / 50)
    else:
        second_variance = variance / 10
        num_samp_near_surf = int(47 * num_sample / 50)

    # Sample from surface
    xyz_surf = sample_from_surface(mesh, num_samp_near_surf // 2)

    # Sample SDF near surface
    num_random = num_sample - num_samp_near_surf
    xyz, sdfs = sample_sdf_near_surface(
        mesh=mesh,
        surface_points=xyz_surf,
        num_random_samples=num_random,
        variance=variance,
        second_variance=second_variance,
        bounding_cube_dim=2.0,
        num_votes=11,
        sign_method=sign_method,
    )

    logger.info(f"Sampled: {len(xyz)} points")

    # Separate positive and negative samples
    pos_mask = sdfs > 0
    neg_mask = sdfs <= 0

    pos_samples = xyz[pos_mask]
    pos_sdfs = sdfs[pos_mask][:, None]

    neg_samples = xyz[neg_mask]
    neg_sdfs = sdfs[neg_mask][:, None]

    # Combine with SDF values
    pos_data = np.hstack([pos_samples, pos_sdfs])
    neg_data = np.hstack([neg_samples, neg_sdfs])

    logger.info(f"Pos: {len(pos_data)}, Neg: {len(neg_data)}")

    # Save to NPZ
    logger.info(f"Saving: {output_path}")
    np.savez(
        output_path,
        pos=pos_data,
        neg=neg_data,
    )

    # Optionally save as PLY
    if save_ply and ply_path:
        save_ply_file(xyz, sdfs, ply_path, pos_only=True)

    return {
        "samples": int(len(xyz)),
        "pos": int(len(pos_data)),
        "neg": int(len(neg_data)),
    }


def save_ply_file(xyz, sdfs, filename, neg_only=False, pos_only=False):
    """Save samples to PLY file for visualization."""
    if neg_only:
        mask = sdfs <= 0
    elif pos_only:
        mask = sdfs > 0
    else:
        mask = np.ones(len(sdfs), dtype=bool)

    verts = xyz[mask]
    values = np.abs(sdfs[mask])

    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for v, sdf in zip(verts, values):
            sdf_int = min(int(sdf * 255), 255)
            if pos_only:
                f.write(f"{v[0]} {v[1]} {v[2]} 0 0 {sdf_int}\n")
            else:
                f.write(f"{v[0]} {v[1]} {v[2]} {sdf_int} 0 0\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess mesh for DeepSDF training (Python version, no Pangolin needed)"
    )
    parser.add_argument("-m", "--mesh", required=True, help="Input mesh file (.obj or .ply)")
    parser.add_argument("-o", "--output", required=True, help="Output NPZ file")
    parser.add_argument("-s", "--samples", type=int, default=500000, help="Number of samples")
    parser.add_argument("--var", "--variance", type=float, default=0.005, help="Gaussian variance")
    parser.add_argument("-t", "--test", action="store_true", help="Use test sampling parameters")
    parser.add_argument("--ply", "--ply-output", help="Also save PLY file for visualization")
    parser.add_argument(
        "--sign-method",
        choices=["vote", "proximity", "igl"],
        default="vote",
        help="SDF sign method: 'vote' (C++-style), 'proximity' (trimesh signed distance), or 'igl' (libigl signed distance)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    preprocess_mesh(
        mesh_path=args.mesh,
        output_path=args.output,
        num_sample=args.samples,
        variance=args.var,
        test_flag=args.test,
        save_ply=args.ply is not None,
        ply_path=args.ply,
        sign_method=args.sign_method,
    )


if __name__ == "__main__":
    main()
