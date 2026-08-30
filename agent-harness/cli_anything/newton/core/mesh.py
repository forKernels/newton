# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mesh preprocessing utilities for Newton CLI.

Provides mesh inspection, stitching (merging seam vertices),
and connectivity checking — key operations for cloth simulation workflows.
"""

from __future__ import annotations

import os
from pathlib import Path


def inspect_mesh(mesh_path: str) -> dict:
    """Load and inspect a mesh file, returning statistics.

    Args:
        mesh_path: Path to mesh file (.obj, .stl, .glb, .ply, etc.)

    Returns:
        Dict with vertex count, face count, bounding box, etc.
    """
    path = Path(mesh_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    try:
        import trimesh
    except ImportError:
        raise RuntimeError("trimesh is required for mesh operations: pip install trimesh")

    mesh = trimesh.load(str(path), force="mesh")

    info = {
        "path": str(path),
        "format": path.suffix,
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "is_watertight": mesh.is_watertight,
        "is_winding_consistent": mesh.is_winding_consistent,
        "euler_number": mesh.euler_number,
        "bbox_min": mesh.bounds[0].tolist(),
        "bbox_max": mesh.bounds[1].tolist(),
        "extents": mesh.extents.tolist(),
        "center_mass": mesh.center_mass.tolist(),
        "area": float(mesh.area),
    }

    try:
        info["volume"] = float(mesh.volume)
    except Exception:
        info["volume"] = None

    # Check for disconnected components
    components = mesh.split(only_watertight=False)
    info["num_components"] = len(components)
    if len(components) > 1:
        info["component_vertex_counts"] = [len(c.vertices) for c in components]

    return info


def stitch_mesh(
    mesh_path: str,
    output_path: str | None = None,
    threshold: float = 0.0005,
) -> dict:
    """Stitch (merge) near-duplicate vertices at garment seams.

    This is essential for cloth simulation — garment meshes often come as
    disconnected panels. Stitching merges vertices within `threshold` distance
    to create a single connected cloth mesh.

    Args:
        mesh_path: Path to input mesh file.
        output_path: Path to write stitched mesh. Defaults to <name>_stitched.<ext>.
        threshold: Distance threshold for merging vertices (meters).

    Returns:
        Dict with before/after vertex counts and output path.
    """
    path = Path(mesh_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    try:
        import numpy as np
        import trimesh
    except ImportError:
        raise RuntimeError("trimesh and numpy are required: pip install trimesh numpy")

    mesh = trimesh.load(str(path), force="mesh")
    original_verts = len(mesh.vertices)
    original_faces = len(mesh.faces)
    original_components = len(mesh.split(only_watertight=False))

    # Merge vertices within threshold
    mesh.merge_vertices(merge_tex=True, merge_norm=True)

    # Additional merging for close vertices
    if threshold > 0:
        from scipy.spatial import cKDTree

        tree = cKDTree(mesh.vertices)
        pairs = tree.query_pairs(threshold)

        if pairs:
            # Build merge mapping
            mapping = list(range(len(mesh.vertices)))
            for i, j in pairs:
                # Map higher index to lower
                lo, hi = min(i, j), max(i, j)
                mapping[hi] = lo

            # Resolve transitive mappings
            for i in range(len(mapping)):
                while mapping[i] != mapping[mapping[i]]:
                    mapping[i] = mapping[mapping[i]]

            # Remap faces
            new_faces = np.array([[mapping[v] for v in f] for f in mesh.faces])

            # Remove degenerate faces
            valid = (
                (new_faces[:, 0] != new_faces[:, 1])
                & (new_faces[:, 1] != new_faces[:, 2])
                & (new_faces[:, 0] != new_faces[:, 2])
            )
            new_faces = new_faces[valid]

            # Compact vertex indices
            unique_verts = np.unique(new_faces)
            vert_remap = np.full(len(mesh.vertices), -1, dtype=int)
            vert_remap[unique_verts] = np.arange(len(unique_verts))
            new_faces = vert_remap[new_faces]
            new_verts = mesh.vertices[unique_verts]

            mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces)

    stitched_verts = len(mesh.vertices)
    stitched_faces = len(mesh.faces)
    stitched_components = len(mesh.split(only_watertight=False))

    # Output path
    if output_path is None:
        output_path = str(path.parent / f"{path.stem}_stitched{path.suffix}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mesh.export(output_path)

    return {
        "input": str(path),
        "output": output_path,
        "threshold": threshold,
        "original_vertices": original_verts,
        "original_faces": original_faces,
        "original_components": original_components,
        "stitched_vertices": stitched_verts,
        "stitched_faces": stitched_faces,
        "stitched_components": stitched_components,
        "vertices_merged": original_verts - stitched_verts,
        "file_size": os.path.getsize(output_path),
    }


def check_connectivity(mesh_path: str) -> dict:
    """Check mesh connectivity and report issues.

    Useful for validating cloth meshes before simulation.

    Args:
        mesh_path: Path to mesh file.

    Returns:
        Dict with connectivity analysis.
    """
    path = Path(mesh_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    try:
        import numpy as np
        import trimesh
    except ImportError:
        raise RuntimeError("trimesh and numpy are required: pip install trimesh numpy")

    mesh = trimesh.load(str(path), force="mesh")
    components = mesh.split(only_watertight=False)

    result = {
        "path": str(path),
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "num_components": len(components),
        "is_single_component": len(components) == 1,
        "is_watertight": mesh.is_watertight,
    }

    if len(components) > 1:
        result["components"] = []
        for i, comp in enumerate(components):
            result["components"].append(
                {
                    "index": i,
                    "vertex_count": len(comp.vertices),
                    "face_count": len(comp.faces),
                    "bbox_min": comp.bounds[0].tolist(),
                    "bbox_max": comp.bounds[1].tolist(),
                }
            )

    # Check for boundary edges (non-manifold)
    edges = mesh.edges_unique
    edge_face_count = np.zeros(len(edges), dtype=int)
    for face in mesh.faces:
        for i in range(3):
            e = tuple(sorted([face[i], face[(i + 1) % 3]]))
            # This is approximate — full boundary detection needs face-edge mapping
    result["has_boundary"] = not mesh.is_watertight

    return result
