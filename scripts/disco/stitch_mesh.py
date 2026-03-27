"""
Stitch garment mesh panels by merging near-duplicate vertices at seams.

Maria garment _sim.obj meshes are exported with separate panels (front, back,
sleeves) that share boundary vertices at the seams but aren't topologically
connected. This preprocessor merges vertices within a threshold distance to
create a single connected mesh suitable for Newton cloth simulation.
"""

import numpy as np
from scipy.spatial import cKDTree


def stitch_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    merge_threshold: float = 0.0005,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge near-duplicate vertices to stitch disconnected panels.

    Args:
        verts: (N, 3) vertex positions
        faces: (M, 3) triangle indices
        merge_threshold: Distance threshold for merging vertices [meters]

    Returns:
        (new_verts, new_faces) with merged vertices and updated indices
    """
    n_verts = len(verts)

    # Build KD-tree for fast nearest-neighbor lookup
    tree = cKDTree(verts)

    # Find all pairs within threshold
    pairs = tree.query_pairs(merge_threshold)

    if not pairs:
        print(f"  stitch_mesh: No vertices within {merge_threshold*1000:.1f}mm to merge")
        return verts, faces

    # Build union-find for merging
    parent = list(range(n_verts))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep lower index as root (preserves vertex ordering)
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for i, j in pairs:
        union(i, j)

    # Build mapping: old vertex index → new vertex index
    roots = {}
    new_idx = 0
    old_to_new = np.zeros(n_verts, dtype=np.int32)

    for v in range(n_verts):
        root = find(v)
        if root not in roots:
            roots[root] = new_idx
            new_idx += 1
        old_to_new[v] = roots[root]

    n_merged = n_verts - new_idx
    print(f"  stitch_mesh: Merged {n_merged} vertices ({n_verts} → {new_idx})")

    # Build new vertex array (average positions of merged vertices)
    new_verts = np.zeros((new_idx, 3), dtype=np.float32)
    counts = np.zeros(new_idx, dtype=np.int32)

    for v in range(n_verts):
        nv = old_to_new[v]
        new_verts[nv] += verts[v]
        counts[nv] += 1

    new_verts /= counts[:, np.newaxis]

    # Remap face indices
    new_faces = old_to_new[faces]

    # Remove degenerate triangles (where two or more vertices collapsed to same index)
    valid = (new_faces[:, 0] != new_faces[:, 1]) & \
            (new_faces[:, 1] != new_faces[:, 2]) & \
            (new_faces[:, 0] != new_faces[:, 2])

    n_degenerate = np.sum(~valid)
    if n_degenerate > 0:
        print(f"  stitch_mesh: Removed {n_degenerate} degenerate triangles")
        new_faces = new_faces[valid]

    # Verify connectivity
    from collections import deque
    adj = [[] for _ in range(new_idx)]
    for f in new_faces:
        for i in range(3):
            for j in range(i + 1, 3):
                adj[f[i]].append(f[j])
                adj[f[j]].append(f[i])

    visited = [False] * new_idx
    n_components = 0
    for start in range(new_idx):
        if visited[start]:
            continue
        n_components += 1
        queue = deque([start])
        visited[start] = True
        while queue:
            v = queue.popleft()
            for nb in adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

    print(f"  stitch_mesh: {n_components} connected component(s) after stitching")

    return new_verts, new_faces


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts/disco")
    from newton_sim_utils import load_mesh_from_file

    path = r"C:\_SimObj\ClothDataset\dress_sleeveless_2550\dress_sleeveless_000YCTJ9HS\dress_sleeveless_000YCTJ9HS_sim.obj"
    verts, faces = load_mesh_from_file(path)

    # Convert cm to m (same as load_garment_mesh)
    verts *= 0.01
    verts[:, 0] *= -1
    verts[:, 2] *= -1

    print(f"Before: {len(verts)} verts, {len(faces)} tris")
    new_verts, new_faces = stitch_mesh(verts, faces, merge_threshold=0.0005)
    print(f"After:  {len(new_verts)} verts, {len(new_faces)} tris")
