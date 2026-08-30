"""Check if garment mesh has disconnected panels (unstitched seams)."""

import sys

sys.path.insert(0, "scripts/disco")

from collections import deque

import numpy as np
from newton_sim_utils import load_mesh_from_file


def find_connected_components(verts, faces):
    """Find connected components via BFS on triangle adjacency."""
    n_verts = len(verts)
    adj = [[] for _ in range(n_verts)]

    for f in faces:
        for i in range(3):
            for j in range(i + 1, 3):
                adj[f[i]].append(f[j])
                adj[f[j]].append(f[i])

    visited = [False] * n_verts
    components = []

    for start in range(n_verts):
        if visited[start]:
            continue
        # BFS
        comp = []
        queue = deque([start])
        visited[start] = True
        while queue:
            v = queue.popleft()
            comp.append(v)
            for nb in adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(comp)

    return components


path = r"C:\_SimObj\ClothDataset\dress_sleeveless_2550\dress_sleeveless_000YCTJ9HS\dress_sleeveless_000YCTJ9HS_sim.obj"
verts, faces = load_mesh_from_file(path)

# Convert cm to m
verts_m = verts * 0.01

print(f"Mesh: {path.rsplit(chr(92), maxsplit=1)[-1]}")
print(f"Vertices: {len(verts)}")
print(f"Triangles: {len(faces)}")
print(
    f"Bounding box: x=[{verts_m[:, 0].min():.3f}, {verts_m[:, 0].max():.3f}] "
    f"y=[{verts_m[:, 1].min():.3f}, {verts_m[:, 1].max():.3f}] "
    f"z=[{verts_m[:, 2].min():.3f}, {verts_m[:, 2].max():.3f}]"
)

components = find_connected_components(verts, faces)
print(f"\nConnected components: {len(components)}")
for i, comp in enumerate(sorted(components, key=len, reverse=True)):
    comp_verts = verts_m[comp]
    bbox = comp_verts.max(axis=0) - comp_verts.min(axis=0)
    print(f"  Component {i}: {len(comp)} vertices, bbox=[{bbox[0]:.3f} x {bbox[1]:.3f} x {bbox[2]:.3f}]m")

# Check for near-duplicate vertices at seam boundaries
if len(components) > 1:
    print("\n--- Seam analysis ---")
    # For each pair of components, find closest vertex pairs
    for i in range(min(len(components), 5)):
        for j in range(i + 1, min(len(components), 5)):
            vi = verts_m[components[i]]
            vj = verts_m[components[j]]
            # Brute force closest pair
            min_dist = float("inf")
            for v in vi[:: max(1, len(vi) // 100)]:  # sample for speed
                dists = np.linalg.norm(vj - v, axis=1)
                d = dists.min()
                if d < min_dist:
                    min_dist = d
            print(
                f"  Components {i}-{j}: closest distance = {min_dist:.6f}m "
                f"({'STITCHABLE' if min_dist < 0.001 else 'SEPARATE'})"
            )

    # Count vertex pairs that are within merge threshold
    all_verts = verts_m
    merge_threshold = 0.0005  # 0.5mm
    near_duplicates = 0
    # Check boundary vertices of each component
    from collections import Counter

    edge_count = Counter()
    for f in faces:
        for k in range(3):
            e = (min(f[k], f[(k + 1) % 3]), max(f[k], f[(k + 1) % 3]))
            edge_count[e] += 1
    boundary_verts = set()
    for (v0, v1), count in edge_count.items():
        if count == 1:
            boundary_verts.add(v0)
            boundary_verts.add(v1)

    print(f"\n  Total boundary vertices: {len(boundary_verts)}")
    boundary_list = list(boundary_verts)
    boundary_pos = verts_m[boundary_list]

    # Find pairs within merge threshold
    merge_pairs = 0
    for i in range(len(boundary_list)):
        for j in range(i + 1, len(boundary_list)):
            if np.linalg.norm(boundary_pos[i] - boundary_pos[j]) < merge_threshold:
                merge_pairs += 1
    print(f"  Boundary vertex pairs within {merge_threshold * 1000:.1f}mm: {merge_pairs}")
    print("  These could be stitched by merging vertices")
