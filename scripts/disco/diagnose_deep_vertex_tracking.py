"""
Deep diagnostic: track per-particle forces, Hessian, and displacement in VBD solver.

Hooks into the solver to extract per-frame data for stuck vs moving particles.
Identifies exactly WHY particles produce zero displacement.
"""

import sys

sys.path.insert(0, "scripts/disco")

import numpy as np
import warp as wp
from newton_sim_utils import *
from test_cloth_self_collision import ORIGINAL_GARMENT_DIR, discover_original_garments

import newton

wp.init()

with wp.ScopedDevice(wp.get_preferred_device()):
    config = load_config()
    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    # Add cone
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mu = 0.6
    cfg.has_particle_collision = True
    cfg.has_shape_collision = True
    builder.add_shape_cone(
        body=-1,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.3), q=wp.quat_identity()),
        radius=0.12,
        half_height=0.3,
        cfg=cfg,
    )

    garments = discover_original_garments(ORIGINAL_GARMENT_DIR, "dress")
    g = garments[0]
    print(f"Garment: {g['name']}")

    cloth_z = 1.1
    add_garment_as_cloth(builder, g, position=(0, 0, cloth_z))
    builder.color()

    model = finalize_cloth_model(builder, solver_type="vbd")

    # Apply all workarounds
    mass_np = model.particle_mass.numpy()
    nonzero = mass_np[mass_np > 0]
    min_mass = float(np.median(nonzero) * 0.1)
    pinned_mask = mass_np < min_mass
    mass_np[pinned_mask] = min_mass
    model.particle_mass.assign(mass_np)
    inv_mass_np = np.divide(1.0, mass_np, out=np.zeros_like(mass_np), where=mass_np != 0.0)
    model.particle_inv_mass.assign(inv_mass_np)
    flags_np = model.particle_flags.numpy()
    flags_np[:] |= 1
    model.particle_flags.assign(flags_np)

    pos_initial = model.particle_q.numpy().copy()
    n_particles = model.particle_count

    solver = build_solver(model, solver_type="vbd", iterations=10, self_contact=True)

    # =========================================================================
    # Run simulation and track positions at every frame
    # =========================================================================
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.02)
    contacts = collision_pipeline.contacts()
    dt = 1.0 / 60.0
    sub_dt = dt / 10

    # Track positions at key frames
    track_frames = [0, 1, 2, 3, 5, 10, 20, 50]
    positions_at_frame = {0: pos_initial.copy()}

    for frame in range(51):
        for _ in range(10):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

        if (frame + 1) in track_frames:
            positions_at_frame[frame + 1] = state_0.particle_q.numpy().copy()

    pos_final = state_0.particle_q.numpy()

    # =========================================================================
    # Identify stuck particles
    # =========================================================================
    displacement = np.linalg.norm(pos_final - pos_initial, axis=1)
    threshold = 0.001
    stuck = displacement < threshold
    moving = ~stuck
    stuck_ids = np.where(stuck)[0]
    moving_ids = np.where(moving)[0]

    print(f"\n{'=' * 70}")
    print(f"DEEP VERTEX TRACKING — {np.sum(stuck)} stuck, {np.sum(moving)} moving")
    print(f"{'=' * 70}")

    # =========================================================================
    # 1. Track per-frame movement for a few stuck and moving particles
    # =========================================================================
    sample_stuck = stuck_ids[:5] if len(stuck_ids) >= 5 else stuck_ids
    sample_moving = moving_ids[:5] if len(moving_ids) >= 5 else moving_ids

    print("\n--- Per-frame Z position (stuck particles) ---")
    for pid in sample_stuck:
        positions = []
        for f in sorted(positions_at_frame.keys()):
            z = positions_at_frame[f][pid, 2]
            positions.append(f"{f}:{z:.6f}")
        print(f"  Particle {pid}: {', '.join(positions)}")

    print("\n--- Per-frame Z position (moving particles) ---")
    for pid in sample_moving:
        positions = []
        for f in sorted(positions_at_frame.keys()):
            z = positions_at_frame[f][pid, 2]
            positions.append(f"{f}:{z:.4f}")
        print(f"  Particle {pid}: {', '.join(positions)}")

    # =========================================================================
    # 2. Check inertia target vs position after forward_step
    # =========================================================================
    # The inertia array is stored on the solver
    inertia_np = solver.inertia.numpy()
    pos_now = state_0.particle_q.numpy()

    print("\n--- Inertia vs Position (stuck particles) ---")
    for pid in sample_stuck:
        inertia_diff = np.linalg.norm(inertia_np[pid] - pos_now[pid])
        print(f"  Particle {pid}: pos={pos_now[pid]}, inertia={inertia_np[pid]}, diff={inertia_diff:.8f}")

    print("\n--- Inertia vs Position (moving particles) ---")
    for pid in sample_moving:
        inertia_diff = np.linalg.norm(inertia_np[pid] - pos_now[pid])
        print(
            f"  Particle {pid}: pos_z={pos_now[pid, 2]:.4f}, inertia_z={inertia_np[pid, 2]:.4f}, diff={inertia_diff:.6f}"
        )

    # =========================================================================
    # 3. Analyze triangle quality for stuck particles
    # =========================================================================
    tri_indices = model.tri_indices.numpy()
    tri_materials = model.tri_materials.numpy() if hasattr(model, "tri_materials") else None

    print("\n--- Triangle quality for stuck particles ---")
    for pid in sample_stuck[:3]:
        # Find triangles containing this particle
        tri_mask = np.any(tri_indices == pid, axis=1)
        adj_tris = np.where(tri_mask)[0]

        areas = []
        for ti in adj_tris:
            v0, v1, v2 = pos_initial[tri_indices[ti]]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            areas.append(area)

            # Check triangle aspect ratio (degenerate detection)
            edges = [np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v1), np.linalg.norm(v0 - v2)]
            aspect = max(edges) / (min(edges) + 1e-20)
            if ti == adj_tris[0]:
                print(f"  Particle {pid}: {len(adj_tris)} adjacent tris")
            print(f"    Tri {ti}: area={area:.2e}, aspect_ratio={aspect:.1f}, edges={[f'{e:.4f}' for e in edges]}")

        if areas:
            print(f"    Area range: [{min(areas):.2e}, {max(areas):.2e}], mean={np.mean(areas):.2e}")

    print("\n--- Triangle quality for moving particles ---")
    for pid in sample_moving[:3]:
        tri_mask = np.any(tri_indices == pid, axis=1)
        adj_tris = np.where(tri_mask)[0]
        areas = []
        for ti in adj_tris:
            v0, v1, v2 = pos_initial[tri_indices[ti]]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            areas.append(area)
        if areas:
            edges_all = []
            for ti in adj_tris:
                v0, v1, v2 = pos_initial[tri_indices[ti]]
                edges_all.append(
                    max(np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v1), np.linalg.norm(v0 - v2))
                    / (min(np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v1), np.linalg.norm(v0 - v2)) + 1e-20)
                )
            print(
                f"  Particle {pid}: {len(adj_tris)} tris, area=[{min(areas):.2e}, {max(areas):.2e}], max_aspect={max(edges_all):.1f}"
            )

    # =========================================================================
    # 4. Statistical comparison of stuck vs moving
    # =========================================================================
    print("\n--- Statistical comparison ---")

    # Compute per-particle average triangle area and max aspect ratio
    avg_areas = np.zeros(n_particles)
    max_aspects = np.zeros(n_particles)
    num_adj_tris = np.zeros(n_particles, dtype=int)

    for ti in range(len(tri_indices)):
        v0, v1, v2 = pos_initial[tri_indices[ti]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        edges = [np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v1), np.linalg.norm(v0 - v2)]
        aspect = max(edges) / (min(edges) + 1e-20)

        for vi in tri_indices[ti]:
            avg_areas[vi] += area
            max_aspects[vi] = max(max_aspects[vi], aspect)
            num_adj_tris[vi] += 1

    avg_areas = np.divide(avg_areas, num_adj_tris, out=np.zeros_like(avg_areas), where=num_adj_tris > 0)

    print(f"  Stuck particles ({np.sum(stuck)}):")
    print(f"    Avg triangle area: mean={avg_areas[stuck].mean():.2e}, median={np.median(avg_areas[stuck]):.2e}")
    print(
        f"    Max aspect ratio:  mean={max_aspects[stuck].mean():.1f}, median={np.median(max_aspects[stuck]):.1f}, max={max_aspects[stuck].max():.1f}"
    )
    print(
        f"    Adj triangle count: mean={num_adj_tris[stuck].mean():.1f}, min={num_adj_tris[stuck].min()}, max={num_adj_tris[stuck].max()}"
    )
    print(f"    Mass: mean={mass_np[stuck].mean():.2e}, min={mass_np[stuck].min():.2e}")

    print(f"  Moving particles ({np.sum(moving)}):")
    print(f"    Avg triangle area: mean={avg_areas[moving].mean():.2e}, median={np.median(avg_areas[moving]):.2e}")
    print(
        f"    Max aspect ratio:  mean={max_aspects[moving].mean():.1f}, median={np.median(max_aspects[moving]):.1f}, max={max_aspects[moving].max():.1f}"
    )
    print(
        f"    Adj triangle count: mean={num_adj_tris[moving].mean():.1f}, min={num_adj_tris[moving].min()}, max={num_adj_tris[moving].max()}"
    )
    print(f"    Mass: mean={mass_np[moving].mean():.2e}, min={mass_np[moving].min():.2e}")

    # =========================================================================
    # 5. Check if stuck particles cluster spatially
    # =========================================================================
    print("\n--- Spatial distribution ---")
    stuck_pos = pos_initial[stuck]
    moving_pos = pos_initial[moving]
    print(
        f"  Stuck centroid:  ({stuck_pos[:, 0].mean():.3f}, {stuck_pos[:, 1].mean():.3f}, {stuck_pos[:, 2].mean():.3f})"
    )
    print(
        f"  Moving centroid: ({moving_pos[:, 0].mean():.3f}, {moving_pos[:, 1].mean():.3f}, {moving_pos[:, 2].mean():.3f})"
    )
    print(f"  Stuck Z range:   [{stuck_pos[:, 2].min():.3f}, {stuck_pos[:, 2].max():.3f}]")
    print(f"  Moving Z range:  [{moving_pos[:, 2].min():.3f}, {moving_pos[:, 2].max():.3f}]")

    # Check if stuck particles are connected to each other (clustering)
    stuck_set = set(stuck_ids.tolist())
    stuck_neighbors_stuck = 0
    stuck_neighbors_moving = 0
    for ti in range(len(tri_indices)):
        verts_in_tri = set(tri_indices[ti].tolist())
        stuck_in_tri = verts_in_tri & stuck_set
        moving_in_tri = verts_in_tri - stuck_set
        if stuck_in_tri:
            stuck_neighbors_stuck += len(stuck_in_tri) * (len(stuck_in_tri) - 1)
            stuck_neighbors_moving += len(stuck_in_tri) * len(moving_in_tri)

    print(f"\n  Stuck-to-stuck connections: {stuck_neighbors_stuck}")
    print(f"  Stuck-to-moving connections: {stuck_neighbors_moving}")
    if stuck_neighbors_stuck + stuck_neighbors_moving > 0:
        ratio = stuck_neighbors_stuck / (stuck_neighbors_stuck + stuck_neighbors_moving)
        print(f"  Clustering ratio: {ratio:.2f} (1.0 = fully clustered, 0.0 = fully dispersed)")
        # Expected ratio if random: stuck_fraction = 891/8922 ≈ 0.10
        print(f"  Expected ratio if random: {np.sum(stuck) / n_particles:.2f}")

    # =========================================================================
    # 6. Check edge bending adjacency for stuck particles
    # =========================================================================
    edge_indices = model.edge_indices.numpy()
    print("\n--- Edge bending adjacency ---")
    print(f"  Total edges: {len(edge_indices)}")

    stuck_in_edges = 0
    for ei in range(len(edge_indices)):
        edge_verts = set(edge_indices[ei].tolist())
        if edge_verts & stuck_set:
            stuck_in_edges += 1
    print(f"  Edges involving stuck particles: {stuck_in_edges}")

    # Check edge rest angles for edges involving stuck particles
    era = model.edge_rest_angle.numpy()
    stuck_edge_mask = np.array([bool(set(edge_indices[ei].tolist()) & stuck_set) for ei in range(len(edge_indices))])
    if np.any(stuck_edge_mask):
        stuck_angles = era[stuck_edge_mask]
        print(f"  Rest angles for stuck-adjacent edges: [{stuck_angles.min():.6f}, {stuck_angles.max():.6f}]")
        print(f"  Confirmed zero: {np.all(stuck_angles == 0)}")

    # =========================================================================
    # 7. Check if stuck particles are on mesh boundary
    # =========================================================================
    # A boundary edge has only one adjacent triangle
    from collections import Counter

    edge_count = Counter()
    for ti in range(len(tri_indices)):
        v0, v1, v2 = tri_indices[ti]
        for e in [(min(v0, v1), max(v0, v1)), (min(v1, v2), max(v1, v2)), (min(v0, v2), max(v0, v2))]:
            edge_count[e] += 1

    boundary_verts = set()
    for (v0, v1), count in edge_count.items():
        if count == 1:  # boundary edge
            boundary_verts.add(v0)
            boundary_verts.add(v1)

    stuck_on_boundary = len(stuck_set & boundary_verts)
    total_boundary = len(boundary_verts)
    print("\n--- Mesh boundary analysis ---")
    print(f"  Total boundary vertices: {total_boundary}")
    print(f"  Stuck particles on boundary: {stuck_on_boundary} / {np.sum(stuck)}")
    print(f"  Moving particles on boundary: {len(boundary_verts - stuck_set)} / {np.sum(moving)}")
    if total_boundary > 0:
        print(f"  Fraction of boundary that's stuck: {stuck_on_boundary / total_boundary:.2f}")

    print("\nDone.")
