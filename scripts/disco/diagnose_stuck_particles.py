"""Diagnose which particles are stuck and WHY by running a few frames and comparing positions."""

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
    pinned = mass_np < min_mass
    mass_np[pinned] = min_mass
    model.particle_mass.assign(mass_np)
    inv_mass_np = np.divide(1.0, mass_np, out=np.zeros_like(mass_np), where=mass_np != 0.0)
    model.particle_inv_mass.assign(inv_mass_np)
    flags_np = model.particle_flags.numpy()
    flags_np[:] |= 1
    model.particle_flags.assign(flags_np)

    # Record initial positions
    pos_initial = model.particle_q.numpy().copy()

    solver = build_solver(model, solver_type="vbd", iterations=10, self_contact=True)

    # Run 50 frames
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.02)
    contacts = collision_pipeline.contacts()
    dt = 1.0 / 60.0
    sub_dt = dt / 10

    for frame in range(50):
        for _ in range(10):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

    pos_final = state_0.particle_q.numpy()

    # Analyze which particles moved
    displacement = np.linalg.norm(pos_final - pos_initial, axis=1)
    z_drop = pos_initial[:, 2] - pos_final[:, 2]  # positive = fell down

    threshold = 0.001  # 1mm
    stuck = displacement < threshold
    moving = ~stuck

    print(f"\n{'=' * 60}")
    print("PARTICLE MOVEMENT ANALYSIS (50 frames)")
    print(f"{'=' * 60}")
    print(f"Total particles: {len(displacement)}")
    print(f"Moving (>1mm): {np.sum(moving)} ({100 * np.sum(moving) / len(displacement):.1f}%)")
    print(f"Stuck (<1mm):  {np.sum(stuck)} ({100 * np.sum(stuck) / len(displacement):.1f}%)")
    print("")
    print("Moving particles:")
    print(f"  Mean displacement: {displacement[moving].mean():.4f}m")
    print(f"  Mean z-drop: {z_drop[moving].mean():.4f}m")
    print("")
    print("Stuck particles:")
    if np.any(stuck):
        print(f"  Count: {np.sum(stuck)}")
        print(f"  Mean displacement: {displacement[stuck].mean():.6f}m")
        stuck_masses = mass_np[stuck]
        stuck_inv_masses = inv_mass_np[stuck]
        stuck_flags = flags_np[stuck]
        print(f"  Mass range: [{stuck_masses.min():.2e}, {stuck_masses.max():.2e}]")
        print(f"  Inv mass range: [{stuck_inv_masses.min():.2e}, {stuck_inv_masses.max():.2e}]")
        print(f"  All ACTIVE: {np.all(stuck_flags & 1)}")

        # Check adjacency — how many triangles/edges reference stuck particles
        tri_indices = model.tri_indices.numpy()
        stuck_indices = set(np.where(stuck)[0])

        stuck_in_tris = 0
        for idx in stuck_indices:
            in_tri = np.any(tri_indices == idx)
            if in_tri:
                stuck_in_tris += 1

        print(f"  Stuck verts in at least one triangle: {stuck_in_tris} / {np.sum(stuck)}")
        print(f"  Stuck verts NOT in any triangle: {np.sum(stuck) - stuck_in_tris}")

        # Show spatial distribution
        stuck_pos = pos_initial[stuck]
        print("  Position range:")
        print(f"    x: [{stuck_pos[:, 0].min():.3f}, {stuck_pos[:, 0].max():.3f}]")
        print(f"    y: [{stuck_pos[:, 1].min():.3f}, {stuck_pos[:, 1].max():.3f}]")
        print(f"    z: [{stuck_pos[:, 2].min():.3f}, {stuck_pos[:, 2].max():.3f}]")

        # Compare to color groups
        print("\n  Color group membership:")
        all_colored = set()
        for cg in model.particle_color_groups:
            cg_np = cg.numpy()
            all_colored.update(cg_np.tolist())
        stuck_not_colored = stuck_indices - all_colored
        print(f"    Stuck but NOT in any color group: {len(stuck_not_colored)}")
        print(f"    Total particles in color groups: {len(all_colored)} / {len(displacement)}")

    print("\nDone.")
