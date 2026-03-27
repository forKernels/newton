"""
Test pinning fix strategies:
  A) Zero elastic stiffness (tri_ke=0, tri_ka=0) — diagnostic
  B) Initial downward velocity kick
  C) Both combined
"""
import sys
sys.path.insert(0, "scripts/disco")

import argparse
import warp as wp
import newton
import numpy as np
from newton_sim_utils import *
from test_cloth_self_collision import discover_original_garments, ORIGINAL_GARMENT_DIR


def run_test(strategy, viewer_type="gl"):
    config = load_config()
    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    # Cone
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mu = 0.6
    cfg.has_particle_collision = True
    cfg.has_shape_collision = True
    builder.add_shape_cone(
        body=-1,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.3), q=wp.quat_identity()),
        radius=0.12, half_height=0.3, cfg=cfg,
    )

    garments = discover_original_garments(ORIGINAL_GARMENT_DIR, "dress")
    g = garments[0]
    print(f"Garment: {g['name']}")

    cloth_z = 1.1

    if strategy == "zero_ke":
        # Strategy A: Zero elastic stiffness
        print("  STRATEGY: Zero elastic stiffness (tri_ke=0, tri_ka=0)")
        add_garment_as_cloth(builder, g, position=(0, 0, cloth_z),
                             tri_ke=0.0, tri_ka=0.0, tri_kd=0.0,
                             edge_ke=0.0, edge_kd=0.0)
    elif strategy == "low_ke":
        # Very low stiffness
        print("  STRATEGY: Very low elastic stiffness (tri_ke=1, tri_ka=1)")
        add_garment_as_cloth(builder, g, position=(0, 0, cloth_z),
                             tri_ke=1.0, tri_ka=1.0, tri_kd=1e-7,
                             edge_ke=1e-6, edge_kd=1e-5)
    elif strategy == "velocity_kick":
        # Strategy B: Normal stiffness + velocity kick
        print("  STRATEGY: Normal stiffness + downward velocity kick")
        add_garment_as_cloth(builder, g, position=(0, 0, cloth_z))
    elif strategy == "both":
        # Strategy C: Low stiffness + velocity kick
        print("  STRATEGY: Low stiffness + velocity kick")
        add_garment_as_cloth(builder, g, position=(0, 0, cloth_z),
                             tri_ke=10.0, tri_ka=10.0, tri_kd=1e-7,
                             edge_ke=1e-5, edge_kd=1e-4)
    elif strategy == "ramp":
        # Strategy D: Start with zero stiffness, ramp up over time
        print("  STRATEGY: Ramp stiffness from 0 to normal over 50 frames")
        add_garment_as_cloth(builder, g, position=(0, 0, cloth_z),
                             tri_ke=0.0, tri_ka=0.0, tri_kd=0.0,
                             edge_ke=0.0, edge_kd=0.0)
    else:
        print("  STRATEGY: Normal (baseline)")
        add_garment_as_cloth(builder, g, position=(0, 0, cloth_z))

    builder.color()
    model = finalize_cloth_model(builder, solver_type="vbd")

    # Apply mass/flags workarounds
    mass_np = model.particle_mass.numpy()
    nonzero = mass_np[mass_np > 0]
    min_mass = float(np.median(nonzero) * 0.1)
    pinned_mask = mass_np < min_mass
    if np.any(pinned_mask):
        mass_np[pinned_mask] = min_mass
        model.particle_mass.assign(mass_np)
        inv_mass_np = np.divide(1.0, mass_np, out=np.zeros_like(mass_np), where=mass_np != 0.0)
        model.particle_inv_mass.assign(inv_mass_np)
    flags_np = model.particle_flags.numpy()
    flags_np[:] |= 1
    model.particle_flags.assign(flags_np)

    pos_initial = model.particle_q.numpy().copy()

    solver = build_solver(model, solver_type="vbd", iterations=10, self_contact=True)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    # Strategy B/C: Apply velocity kick
    if strategy in ("velocity_kick", "both"):
        vel_np = state_0.particle_qd.numpy()
        vel_np[:, 2] = -2.0  # 2 m/s downward
        state_0.particle_qd.assign(vel_np)
        print("  Applied -2.0 m/s downward velocity to all particles")

    collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.02)
    contacts = collision_pipeline.contacts()
    dt = 1.0 / 60.0
    sub_dt = dt / 10

    viewer = create_viewer(viewer_type=viewer_type)
    viewer.set_model(model)
    viewer.begin_frame(0.0)
    viewer.log_state(state_0)
    viewer.end_frame()

    total_frames = 300
    target_tri_ke = 100.0
    target_tri_ka = 100.0
    target_edge_ke = 1e-4

    for frame in range(total_frames):
        # Strategy D: Ramp stiffness
        if strategy == "ramp" and frame < 50:
            t = (frame + 1) / 50.0  # 0→1 over 50 frames
            tri_mat = model.tri_materials.numpy()
            tri_mat[:, 0] = target_tri_ke * t  # tri_ke
            tri_mat[:, 1] = target_tri_ka * t  # tri_ka
            model.tri_materials.assign(tri_mat)

            edge_bp = model.edge_bending_properties.numpy()
            edge_bp[:, 0] = target_edge_ke * t
            model.edge_bending_properties.assign(edge_bp)

        if strategy == "ramp" and frame == 50:
            print(f"  Frame 50: Stiffness ramp complete (tri_ke={target_tri_ke})")

        for _ in range(10):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

        viewer.begin_frame((frame + 1) * dt)
        viewer.log_state(state_0)
        viewer.end_frame()

        if (frame + 1) % 50 == 0:
            pq = state_0.particle_q.numpy()
            disp = np.linalg.norm(pq - pos_initial, axis=1)
            stuck = np.sum(disp < 0.001)
            z_range = pq[:, 2].max() - pq[:, 2].min()
            print(f"  Frame {frame + 1}/{total_frames}  stuck={stuck}  z_range={z_range:.3f}m")

    viewer.close()

    pos_final = state_0.particle_q.numpy()
    disp = np.linalg.norm(pos_final - pos_initial, axis=1)
    stuck = np.sum(disp < 0.001)
    print(f"\n  RESULT [{strategy}]: {stuck} stuck particles, z_range={pos_final[:, 2].max() - pos_final[:, 2].min():.3f}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="zero_ke",
                        choices=["normal", "zero_ke", "low_ke", "velocity_kick", "both", "ramp"])
    parser.add_argument("--viewer", type=str, default="gl", choices=["gl", "null"])
    args = parser.parse_args()

    wp.init()
    with wp.ScopedDevice(wp.get_preferred_device()):
        run_test(args.strategy, args.viewer)
