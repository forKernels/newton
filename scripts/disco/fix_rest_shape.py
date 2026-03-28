"""
Fix VBD mesh cloth pinning by resetting tri_poses to current deformed shape.

tri_poses stores the inverse 2D reference matrix (inv_D) for each triangle.
When the garment is imported in its 3D rest shape, the StVK elastic energy
tries to maintain that shape, causing boundary vertices to resist gravity.

This fix recomputes tri_poses from the current particle positions after
a warm-up period, so the elastic energy treats the current (partially
draped) configuration as the rest shape.
"""
import sys
sys.path.insert(0, "scripts/disco")

import warp as wp
import newton
import numpy as np
from newton_sim_utils import *
from test_cloth_self_collision import discover_original_garments, ORIGINAL_GARMENT_DIR


@wp.kernel
def recompute_tri_poses(
    particle_q: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses_out: wp.array(dtype=wp.mat22),
    tri_areas_out: wp.array(dtype=float),
):
    """Recompute tri_poses (inv_D) and tri_areas from current particle positions.

    This is the same math as builder.add_triangle() but runs on GPU
    for all triangles at once.
    """
    tid = wp.tid()

    i = tri_indices[tid, 0]
    j = tri_indices[tid, 1]
    k = tri_indices[tid, 2]

    p = particle_q[i]
    q = particle_q[j]
    r = particle_q[k]

    qp = q - p
    rp = r - p

    # Construct 2D basis aligned with triangle
    n = wp.normalize(wp.cross(qp, rp))
    e1 = wp.normalize(qp)
    e2 = wp.normalize(wp.cross(n, e1))

    # Project edges onto 2D basis → D matrix
    d00 = wp.dot(e1, qp)
    d10 = wp.dot(e2, qp)
    d01 = wp.dot(e1, rp)
    d11 = wp.dot(e2, rp)

    D = wp.mat22(d00, d01, d10, d11)
    det = d00 * d11 - d01 * d10
    area = det / 2.0

    if area > 1.0e-12:
        inv_D = wp.inverse(D)
        tri_poses_out[tid] = inv_D
        tri_areas_out[tid] = area
    # else: keep existing values (degenerate triangle)


def reset_rest_shape(model, state):
    """Reset tri_poses and edge_rest_angle to match current deformed shape.

    Call this after a warm-up period to "bake" the current configuration
    as the new rest shape, eliminating elastic forces that fight gravity.
    """
    # Recompute tri_poses from current positions
    wp.launch(
        kernel=recompute_tri_poses,
        dim=model.tri_count,
        inputs=[state.particle_q, model.tri_indices],
        outputs=[model.tri_poses, model.tri_areas],
        device=model.device,
    )

    # Zero edge rest angles (bending should reference flat)
    model.edge_rest_angle.zero_()

    # Recompute edge rest lengths from current positions
    edge_indices = model.edge_indices.numpy()
    pos = state.particle_q.numpy()

    rest_lengths = np.zeros(len(edge_indices), dtype=np.float32)
    for ei in range(len(edge_indices)):
        v0 = edge_indices[ei, 2]
        v1 = edge_indices[ei, 3]
        rest_lengths[ei] = np.linalg.norm(pos[v0] - pos[v1])
    model.edge_rest_length.assign(rest_lengths)


def main():
    wp.init()
    device = wp.get_preferred_device()

    with wp.ScopedDevice(device):
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
        add_garment_as_cloth(builder, g, position=(0, 0, cloth_z))
        builder.color()

        model = finalize_cloth_model(builder, solver_type="vbd")

        # Standard workarounds
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

        # Simulation
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.02)
        contacts = collision_pipeline.contacts()
        dt = 1.0 / 60.0
        sub_dt = dt / 10

        viewer = create_viewer(viewer_type="gl")
        viewer.set_model(model)
        viewer.begin_frame(0.0)
        viewer.log_state(state_0)
        viewer.end_frame()

        warmup_frames = 5
        reset_interval = 50
        total_frames = 300

        print(f"  Strategy: warmup={warmup_frames} frames, then reset rest shape every {reset_interval} frames")
        print(f"  Total: {total_frames} frames")
        print(f"  Particles: {model.particle_count}")
        print(f"  Device: {device}")

        for frame in range(total_frames):
            for _ in range(10):
                state_0.clear_forces()
                collision_pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sub_dt)
                state_0, state_1 = state_1, state_0

            viewer.begin_frame((frame + 1) * dt)
            viewer.log_state(state_0)
            viewer.end_frame()

            # Reset rest shape after warmup and periodically
            if (frame + 1) == warmup_frames or ((frame + 1) > warmup_frames and (frame + 1) % reset_interval == 0):
                print(f"  Frame {frame + 1}: RESETTING REST SHAPE")
                reset_rest_shape(model, state_0)

            if (frame + 1) % 50 == 0:
                pq = state_0.particle_q.numpy()
                disp = np.linalg.norm(pq - pos_initial, axis=1)
                stuck = np.sum(disp < 0.001)
                z_range = pq[:, 2].max() - pq[:, 2].min()
                print(f"  Frame {frame + 1}/{total_frames}  stuck={stuck}  z_range={z_range:.3f}m")

        viewer.close()

        # Final analysis
        pos_final = state_0.particle_q.numpy()
        disp = np.linalg.norm(pos_final - pos_initial, axis=1)
        stuck = np.sum(disp < 0.001)
        print(f"\n  Final: {stuck} stuck particles (was 891 before fix)")
        print(f"  z_range: {pos_final[:, 2].max() - pos_final[:, 2].min():.3f}m")
        print("  Done.")


if __name__ == "__main__":
    main()
