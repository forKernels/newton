"""Check truncation_ts values for stuck vs moving particles."""
import sys
sys.path.insert(0, "scripts/disco")

import warp as wp
import newton
import numpy as np
from newton_sim_utils import *
from test_cloth_self_collision import discover_original_garments, ORIGINAL_GARMENT_DIR

wp.init()

with wp.ScopedDevice(wp.get_preferred_device()):
    config = load_config()
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mu = 0.6; cfg.has_particle_collision = True; cfg.has_shape_collision = True
    builder.add_shape_cone(body=-1, xform=wp.transform(p=wp.vec3(0,0,0.3), q=wp.quat_identity()),
                           radius=0.12, half_height=0.3, cfg=cfg)

    g = discover_original_garments(ORIGINAL_GARMENT_DIR, "dress")[0]
    print(f"Garment: {g['name']}")
    add_garment_as_cloth(builder, g, position=(0, 0, 1.1))
    builder.color()
    model = finalize_cloth_model(builder, solver_type="vbd")

    # Mass workarounds
    mass_np = model.particle_mass.numpy()
    nz = mass_np[mass_np > 0]; mm = float(np.median(nz) * 0.1)
    mass_np[mass_np < mm] = mm; model.particle_mass.assign(mass_np)
    inv_m = np.divide(1.0, mass_np, out=np.zeros_like(mass_np), where=mass_np != 0)
    model.particle_inv_mass.assign(inv_m)
    fl = model.particle_flags.numpy(); fl[:] |= 1; model.particle_flags.assign(fl)

    pos0 = model.particle_q.numpy().copy()

    solver = build_solver(model, solver_type="vbd", iterations=10, self_contact=True)

    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    cp = newton.CollisionPipeline(model, soft_contact_margin=0.02)
    ct = cp.contacts()
    dt = 1.0/60.0; sdt = dt/10

    # Run 10 frames
    for f in range(10):
        for _ in range(10):
            s0.clear_forces()
            cp.collide(s0, ct)
            solver.step(s0, s1, ctrl, ct, sdt)
            s0, s1 = s1, s0

    # Read truncation_ts from solver
    trunc = solver.truncation_ts.numpy()
    disps = solver.particle_displacements.numpy()

    # Identify stuck particles
    pos_now = s0.particle_q.numpy()
    displacement = np.linalg.norm(pos_now - pos0, axis=1)
    stuck = displacement < 0.001
    moving = ~stuck

    print(f"\nStuck: {np.sum(stuck)}, Moving: {np.sum(moving)}")

    print(f"\n--- truncation_ts ---")
    print(f"  Stuck:  min={trunc[stuck].min():.6f}, max={trunc[stuck].max():.6f}, mean={trunc[stuck].mean():.6f}")
    print(f"  Moving: min={trunc[moving].min():.6f}, max={trunc[moving].max():.6f}, mean={trunc[moving].mean():.6f}")
    print(f"  Stuck with t<0.5: {np.sum(trunc[stuck] < 0.5)}")
    print(f"  Stuck with t<0.1: {np.sum(trunc[stuck] < 0.1)}")
    print(f"  Stuck with t<0.01: {np.sum(trunc[stuck] < 0.01)}")
    print(f"  Moving with t<0.5: {np.sum(trunc[moving] < 0.5)}")

    print(f"\n--- particle_displacements (pre-truncation) ---")
    disp_mag = np.linalg.norm(disps, axis=1)
    print(f"  Stuck:  min={disp_mag[stuck].min():.8f}, max={disp_mag[stuck].max():.8f}, mean={disp_mag[stuck].mean():.8f}")
    print(f"  Moving: min={disp_mag[moving].min():.8f}, max={disp_mag[moving].max():.8f}, mean={disp_mag[moving].mean():.8f}")

    # Check conservative bounds
    if hasattr(solver, 'particle_conservative_bounds'):
        cb = solver.particle_conservative_bounds.numpy()
        print(f"\n--- conservative_bounds ---")
        print(f"  Stuck:  min={cb[stuck].min():.8f}, max={cb[stuck].max():.8f}, mean={cb[stuck].mean():.8f}")
        print(f"  Moving: min={cb[moving].min():.8f}, max={cb[moving].max():.8f}, mean={cb[moving].mean():.8f}")

    print("\nDone.")
