"""
Test cloth self-collision: drop garment (or grid) onto a cone.

The cone forces the cloth to drape, fold and bunch, making self-collision
thickness clearly visible.

Usage:
    python scripts/disco/test_cloth_self_collision.py
    python scripts/disco/test_cloth_self_collision.py --garment dress
    python scripts/disco/test_cloth_self_collision.py --use-grid --grid-res 60
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import warp as wp
from newton_sim_utils import (
    CLOTH_PRESETS,
    add_garment_as_cloth,
    add_ground_plane,
    build_solver,
    create_spring_assist,
    create_viewer,
    discover_garments,
    finalize_cloth_model,
    load_config,
)

import newton

ORIGINAL_GARMENT_DIR = r"C:\_SimObj\ClothDataset"


def discover_original_garments(garment_dir: str, category_filter: str | None = None):
    """Find original *_sim.obj garments (no thickness) for Newton."""
    from pathlib import Path

    results = []
    root = Path(garment_dir)
    if not root.exists():
        return results
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        if category_filter and category_filter.lower() not in category.lower():
            continue
        for garment_dir_item in sorted(cat_dir.iterdir()):
            if not garment_dir_item.is_dir():
                continue
            sim_objs = list(garment_dir_item.glob("*_sim.obj"))
            # Exclude *_sim_prep.obj
            sim_objs = [p for p in sim_objs if "_sim_prep" not in p.name]
            if sim_objs:
                results.append(
                    {
                        "name": garment_dir_item.name,
                        "category": category,
                        "path": str(sim_objs[0]),
                        "obj_path": str(sim_objs[0]),
                        "is_clean": False,
                    }
                )
    return results


def add_cone(
    builder: newton.ModelBuilder,
    position=(0.0, 0.0, 0.0),
    radius=0.15,
    half_height=0.3,
):
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mu = 0.6
    cfg.restitution = 0.05
    cfg.has_particle_collision = True
    cfg.has_shape_collision = True
    builder.add_shape_cone(
        body=-1,
        xform=wp.transform(p=wp.vec3(*position), q=wp.quat_identity()),
        radius=radius,
        half_height=half_height,
        cfg=cfg,
    )


def add_sphere(
    builder: newton.ModelBuilder,
    position=(0.0, 0.0, 0.0),
    radius=0.2,
):
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mu = 0.6
    cfg.restitution = 0.05
    cfg.has_particle_collision = True
    cfg.has_shape_collision = True
    builder.add_shape_sphere(
        body=-1,
        xform=wp.transform(p=wp.vec3(*position), q=wp.quat_identity()),
        radius=radius,
        cfg=cfg,
    )


def main():
    parser = argparse.ArgumentParser(description="Test cloth self-collision on a cone")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", type=str, default="cotton", choices=list(CLOTH_PRESETS.keys()))
    parser.add_argument("--use-grid", action="store_true", help="Use procedural grid instead of garment")
    parser.add_argument("--grid-res", type=int, default=60)
    parser.add_argument("--garment", type=str, default=None, help="Garment name substring filter")
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--solver", type=str, default="vbd", choices=["vbd", "xpbd"], help="Solver type")
    parser.add_argument("--solver-iters", type=int, default=10)
    parser.add_argument("--obstacle", type=str, default="cone", choices=["cone", "sphere", "both"])
    parser.add_argument("--cone-radius", type=float, default=0.15)
    parser.add_argument("--cone-height", type=float, default=0.35)
    parser.add_argument("--drop-height", type=float, default=0.8)
    parser.add_argument("--viewer", type=str, default="gl", choices=["gl", "usd", "null"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--spring-assist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mass-spring force injection for stuck particles (default: True)",
    )
    parser.add_argument("--spring-ks", type=float, default=500.0, help="Spring stretch stiffness")
    parser.add_argument("--spring-kb", type=float, default=25.0, help="Spring bend stiffness")
    parser.add_argument("--spring-boost", type=float, default=1.0, help="Spring force boost factor")
    args = parser.parse_args()

    wp.init()
    device = wp.get_device(args.device) if args.device else wp.get_preferred_device()

    with wp.ScopedDevice(device):
        config = load_config(args.config)
        cloth_params = CLOTH_PRESETS.get(args.preset, CLOTH_PRESETS["cotton"])

        builder = newton.ModelBuilder()
        add_ground_plane(builder)

        # Obstacle(s) — cone/sphere on the ground
        obstacle_z = args.cone_height if args.obstacle != "sphere" else args.cone_radius
        if args.obstacle in ("cone", "both"):
            add_cone(builder, position=(0.0, 0.0, obstacle_z), radius=args.cone_radius, half_height=args.cone_height)
            print(f"  Cone: radius={args.cone_radius}, half_height={args.cone_height}")
        if args.obstacle in ("sphere", "both"):
            sz = obstacle_z + (0.25 if args.obstacle == "both" else 0.0)
            add_sphere(builder, position=(0.0, 0.0, sz), radius=0.15)
            print(f"  Sphere: radius=0.15 at z={sz:.2f}")

        # Cloth
        cloth_z = obstacle_z + args.cone_height + args.drop_height
        if args.use_grid:
            size = 1.2
            cell = size / args.grid_res
            builder.add_cloth_grid(
                pos=wp.vec3(-(size / 2), -(size / 2), cloth_z),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=args.grid_res,
                dim_y=args.grid_res,
                cell_x=cell,
                cell_y=cell,
                mass=cloth_params["density"] * cell * cell,
                tri_ke=cloth_params["tri_ke"],
                tri_ka=cloth_params["tri_ka"],
                tri_kd=cloth_params["tri_kd"],
                edge_ke=cloth_params["edge_ke"],
                edge_kd=cloth_params["edge_kd"],
                particle_radius=0.005,
            )
            print(f"  Grid cloth: {args.grid_res}x{args.grid_res}, size={size}m, preset={args.preset}")
        else:
            garments = discover_original_garments(ORIGINAL_GARMENT_DIR, args.garment)
            if not garments:
                garments = discover_garments(config.get("garment_dir", ""))
            if not garments:
                print("ERROR: No garments found. Use --use-grid or set garment_dir in config.")
                sys.exit(1)

            if args.garment:
                matches = [g for g in garments if args.garment.lower() in g["name"].lower()]
                if matches:
                    garment = random.Random(args.seed).choice(matches)
                else:
                    print(f"No garment matching '{args.garment}', picking random")
                    garment = random.Random(args.seed).choice(garments)
            else:
                garment = random.Random(args.seed).choice(garments)

            print(f"  Garment: {garment['name']} ({garment['category']}), preset={args.preset}")
            add_garment_as_cloth(
                builder,
                garment,
                position=(0.0, 0.0, cloth_z),
                **cloth_params,
            )

        # Color for VBD (skip for XPBD)
        if args.solver == "vbd":
            builder.color()

        model = finalize_cloth_model(builder, solver_type=args.solver)

        # WORKAROUND: Known Newton bug — some particles get zero/tiny mass
        # and behave as pinned/kinematic. Two independent checks in VBD:
        #   forward_step checks inv_mass == 0 → pins particle inertia
        #   solve_elasticity checks mass == 0 → pins particle displacement
        # Must fix BOTH arrays.
        mass_np = model.particle_mass.numpy()
        nonzero = mass_np[mass_np > 0]
        if len(nonzero) > 0:
            min_mass = float(np.median(nonzero) * 0.1)
            pinned = mass_np < min_mass
            num_pinned = int(np.sum(pinned))
            if num_pinned > 0:
                print(f"  WORKAROUND: {num_pinned} particles had near-zero mass (pinned), clamping to {min_mass:.2e}")
                mass_np[pinned] = min_mass
                model.particle_mass.assign(mass_np)
                # CRITICAL: Also fix inv_mass — forward_step checks this independently
                inv_mass_np = np.divide(1.0, mass_np, out=np.zeros_like(mass_np), where=mass_np != 0.0)
                model.particle_inv_mass.assign(inv_mass_np)

        # Also ensure all particle flags have ACTIVE bit set
        flags_np = model.particle_flags.numpy()
        inactive = (flags_np & 1) == 0
        num_inactive = int(np.sum(inactive))
        if num_inactive > 0:
            print(f"  WORKAROUND: {num_inactive} particles had ACTIVE flag cleared, setting to ACTIVE")
            flags_np[inactive] |= 1
            model.particle_flags.assign(flags_np)

        solver = build_solver(model, solver_type=args.solver, iterations=args.solver_iters, self_contact=True)

        # Spring force assist — inject mass-spring forces for stuck particles
        spring_assist = None
        if args.spring_assist and model.particle_count > 0:
            spring_assist = create_spring_assist(
                model,
                ks=args.spring_ks,
                kb=args.spring_kb,
                boost_factor=args.spring_boost,
            )

        print(f"  Solver: {args.solver.upper()}")
        print("  Self-contact: ENABLED (radius=0.002, margin=0.003)")
        print(f"  Drop height: {args.drop_height}m above obstacle top")
        print(f"  Particles: {model.particle_count}")
        print(f"  Device: {device}")

        # Viewer
        try:
            viewer = create_viewer(viewer_type=args.viewer)
        except ImportError:
            print(f"  WARNING: {args.viewer} viewer unavailable, using null")
            viewer = create_viewer(viewer_type="null")

        # Simulation loop
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.02)
        contacts = collision_pipeline.contacts()
        dt = 1.0 / 60.0
        sub_dt = dt / args.substeps

        if viewer:
            viewer.set_model(model)
            viewer.begin_frame(0.0)
            viewer.log_state(state_0)
            viewer.end_frame()

        # Record initial positions for stuck particle analysis
        pos_initial = state_0.particle_q.numpy().copy()

        for frame in range(args.num_frames):
            for _ in range(args.substeps):
                state_0.clear_forces()

                if spring_assist is not None:
                    spring_assist.apply(state_0)

                collision_pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sub_dt)
                state_0, state_1 = state_1, state_0

            if viewer:
                viewer.begin_frame((frame + 1) * dt)
                viewer.log_state(state_0)
                viewer.end_frame()

            if (frame + 1) % 50 == 0:
                pq = state_0.particle_q.numpy()
                z_range = pq[:, 2].max() - pq[:, 2].min()
                print(f"  Frame {frame + 1}/{args.num_frames}  z_range={z_range:.3f}m")

        if viewer:
            print("\n  Simulation complete. Press Enter in the terminal to close the viewer...")
            input()
            viewer.close()

        pq = state_0.particle_q.numpy()
        print("\n  Final cloth bbox:")
        print(f"    min = {np.min(pq, axis=0)}")
        print(f"    max = {np.max(pq, axis=0)}")
        print(f"    z_range = {pq[:, 2].max() - pq[:, 2].min():.4f}m")

        # Stuck particle analysis
        displacement = np.linalg.norm(pq - pos_initial, axis=1)
        stuck_count = int(np.sum(displacement < 0.001))
        total = len(displacement)
        print("\n  Stuck particle analysis (threshold=1mm):")
        print(f"    Stuck: {stuck_count} / {total} ({100 * stuck_count / max(total, 1):.1f}%)")
        print(f"    Moving: {total - stuck_count} / {total}")
        if spring_assist is not None:
            spring_assist.log_stats()
        print("  Done.")


if __name__ == "__main__":
    main()
