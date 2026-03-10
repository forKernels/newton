"""
Newton Card Stacking / Scatter Simulation
============================================
Drops and stacks playing cards on a surface, optionally knocked by a sphere.
Based on Newton's poker_cards_stacking example — cards are cloth grids with
high bending stiffness to stay rigid-like while interacting naturally.

Usage:
    uv run python scripts/disco/newton_cards_sim.py
    uv run python scripts/disco/newton_cards_sim.py --test
    uv run python scripts/disco/newton_cards_sim.py --seed 42 --num-cards 52 --knock
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import warp as wp

import newton

from newton_sim_utils import (
    TEST_FRAMES,
    add_ground_plane,
    create_blend_file,
    create_viewer,
    load_config,
    write_sim_metadata,
)

# Standard poker card dimensions
CARD_WIDTH = 0.0635   # 6.35 cm
CARD_HEIGHT = 0.0889  # 8.89 cm
CARD_MASS = 1.8e-3    # 1.8 grams
CARD_DIM_X = 4        # cells along width
CARD_DIM_Y = 6        # cells along height


def build_cards_scene(
    seed: int = 0,
    num_cards: int = 52,
    drop_spread: float = 0.005,
    platform_size: float = 0.1,
    platform_height: float = 0.1,
    knock: bool = False,
    solver_iters: int = 10,
):
    """Build card stacking scene: cards drop onto platform, optionally knocked."""
    rng = np.random.default_rng(seed)

    builder = newton.ModelBuilder(gravity=-9.8)
    add_ground_plane(builder)

    # Platform (static cube)
    body_cube = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, platform_height), q=wp.quat_identity()),
    )
    cube_cfg = newton.ModelBuilder.ShapeConfig()
    cube_cfg.density = 0.0
    cube_cfg.ke = 5.0e6
    cube_cfg.kd = 1.0e-4
    cube_cfg.mu = 0.1
    builder.add_shape_box(
        body_cube, hx=platform_size, hy=platform_size, hz=platform_size, cfg=cube_cfg,
    )

    # Kinematic sphere for knocking cards
    sphere_body_idx = None
    if knock:
        body_sphere = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.35, 0.0, 0.22), q=wp.quat_identity()),
        )
        sphere_cfg = newton.ModelBuilder.ShapeConfig()
        sphere_cfg.density = 0.0
        sphere_cfg.ke = 1.0e5
        sphere_cfg.kd = 1.0e-4
        sphere_cfg.mu = 0.3
        builder.add_shape_sphere(body_sphere, radius=0.02, cfg=sphere_cfg)
        sphere_body_idx = body_sphere

    # Card cloth params
    cell_x = CARD_WIDTH / CARD_DIM_X
    cell_y = CARD_HEIGHT / CARD_DIM_Y
    num_particles_per_card = (CARD_DIM_X + 1) * (CARD_DIM_Y + 1)
    mass_per_particle = CARD_MASS / num_particles_per_card

    drop_base = platform_height + platform_size + 0.05

    for i in range(num_cards):
        ox = rng.uniform(-drop_spread, drop_spread)
        oy = rng.uniform(-drop_spread, drop_spread)
        oz = drop_base + i * 0.001
        angle = rng.uniform(-0.1, 0.1)

        builder.add_cloth_grid(
            pos=wp.vec3(-CARD_WIDTH / 2 + ox, -CARD_HEIGHT / 2 + oy, oz),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=CARD_DIM_X,
            dim_y=CARD_DIM_Y,
            cell_x=cell_x,
            cell_y=cell_y,
            mass=mass_per_particle,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=1.0e-4,
            edge_ke=1.0e2,  # high bending = rigid card
            edge_kd=1.0e-2,
            particle_radius=0.003,
        )

    builder.color(include_bending=True)
    model = builder.finalize()

    model.soft_contact_ke = 1.0e5
    model.soft_contact_kd = 1.0e-4
    model.soft_contact_mu = 0.3

    solver = newton.solvers.SolverVBD(
        model, iterations=solver_iters,
        particle_enable_self_contact=True,
        particle_self_contact_radius=0.001,
        particle_self_contact_margin=0.0015,
        particle_topological_contact_filter_threshold=2,
        particle_rest_shape_contact_exclusion_radius=0.0,
    )

    # Custom collision pipeline for cards
    collision_pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=0.005,
    )

    scene_info = {
        "num_cards": num_cards,
        "knock": knock,
        "platform_size": platform_size,
        "platform_height": platform_height,
    }

    return model, solver, collision_pipeline, sphere_body_idx, scene_info


def run_cards_sim(
    model,
    solver,
    collision_pipeline,
    sphere_body_idx,
    knock: bool = False,
    num_frames: int = 250,
    substeps: int = 20,
    dt: float = 1.0 / 60.0,
    viewer=None,
):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = collision_pipeline.contacts()

    sub_dt = dt / substeps
    sphere_x = -0.35
    sphere_vx = 0.5

    if viewer:
        viewer.set_model(model)
        viewer.begin_frame(0.0)
        viewer.log_state(state_0)
        viewer.end_frame()

    for frame in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()

            # Animate sphere
            if knock and sphere_body_idx is not None:
                sphere_x += sphere_vx * sub_dt
                body_q = state_0.body_q.numpy()
                body_q[sphere_body_idx][0] = sphere_x
                state_0.body_q = wp.array(body_q, dtype=wp.transform)

            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

        if viewer:
            viewer.begin_frame((frame + 1) * dt)
            viewer.log_state(state_0)
            viewer.end_frame()

        if (frame + 1) % 50 == 0:
            print(f"  Frame {frame + 1}/{num_frames}")

    if viewer:
        viewer.close()

    return state_0


def main():
    parser = argparse.ArgumentParser(description="Newton Card Stacking Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-cards", type=int, default=52)
    parser.add_argument("--knock", action="store_true",
                        help="Add kinematic sphere to knock cards off")
    parser.add_argument("--platform-size", type=float, default=0.1)
    parser.add_argument("--num-frames", type=int, default=250)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--solver-iters", type=int, default=10)
    parser.add_argument("--viewer", type=str, default="null",
                        choices=["null", "usd", "gl"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--test", action="store_true",
                        help="Test run: few frames, GL viewer")
    args = parser.parse_args()

    if args.test:
        args.num_frames = TEST_FRAMES
        args.viewer = args.viewer if args.viewer != "null" else "gl"

    wp.init()
    if args.device:
        device = wp.get_device(args.device)
    else:
        device = wp.get_preferred_device()

    with wp.ScopedDevice(device):
        config = load_config(args.config)

        output_dir = Path(args.output_dir) if args.output_dir else Path(config.get("output_dir", "./output"))
        sim_name = f"cards_s{args.seed:06d}"
        sim_dir = output_dir / "cards" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Cards (seed={args.seed}, {args.num_cards} cards) ===")

        model, solver, collision_pipeline, sphere_body_idx, scene_info = build_cards_scene(
            seed=args.seed,
            num_cards=args.num_cards,
            knock=args.knock,
            platform_size=args.platform_size,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_cards_sim(
            model, solver, collision_pipeline, sphere_body_idx,
            knock=args.knock,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        particle_q = final_state.particle_q.numpy()
        bbox_size = np.linalg.norm(np.max(particle_q, axis=0) - np.min(particle_q, axis=0))

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="cards",
            seed=args.seed,
            bbox_size=float(bbox_size),
            **scene_info,
        )

        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
