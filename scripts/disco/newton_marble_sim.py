"""
Newton Marble Run Simulation
================================
Drops marbles (spheres) through a funnel / ramp setup with DTC props as obstacles.

Usage:
    uv run python scripts/disco/newton_marble_sim.py
    uv run python scripts/disco/newton_marble_sim.py --seed 42 --num-marbles 20
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import warp as wp
from newton_sim_utils import (
    TEST_FRAMES,
    add_ground_plane,
    create_blend_file,
    create_viewer,
    load_config,
    write_sim_metadata,
)

import newton

MARBLE_RADIUS = 0.008  # 16mm diameter glass marble
MARBLE_DENSITY = 2500.0  # glass kg/m3


def build_marble_scene(
    seed: int = 0,
    num_marbles: int = 15,
    num_ramps: int = 4,
    drop_height: float = 1.5,
    solver_iters: int = 10,
):
    """Build marble run: ramps (angled boxes) + marbles dropped from above."""
    rng = random.Random(seed)

    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    # Container walls
    wall_cfg = newton.ModelBuilder.ShapeConfig()
    wall_cfg.mu = 0.2
    wall_cfg.restitution = 0.5

    container_width = 0.3
    # Back wall
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(p=wp.vec3(0.0, -0.01, drop_height / 2), q=wp.quat_identity()),
        hx=container_width,
        hy=0.005,
        hz=drop_height / 2,
        cfg=wall_cfg,
    )
    # Front wall (transparent in rendering)
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(p=wp.vec3(0.0, 0.06, drop_height / 2), q=wp.quat_identity()),
        hx=container_width,
        hy=0.005,
        hz=drop_height / 2,
        cfg=wall_cfg,
    )

    # Ramps (angled static boxes)
    ramp_cfg = newton.ModelBuilder.ShapeConfig()
    ramp_cfg.mu = 0.15
    ramp_cfg.restitution = 0.6

    ramp_info = []
    for i in range(num_ramps):
        z = drop_height - (i + 1) * (drop_height / (num_ramps + 1))
        # Alternate ramp direction
        sign = 1.0 if i % 2 == 0 else -1.0
        x = sign * rng.uniform(0.02, 0.12)
        angle = sign * rng.uniform(0.2, 0.5)  # ramp angle

        q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(x, 0.025, z), q=q),
            hx=0.12,
            hy=0.025,
            hz=0.005,
            cfg=ramp_cfg,
        )
        ramp_info.append({"x": x, "z": z, "angle": angle})

    # Marbles
    marble_cfg = newton.ModelBuilder.ShapeConfig()
    marble_cfg.density = MARBLE_DENSITY
    marble_cfg.mu = 0.08
    marble_cfg.restitution = 0.85  # bouncy glass

    for i in range(num_marbles):
        px = rng.uniform(-0.05, 0.05)
        py = rng.uniform(0.01, 0.04)
        pz = drop_height + rng.uniform(0.0, 0.3)

        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(px, py, pz), q=wp.quat_identity()),
        )
        builder.add_shape_sphere(body=body, radius=MARBLE_RADIUS, cfg=marble_cfg)

    model = builder.finalize()
    solver = newton.solvers.SolverXPBD(model, iterations=solver_iters)

    scene_info = {
        "num_marbles": num_marbles,
        "num_ramps": num_ramps,
        "drop_height": drop_height,
    }

    return model, solver, scene_info


def run_marble_sim(
    model,
    solver,
    num_frames: int = 300,
    substeps: int = 10,
    dt: float = 1.0 / 60.0,
    viewer=None,
):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    sub_dt = dt / substeps

    if viewer:
        viewer.set_model(model)
        viewer.begin_frame(0.0)
        viewer.log_state(state_0)
        viewer.end_frame()

    for frame in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

        if viewer:
            viewer.begin_frame((frame + 1) * dt)
            viewer.log_state(state_0)
            viewer.end_frame()

    if viewer:
        viewer.close()

    return state_0


def main():
    parser = argparse.ArgumentParser(description="Newton Marble Run Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-marbles", type=int, default=15)
    parser.add_argument("--num-ramps", type=int, default=4)
    parser.add_argument("--drop-height", type=float, default=1.5)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--solver-iters", type=int, default=10)
    parser.add_argument("--viewer", type=str, default="null", choices=["null", "usd", "gl"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--test", action="store_true", help="Test run: few frames, GL viewer")
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
        sim_name = f"marble_s{args.seed:06d}"
        sim_dir = output_dir / "marble" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Marble Run (seed={args.seed}) ===")

        model, solver, scene_info = build_marble_scene(
            seed=args.seed,
            num_marbles=args.num_marbles,
            num_ramps=args.num_ramps,
            drop_height=args.drop_height,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_marble_sim(
            model,
            solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="marble",
            seed=args.seed,
            **scene_info,
        )

        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
