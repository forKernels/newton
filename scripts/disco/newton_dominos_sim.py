"""
Newton Dominos Simulation
============================
Line of thin box dominoes toppling in sequence.

Usage:
    uv run python scripts/disco/newton_dominos_sim.py
    uv run python scripts/disco/newton_dominos_sim.py --test
    uv run python scripts/disco/newton_dominos_sim.py --seed 42 --num-dominos 30 --curve
"""

from __future__ import annotations

import argparse
import math
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

# Standard domino dimensions
DOMINO_HX = 0.005   # 1cm thick
DOMINO_HY = 0.012   # 2.4cm wide
DOMINO_HZ = 0.024   # 4.8cm tall
DOMINO_DENSITY = 1800.0  # bone/plastic


def build_dominos_scene(
    seed: int = 0,
    num_dominos: int = 20,
    spacing: float = 0.03,
    curve: bool = False,
    curve_radius: float = 0.5,
    solver_iters: int = 10,
):
    """Build dominos scene: line of upright thin boxes, first one tipped."""
    rng = random.Random(seed)

    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    domino_cfg = newton.ModelBuilder.ShapeConfig()
    domino_cfg.density = DOMINO_DENSITY
    domino_cfg.mu = 0.4
    domino_cfg.restitution = 0.1

    for i in range(num_dominos):
        if curve:
            angle = i * spacing / curve_radius
            px = curve_radius * math.sin(angle)
            py = curve_radius * (1.0 - math.cos(angle))
            facing = angle
        else:
            px = i * spacing
            py = rng.uniform(-0.001, 0.001)  # tiny wobble
            facing = 0.0

        pz = DOMINO_HZ  # standing upright

        # First domino is tilted to start the chain
        if i == 0:
            tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.2)
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), facing) * tilt
        else:
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), facing)

        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(px, py, pz), q=rot),
        )
        builder.add_shape_box(body=body, hx=DOMINO_HX, hy=DOMINO_HY, hz=DOMINO_HZ, cfg=domino_cfg)

    model = builder.finalize()
    solver = newton.solvers.SolverXPBD(model, iterations=solver_iters)

    scene_info = {
        "num_dominos": num_dominos,
        "spacing": spacing,
        "curve": curve,
        "curve_radius": curve_radius if curve else None,
    }

    return model, solver, scene_info


def run_dominos_sim(
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

        if (frame + 1) % 50 == 0:
            print(f"  Frame {frame + 1}/{num_frames}")

    if viewer:
        viewer.close()

    return state_0


def count_fallen(model, state, num_dominos: int) -> int:
    """Count dominos that have tipped (z below half height)."""
    if model.body_count < num_dominos:
        return 0
    body_q = state.body_q.numpy()
    fallen = 0
    for i in range(num_dominos):
        if body_q[i][2] < DOMINO_HZ * 0.7:
            fallen += 1
    return fallen


def main():
    parser = argparse.ArgumentParser(description="Newton Dominos Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-dominos", type=int, default=20)
    parser.add_argument("--spacing", type=float, default=0.03)
    parser.add_argument("--curve", action="store_true",
                        help="Arrange dominos in a curve")
    parser.add_argument("--curve-radius", type=float, default=0.5)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--substeps", type=int, default=10)
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
        sim_name = f"dominos_s{args.seed:06d}"
        sim_dir = output_dir / "dominos" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Dominos (seed={args.seed}) ===")

        model, solver, scene_info = build_dominos_scene(
            seed=args.seed,
            num_dominos=args.num_dominos,
            spacing=args.spacing,
            curve=args.curve,
            curve_radius=args.curve_radius,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_dominos_sim(
            model, solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        fallen = count_fallen(model, final_state, args.num_dominos)
        print(f"  Dominos fallen: {fallen}/{args.num_dominos}")

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="dominos",
            seed=args.seed,
            dominos_fallen=fallen,
            **scene_info,
        )

        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
