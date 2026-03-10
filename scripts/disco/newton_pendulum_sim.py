"""
Newton Pendulum / Chain Simulation
=====================================
Articulated chain (Newton's cradle style or free-swinging pendulum).
Bodies connected by revolute joints, dropped from various configurations.

Usage:
    uv run python scripts/disco/newton_pendulum_sim.py
    uv run python scripts/disco/newton_pendulum_sim.py --test
    uv run python scripts/disco/newton_pendulum_sim.py --seed 42 --num-links 10 --cradle
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


def build_pendulum_scene(
    seed: int = 0,
    num_links: int = 5,
    link_length: float = 0.2,
    link_radius: float = 0.02,
    cradle: bool = False,
    num_pendulums: int = 5,
    hang_height: float = 1.5,
    solver_iters: int = 10,
):
    """Build pendulum scene: single chain or Newton's cradle."""
    rng = random.Random(seed)

    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    if cradle:
        # Newton's cradle: N pendulums in a row, leftmost raised
        spacing = link_radius * 2.5
        for p in range(num_pendulums):
            px = (p - (num_pendulums - 1) / 2) * spacing
            anchor = wp.vec3(px, 0.0, hang_height)

            # Fixed anchor point
            anchor_body = builder.add_body(
                xform=wp.transform(p=anchor, q=wp.quat_identity()),
            )
            anchor_cfg = newton.ModelBuilder.ShapeConfig()
            anchor_cfg.density = 0.0
            builder.add_shape_sphere(body=anchor_body, radius=0.005, cfg=anchor_cfg)

            # Pendulum ball
            if p == 0:
                # Leftmost ball raised
                ball_pos = wp.vec3(px - link_length * 0.7, 0.0, hang_height)
            else:
                ball_pos = wp.vec3(px, 0.0, hang_height - link_length)

            ball_body = builder.add_body(
                xform=wp.transform(p=ball_pos, q=wp.quat_identity()),
            )
            ball_cfg = newton.ModelBuilder.ShapeConfig()
            ball_cfg.density = 7800.0  # steel
            ball_cfg.mu = 0.05
            ball_cfg.restitution = 0.95
            builder.add_shape_sphere(body=ball_body, radius=link_radius * 2, cfg=ball_cfg)

            # Connect anchor to ball with ball joint
            builder.add_joint_ball(
                parent_body=anchor_body,
                child_body=ball_body,
                parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, link_length), q=wp.quat_identity()),
            )
    else:
        # Free chain: series of capsules connected by revolute joints
        parent = -1
        initial_angle = rng.uniform(0.3, 1.2)  # start angle from vertical

        for i in range(num_links):
            angle = initial_angle if i == 0 else 0.0
            px = hang_height * math.sin(angle) if i == 0 else 0.0
            pz = hang_height - i * link_length

            body = builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(px, 0.0, pz),
                    q=wp.quat_identity(),
                ),
            )
            builder.add_shape_capsule(body=body, radius=link_radius, half_height=link_length / 2 - link_radius)

            if parent == -1:
                # First link: fixed joint to world
                builder.add_joint_revolute(
                    parent_body=-1,
                    child_body=body,
                    parent_xform=wp.transform(
                        p=wp.vec3(0.0, 0.0, hang_height),
                        q=wp.quat_identity(),
                    ),
                    child_xform=wp.transform(
                        p=wp.vec3(0.0, 0.0, link_length / 2),
                        q=wp.quat_identity(),
                    ),
                    axis=wp.vec3(0.0, 1.0, 0.0),
                )
            else:
                builder.add_joint_revolute(
                    parent_body=parent,
                    child_body=body,
                    parent_xform=wp.transform(
                        p=wp.vec3(0.0, 0.0, -link_length / 2),
                        q=wp.quat_identity(),
                    ),
                    child_xform=wp.transform(
                        p=wp.vec3(0.0, 0.0, link_length / 2),
                        q=wp.quat_identity(),
                    ),
                    axis=wp.vec3(0.0, 1.0, 0.0),
                )
            parent = body

    model = builder.finalize()
    solver = newton.solvers.SolverXPBD(model, iterations=solver_iters)

    scene_info = {
        "cradle": cradle,
        "num_links": num_links if not cradle else num_pendulums,
        "link_length": link_length,
        "hang_height": hang_height,
    }

    return model, solver, scene_info


def run_pendulum_sim(
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

    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
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
    parser = argparse.ArgumentParser(description="Newton Pendulum / Chain Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-links", type=int, default=5)
    parser.add_argument("--link-length", type=float, default=0.2)
    parser.add_argument("--cradle", action="store_true",
                        help="Newton's cradle mode")
    parser.add_argument("--num-pendulums", type=int, default=5,
                        help="Number of pendulums in cradle mode")
    parser.add_argument("--hang-height", type=float, default=1.5)
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
        sim_name = f"pendulum_s{args.seed:06d}"
        sim_dir = output_dir / "pendulum" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Pendulum (seed={args.seed}, {'cradle' if args.cradle else 'chain'}) ===")

        model, solver, scene_info = build_pendulum_scene(
            seed=args.seed,
            num_links=args.num_links,
            link_length=args.link_length,
            cradle=args.cradle,
            num_pendulums=args.num_pendulums,
            hang_height=args.hang_height,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_pendulum_sim(
            model, solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="pendulum",
            seed=args.seed,
            **scene_info,
        )

        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
