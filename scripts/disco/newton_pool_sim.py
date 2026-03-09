"""
Newton Pool Break Simulation
================================
Simulates a pool/billiards break shot with 16 balls on a table.

Usage:
    uv run python scripts/disco/newton_pool_sim.py
    uv run python scripts/disco/newton_pool_sim.py --seed 42 --cue-speed 4.0
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
    create_viewer,
    load_config,
    write_sim_metadata,
)

# Pool ball constants
BALL_RADIUS = 0.02625  # 52.5mm diameter (standard)
BALL_MASS = 0.170  # kg (~6 oz)
TABLE_LENGTH = 2.24  # 7-foot table inner
TABLE_WIDTH = 1.12
CUSHION_HEIGHT = 0.04
CUSHION_THICKNESS = 0.05


def _rack_positions(start_x: float = 0.5) -> list[tuple[float, float]]:
    """Standard 15-ball triangle rack positions (x, y)."""
    positions = []
    spacing = BALL_RADIUS * 2.02  # tight pack
    rows = [1, 2, 3, 4, 5]
    for row_idx, count in enumerate(rows):
        x = start_x + row_idx * spacing * math.sqrt(3) / 2
        y_start = -(count - 1) * spacing / 2
        for j in range(count):
            y = y_start + j * spacing
            positions.append((x, y))
    return positions


def build_pool_scene(
    seed: int = 0,
    cue_speed: float = 3.5,
    aim_offset_deg: float | None = None,
    solver_iters: int = 10,
):
    """Build pool break scene: table, cushions, 16 balls."""
    rng = random.Random(seed)

    if aim_offset_deg is None:
        aim_offset_deg = rng.uniform(-2.0, 2.0)

    builder = newton.ModelBuilder()

    table_z = 0.75  # table surface height

    # Table surface (static)
    felt_cfg = newton.ModelBuilder.ShapeConfig()
    felt_cfg.mu = 0.2  # felt friction
    felt_cfg.restitution = 0.3
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, table_z - 0.02), q=wp.quat_identity()),
        hx=TABLE_LENGTH / 2, hy=TABLE_WIDTH / 2, hz=0.02,
        cfg=felt_cfg,
    )

    # Cushions (4 walls)
    cushion_cfg = newton.ModelBuilder.ShapeConfig()
    cushion_cfg.mu = 0.3
    cushion_cfg.restitution = 0.7  # bouncy cushions

    # Long cushions (left/right along X)
    for sign in [-1, 1]:
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                p=wp.vec3(0.0, sign * (TABLE_WIDTH / 2 + CUSHION_THICKNESS / 2), table_z + CUSHION_HEIGHT / 2),
                q=wp.quat_identity(),
            ),
            hx=TABLE_LENGTH / 2 + CUSHION_THICKNESS,
            hy=CUSHION_THICKNESS / 2,
            hz=CUSHION_HEIGHT / 2,
            cfg=cushion_cfg,
        )

    # Short cushions (top/bottom along Y)
    for sign in [-1, 1]:
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                p=wp.vec3(sign * (TABLE_LENGTH / 2 + CUSHION_THICKNESS / 2), 0.0, table_z + CUSHION_HEIGHT / 2),
                q=wp.quat_identity(),
            ),
            hx=CUSHION_THICKNESS / 2,
            hy=TABLE_WIDTH / 2,
            hz=CUSHION_HEIGHT / 2,
            cfg=cushion_cfg,
        )

    # Balls
    ball_cfg = newton.ModelBuilder.ShapeConfig()
    ball_cfg.density = BALL_MASS / ((4 / 3) * math.pi * BALL_RADIUS**3)
    ball_cfg.mu = 0.05  # ball-to-ball friction
    ball_cfg.restitution = 0.92  # elastic collisions

    ball_z = table_z + BALL_RADIUS

    # Racked balls (15)
    rack_pos = _rack_positions(start_x=0.3)
    for bx, by in rack_pos:
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(bx, by, ball_z), q=wp.quat_identity()),
        )
        builder.add_shape_sphere(body=body, radius=BALL_RADIUS, cfg=ball_cfg)

    # Cue ball
    aim_rad = math.radians(aim_offset_deg)
    vx = cue_speed * math.cos(aim_rad)
    vy = cue_speed * math.sin(aim_rad)

    cue_body = builder.add_body(
        xform=wp.transform(p=wp.vec3(-0.7, 0.0, ball_z), q=wp.quat_identity()),
    )
    builder.add_shape_sphere(body=cue_body, radius=BALL_RADIUS, cfg=ball_cfg)

    model = builder.finalize()
    solver = newton.solvers.SolverXPBD(model, iterations=solver_iters)

    # Set cue ball velocity
    state = model.state()
    if model.body_count > 0:
        body_qd = state.body_qd.numpy()
        cue_idx = model.body_count - 1
        body_qd[cue_idx] = [0.0, 0.0, 0.0, vx, vy, 0.0]
        state.body_qd = wp.array(body_qd, dtype=wp.spatial_vector)

    scene_info = {
        "cue_speed": cue_speed,
        "aim_offset_deg": aim_offset_deg,
        "num_balls": 16,
    }

    return model, solver, state, scene_info


def run_pool_sim(
    model,
    solver,
    initial_state,
    num_frames: int = 300,
    substeps: int = 10,
    dt: float = 1.0 / 60.0,
    viewer=None,
):
    state_0 = initial_state
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, None, state_0)
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
    parser = argparse.ArgumentParser(description="Newton Pool Break Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cue-speed", type=float, default=3.5)
    parser.add_argument("--aim-offset", type=float, default=None)
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
        sim_name = f"pool_s{args.seed:06d}"
        sim_dir = output_dir / "pool" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Pool Break (seed={args.seed}) ===")

        model, solver, initial_state, scene_info = build_pool_scene(
            seed=args.seed,
            cue_speed=args.cue_speed,
            aim_offset_deg=args.aim_offset,
            solver_iters=args.solver_iters,
        )

        usd_path = sim_dir / f"{sim_name}.usd" if args.viewer == "usd" else None
        viewer = create_viewer(output_path=usd_path, viewer_type=args.viewer)

        final_state = run_pool_sim(
            model, solver, initial_state,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="pool",
            seed=args.seed,
            **scene_info,
        )

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
