"""
Newton Bowling Simulation
============================
Rolls a ball toward pins using XPBD rigid body dynamics.

Usage:
    uv run python scripts/disco/newton_bowling_sim.py
    uv run python scripts/disco/newton_bowling_sim.py --seed 42 --speed 4.0
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

# Bowling geometry constants
BALL_RADIUS = 0.109  # ~21.8 cm diameter
BALL_MASS = 6.8  # kg
PIN_RADIUS = 0.028  # ~56 mm diameter at widest
PIN_HEIGHT = 0.381  # 15 inches
PIN_MASS = 1.5  # kg
LANE_WIDTH = 1.06  # ~42 inches
PIN_SPACING = 0.305  # ~12 inches center-to-center


def _pin_positions() -> list[tuple[float, float, float]]:
    """Standard 10-pin triangle layout. Ball rolls in +X, pins at ~18m.
    For sim we compress to ~3m lane.
    """
    pin_z = PIN_HEIGHT / 2  # capsule half-height
    positions = []
    rows = [1, 2, 3, 4]
    start_x = 3.0  # pins at 3m
    for row_idx, count in enumerate(rows):
        x = start_x + row_idx * PIN_SPACING
        y_start = -(count - 1) * PIN_SPACING / 2
        for j in range(count):
            y = y_start + j * PIN_SPACING
            positions.append((x, y, pin_z))
    return positions


def build_bowling_scene(
    seed: int = 0,
    speed: float = 3.5,
    aim_offset_deg: float | None = None,
    solver_iters: int = 10,
):
    """Build bowling scene: lane, ball, 10 pins."""
    rng = random.Random(seed)

    if aim_offset_deg is None:
        aim_offset_deg = rng.uniform(-4.0, 4.0)
    speed = speed or rng.uniform(2.5, 4.5)

    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    # Lane surface (static box)
    lane_cfg = newton.ModelBuilder.ShapeConfig()
    lane_cfg.mu = 0.04  # oiled lane
    lane_cfg.restitution = 0.1
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(p=wp.vec3(2.0, 0.0, -0.05), q=wp.quat_identity()),
        hx=3.0, hy=LANE_WIDTH / 2, hz=0.05,
        cfg=lane_cfg,
    )

    # Pins (capsules)
    pin_cfg = newton.ModelBuilder.ShapeConfig()
    pin_cfg.density = PIN_MASS / (math.pi * PIN_RADIUS**2 * PIN_HEIGHT)
    pin_cfg.mu = 0.35
    pin_cfg.restitution = 0.15

    pin_positions = _pin_positions()
    for i, (px, py, pz) in enumerate(pin_positions):
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(px, py, pz), q=wp.quat_identity()),
        )
        builder.add_shape_capsule(
            body=body,
            radius=PIN_RADIUS,
            half_height=PIN_HEIGHT / 2 - PIN_RADIUS,
            cfg=pin_cfg,
        )

    # Ball
    aim_rad = math.radians(aim_offset_deg)
    vx = speed * math.cos(aim_rad)
    vy = speed * math.sin(aim_rad)

    ball_cfg = newton.ModelBuilder.ShapeConfig()
    ball_cfg.density = BALL_MASS / ((4 / 3) * math.pi * BALL_RADIUS**3)
    ball_cfg.mu = 0.30
    ball_cfg.restitution = 0.05

    ball_body = builder.add_body(
        xform=wp.transform(
            p=wp.vec3(0.0, 0.0, BALL_RADIUS),
            q=wp.quat_identity(),
        ),
    )
    builder.add_shape_sphere(body=ball_body, radius=BALL_RADIUS, cfg=ball_cfg)

    model = builder.finalize()
    solver = newton.solvers.SolverXPBD(model, iterations=solver_iters)

    # Set initial ball velocity
    state = model.state()
    if model.body_count > 0:
        body_qd = state.body_qd.numpy()
        ball_idx = model.body_count - 1  # ball is last body added
        body_qd[ball_idx] = [0.0, 0.0, 0.0, vx, vy, 0.0]  # angular, linear
        state.body_qd = wp.array(body_qd, dtype=wp.spatial_vector)

    scene_info = {
        "speed": speed,
        "aim_offset_deg": aim_offset_deg,
        "num_pins": len(pin_positions),
    }

    return model, solver, state, scene_info


def run_bowling_sim(
    model,
    solver,
    initial_state,
    num_frames: int = 200,
    substeps: int = 10,
    dt: float = 1.0 / 60.0,
    viewer=None,
):
    state_0 = initial_state
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


def count_fallen_pins(model, state, pin_count: int = 10) -> int:
    """Count pins that have tipped over (z-angle > 30 deg)."""
    if model.body_count < pin_count + 1:
        return 0

    body_q = state.body_q.numpy()
    fallen = 0
    for i in range(pin_count):
        q = body_q[i]  # pins are first bodies
        # Check if pin center is below half height (tipped)
        z_pos = q[2]
        if z_pos < PIN_HEIGHT / 4:
            fallen += 1
    return fallen


def main():
    parser = argparse.ArgumentParser(description="Newton Bowling Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=3.5)
    parser.add_argument("--aim-offset", type=float, default=None,
                        help="Aim offset [degrees]")
    parser.add_argument("--num-frames", type=int, default=200)
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
        sim_name = f"bowling_s{args.seed:06d}"
        sim_dir = output_dir / "bowling" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Bowling (seed={args.seed}) ===")

        model, solver, initial_state, scene_info = build_bowling_scene(
            seed=args.seed,
            speed=args.speed,
            aim_offset_deg=args.aim_offset,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_bowling_sim(
            model, solver, initial_state,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        fallen = count_fallen_pins(model, final_state)
        print(f"  Pins fallen: {fallen}/10")

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="bowling",
            seed=args.seed,
            pins_fallen=fallen,
            **scene_info,
        )

        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
