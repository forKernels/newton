"""
Newton Lego Drop Simulation
===============================
Drops lego-like brick assemblies from height, breaking apart on impact.

Loads lego model meshes from lego_model_dir if available,
otherwise generates procedural brick stacks.

Usage:
    uv run python scripts/disco/newton_lego_sim.py
    uv run python scripts/disco/newton_lego_sim.py --seed 42 --num-bricks 20
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
    load_mesh_from_file,
    write_sim_metadata,
)

# Standard Lego brick: 31.8mm x 15.8mm x 11.4mm (with studs ~9.6mm without)
BRICK_HX = 0.016
BRICK_HY = 0.008
BRICK_HZ = 0.005
BRICK_DENSITY = 1050.0  # ABS plastic kg/m3


def build_lego_scene(
    config: dict,
    seed: int = 0,
    num_bricks: int = 15,
    drop_height: float = 1.0,
    stack_width: int = 3,
    solver_iters: int = 10,
):
    """Build lego drop scene: stacked bricks dropped from height."""
    rng = random.Random(seed)

    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    # Try to load real lego meshes
    lego_dir = Path(config.get("lego_model_dir", ""))
    lego_meshes = []
    if lego_dir.is_dir():
        for ext in ("*.glb", "*.obj", "*.stl"):
            lego_meshes.extend(lego_dir.glob(ext))
    if lego_meshes:
        print(f"  Found {len(lego_meshes)} lego model files")

    brick_cfg = newton.ModelBuilder.ShapeConfig()
    brick_cfg.density = BRICK_DENSITY
    brick_cfg.mu = 0.4
    brick_cfg.restitution = 0.3

    # Build a brick stack
    bricks_placed = 0
    rows = max(1, num_bricks // stack_width)

    for row in range(rows):
        for col in range(stack_width):
            if bricks_placed >= num_bricks:
                break

            # Offset alternate rows (brick-like staggering)
            x_offset = BRICK_HX if (row % 2 == 1) else 0.0
            px = col * BRICK_HX * 2.1 + x_offset - (stack_width * BRICK_HX)
            py = rng.uniform(-BRICK_HY, BRICK_HY) * 0.3  # slight random Y
            pz = drop_height + row * BRICK_HZ * 2.1 + BRICK_HZ

            # Small random rotation for realism
            angle = rng.uniform(-0.05, 0.05)
            q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)

            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(px, py, pz), q=q),
            )

            if lego_meshes and rng.random() < 0.3:
                # Occasionally use a real mesh
                mesh_path = rng.choice(lego_meshes)
                try:
                    verts, faces = load_mesh_from_file(mesh_path)
                    # Scale to brick size
                    extents = verts.max(axis=0) - verts.min(axis=0)
                    scale = min(BRICK_HX * 2 / max(extents[0], 0.001),
                                BRICK_HY * 2 / max(extents[1], 0.001),
                                BRICK_HZ * 2 / max(extents[2], 0.001))
                    verts *= scale
                    mesh = newton.Mesh(
                        vertices=wp.array(verts, dtype=wp.vec3),
                        indices=wp.array(faces.flatten(), dtype=wp.int32),
                    )
                    builder.add_shape_mesh(body=body, mesh=mesh, cfg=brick_cfg)
                except Exception:
                    builder.add_shape_box(body=body, hx=BRICK_HX, hy=BRICK_HY, hz=BRICK_HZ, cfg=brick_cfg)
            else:
                builder.add_shape_box(body=body, hx=BRICK_HX, hy=BRICK_HY, hz=BRICK_HZ, cfg=brick_cfg)

            bricks_placed += 1

    model = builder.finalize()
    solver = newton.solvers.SolverXPBD(model, iterations=solver_iters)

    scene_info = {
        "num_bricks": bricks_placed,
        "drop_height": drop_height,
        "stack_width": stack_width,
        "has_lego_meshes": bool(lego_meshes),
    }

    return model, solver, scene_info


def run_lego_sim(
    model,
    solver,
    num_frames: int = 200,
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
    parser = argparse.ArgumentParser(description="Newton Lego Drop Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-bricks", type=int, default=15)
    parser.add_argument("--drop-height", type=float, default=1.0)
    parser.add_argument("--stack-width", type=int, default=3)
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
        sim_name = f"lego_s{args.seed:06d}"
        sim_dir = output_dir / "lego" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Lego Drop (seed={args.seed}) ===")

        model, solver, scene_info = build_lego_scene(
            config=config,
            seed=args.seed,
            num_bricks=args.num_bricks,
            drop_height=args.drop_height,
            stack_width=args.stack_width,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_lego_sim(
            model, solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="lego",
            seed=args.seed,
            **scene_info,
        )

        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
