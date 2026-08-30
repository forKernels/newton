"""
Newton Cable Pile Simulation
================================
Drops layered cables (alternating X/Y orientation) that settle into a tangled pile.
Great for wire/cable/spaghetti-type data.

Based on Newton's cable_pile example — uses articulated bodies with capsule shapes.

Usage:
    uv run python scripts/disco/newton_cable_pile_sim.py
    uv run python scripts/disco/newton_cable_pile_sim.py --test
    uv run python scripts/disco/newton_cable_pile_sim.py --seed 42 --num-cables 20
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
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


def _add_cable(
    builder: newton.ModelBuilder,
    start: tuple[float, float, float],
    direction: tuple[float, float, float],
    num_elements: int = 40,
    segment_length: float = 0.05,
    cable_radius: float = 0.012,
    waviness: float = 0.5,
    cfg=None,
):
    """Add a single cable as a chain of capsule bodies connected by ball joints."""
    dir_vec = np.array(direction, dtype=np.float32)
    dir_vec /= np.linalg.norm(dir_vec)

    # Orthogonal vector for waviness
    if abs(dir_vec[0]) > 0.9:
        ortho = np.array([0.0, 1.0, 0.0])
    else:
        ortho = np.array([1.0, 0.0, 0.0])
    ortho = ortho - ortho.dot(dir_vec) * dir_vec
    ortho /= np.linalg.norm(ortho)

    bodies = []
    parent = -1
    half_height = segment_length / 2.0 - cable_radius

    for i in range(num_elements):
        t = i * segment_length
        wave = waviness * math.sin(2.0 * math.pi * t / (num_elements * segment_length) * 2.0)

        pos = np.array(start) + dir_vec * t + ortho * wave
        xform = wp.transform(p=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), q=wp.quat_identity())

        body = builder.add_body(xform=xform)

        if cfg:
            builder.add_shape_capsule(body=body, radius=cable_radius, half_height=max(half_height, 0.001), cfg=cfg)
        else:
            builder.add_shape_capsule(body=body, radius=cable_radius, half_height=max(half_height, 0.001))

        # Connect to previous body with a ball joint
        if parent >= 0:
            builder.add_joint_ball(
                parent_body=parent,
                child_body=body,
                parent_xform=wp.transform(
                    p=wp.vec3(0.0, 0.0, segment_length / 2.0),
                    q=wp.quat_identity(),
                ),
                child_xform=wp.transform(
                    p=wp.vec3(0.0, 0.0, -segment_length / 2.0),
                    q=wp.quat_identity(),
                ),
            )

        bodies.append(body)
        parent = body

    return bodies


def build_cable_pile_scene(
    seed: int = 0,
    num_cables: int = 20,
    num_elements: int = 30,
    cable_radius: float = 0.012,
    layers: int = 3,
    solver_iters: int = 10,
):
    """Build cable pile: layered cables with alternating orientations."""
    rng = random.Random(seed)

    builder = newton.ModelBuilder()
    builder.rigid_contact_margin = 0.05

    builder.default_shape_cfg.ke = 1.0e6
    builder.default_shape_cfg.kd = 1.0e-1
    builder.default_shape_cfg.mu = 3.0

    add_ground_plane(builder)

    segment_length = 0.05
    cable_length = num_elements * segment_length
    lane_spacing = max(8.0 * cable_radius, 0.12)
    layer_gap = cable_radius * 6.0
    cables_per_layer = max(1, num_cables // layers)

    cable_cfg = newton.ModelBuilder.ShapeConfig()
    cable_cfg.ke = builder.default_shape_cfg.ke
    cable_cfg.kd = builder.default_shape_cfg.kd
    cable_cfg.mu = builder.default_shape_cfg.mu

    total_cables = 0
    for layer in range(layers):
        orient = "x" if layer % 2 == 0 else "y"
        z0 = 0.3 + layer * layer_gap

        for lane in range(cables_per_layer):
            if total_cables >= num_cables:
                break

            offset = (lane - (cables_per_layer - 1) * 0.5) * lane_spacing
            wav = rng.uniform(0.2, 0.8)

            if orient == "x":
                start = (-cable_length / 2, offset, z0)
                direction = (1.0, 0.0, 0.0)
            else:
                start = (offset, -cable_length / 2, z0)
                direction = (0.0, 1.0, 0.0)

            _add_cable(
                builder,
                start,
                direction,
                num_elements=num_elements,
                segment_length=segment_length,
                cable_radius=cable_radius,
                waviness=wav,
                cfg=cable_cfg,
            )
            total_cables += 1

    print(f"  Cables: {total_cables}, bodies: {builder.body_count}")

    model = builder.finalize()
    solver = newton.solvers.SolverXPBD(model, iterations=solver_iters)

    scene_info = {
        "num_cables": total_cables,
        "num_elements": num_elements,
        "cable_radius": cable_radius,
        "layers": layers,
    }

    return model, solver, scene_info


def run_cable_pile_sim(
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

        if (frame + 1) % 50 == 0:
            print(f"  Frame {frame + 1}/{num_frames}")

    if viewer:
        viewer.close()

    return state_0


def main():
    parser = argparse.ArgumentParser(description="Newton Cable Pile Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-cables", type=int, default=20)
    parser.add_argument("--num-elements", type=int, default=30)
    parser.add_argument("--cable-radius", type=float, default=0.012)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--num-frames", type=int, default=200)
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
        sim_name = f"cable_pile_s{args.seed:06d}"
        sim_dir = output_dir / "cable_pile" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Cable Pile (seed={args.seed}) ===")

        model, solver, scene_info = build_cable_pile_scene(
            seed=args.seed,
            num_cables=args.num_cables,
            num_elements=args.num_elements,
            cable_radius=args.cable_radius,
            layers=args.layers,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_cable_pile_sim(
            model,
            solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="cable_pile",
            seed=args.seed,
            **scene_info,
        )

        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
