"""
Newton Rope / Cable Simulation
=================================
Simulates a rope (chain of particles with edge constraints) hanging or
draping over DTC props.

Uses a 1D cloth grid (dim_y=1) as the rope representation.

Usage:
    uv run python scripts/disco/newton_rope_sim.py
    uv run python scripts/disco/newton_rope_sim.py --seed 42 --rope-length 2.0 --num-props 3
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import warp as wp
from newton_sim_utils import (
    TEST_FRAMES,
    add_dtc_prop_as_rigid_body,
    add_ground_plane,
    add_table,
    create_blend_file,
    create_viewer,
    discover_dtc_props,
    load_config,
    write_sim_metadata,
)

import newton


def build_rope_scene(
    config: dict,
    seed: int = 0,
    rope_length: float = 1.5,
    rope_segments: int = 60,
    rope_density: float = 0.5,
    num_props: int = 2,
    table_height: float = 0.4,
    hang_height: float = 1.2,
    fix_ends: str = "left",
    solver_type: str = "vbd",
    solver_iters: int = 10,
):
    """Build rope scene: 1D cloth grid as rope, optionally over props.

    fix_ends: "left", "right", "both", "none"
    """
    rng = random.Random(seed)

    dtc_props = discover_dtc_props(
        config.get("dtc_dir", ""),
        unique_only=config.get("dtc_unique_only", True),
    )

    selected_props = []
    if dtc_props and num_props > 0:
        n = min(num_props, len(dtc_props))
        selected_props = rng.sample(dtc_props, n)

    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    table_half = (0.5, 0.5, 0.02)
    add_table(builder, position=(0.0, 0.0, table_height), size=table_half)
    table_top_z = table_height + table_half[2]

    # Place props
    prop_names = []
    for i, prop in enumerate(selected_props):
        meta = prop["metadata"]
        dims = meta.get("dimensions_m", [0.1, 0.1, 0.1])
        prop_height = max(dims) / 2.0
        px = rng.uniform(-0.3, 0.3)
        py = 0.0
        pz = table_top_z + prop_height + 0.01

        add_dtc_prop_as_rigid_body(builder, prop, position=(px, py, pz), static=True)
        prop_names.append(prop["name"])

    # Rope as 1D cloth grid
    cell = rope_length / rope_segments
    rope_z = table_top_z + hang_height

    builder.add_cloth_grid(
        pos=wp.vec3(-rope_length / 2, 0.0, rope_z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=rope_segments,
        dim_y=1,
        cell_x=cell,
        cell_y=cell,
        mass=rope_density * cell,
        tri_ke=1e4,
        tri_ka=1e4,
        tri_kd=1e-1,
        edge_ke=1e-1,
        edge_kd=1e-2,
        fix_left=fix_ends in ("left", "both"),
        fix_right=fix_ends in ("right", "both"),
        particle_radius=cell / 2,
    )

    if solver_type == "vbd":
        builder.color(include_bending=True)

    model = builder.finalize()
    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 1.0e0
    model.soft_contact_mu = 0.6

    if solver_type == "vbd":
        solver = newton.solvers.SolverVBD(
            model,
            iterations=solver_iters,
            particle_enable_self_contact=True,
            particle_self_contact_radius=cell,
            particle_self_contact_margin=cell * 1.5,
        )
    else:
        solver = newton.solvers.SolverXPBD(model, iterations=solver_iters)

    scene_info = {
        "rope_length": rope_length,
        "rope_segments": rope_segments,
        "fix_ends": fix_ends,
        "props": prop_names,
        "table_height": table_height,
    }

    return model, solver, scene_info


def run_rope_sim(
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

        if (frame + 1) % 50 == 0:
            print(f"  Frame {frame + 1}/{num_frames}")

    if viewer:
        viewer.close()

    return state_0


def main():
    parser = argparse.ArgumentParser(description="Newton Rope Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rope-length", type=float, default=1.5)
    parser.add_argument("--rope-segments", type=int, default=60)
    parser.add_argument("--fix-ends", type=str, default="left", choices=["left", "right", "both", "none"])
    parser.add_argument("--num-props", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--solver", type=str, default="vbd", choices=["vbd", "xpbd"])
    parser.add_argument("--solver-iters", type=int, default=10)
    parser.add_argument("--table-height", type=float, default=0.4)
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
        sim_name = f"rope_s{args.seed:06d}"
        sim_dir = output_dir / "rope" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Rope (seed={args.seed}) ===")

        model, solver, scene_info = build_rope_scene(
            config=config,
            seed=args.seed,
            rope_length=args.rope_length,
            rope_segments=args.rope_segments,
            fix_ends=args.fix_ends,
            num_props=args.num_props,
            table_height=args.table_height,
            solver_type=args.solver,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_rope_sim(
            model,
            solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        particle_q = final_state.particle_q.numpy()
        bbox_size = np.linalg.norm(np.max(particle_q, axis=0) - np.min(particle_q, axis=0))

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="rope",
            seed=args.seed,
            solver=args.solver,
            bbox_size=float(bbox_size),
            **scene_info,
        )

        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
