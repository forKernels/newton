"""
Newton Softbody Simulation
=============================
Drops tetrahedral soft bodies onto ground/table/props.

Usage:
    uv run python scripts/disco/newton_softbody_sim.py
    uv run python scripts/disco/newton_softbody_sim.py --seed 42 --num-softbodies 3
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


def build_softbody_scene(
    config: dict,
    seed: int = 0,
    num_softbodies: int = 2,
    num_props: int = 1,
    table_height: float = 0.4,
    drop_height_range: tuple[float, float] = (0.5, 1.5),
    stiffness: float = 1e4,
    solver_iters: int = 10,
):
    """Build softbody scene: tetrahedral grids dropping onto table + props."""
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

    # Place DTC props (static)
    prop_names = []
    for prop in selected_props:
        meta = prop["metadata"]
        dims = meta.get("dimensions_m", [0.1, 0.1, 0.1])
        px = rng.uniform(-0.2, 0.2)
        py = rng.uniform(-0.2, 0.2)
        pz = table_top_z + max(dims) / 2.0 + 0.01
        add_dtc_prop_as_rigid_body(builder, prop, position=(px, py, pz), static=True)
        prop_names.append(prop["name"])

    # Add soft bodies
    softbody_info = []
    for i in range(num_softbodies):
        dim = rng.choice([4, 5, 6])
        cell_size = rng.uniform(0.06, 0.12)
        px = rng.uniform(-0.2, 0.2)
        py = rng.uniform(-0.2, 0.2)
        pz = table_top_z + rng.uniform(*drop_height_range)

        builder.add_soft_grid(
            pos=wp.vec3(px, py, pz),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim,
            dim_y=dim,
            dim_z=max(2, dim // 2),
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=1.0e3,
            k_mu=stiffness,
            k_lambda=stiffness,
            k_damp=1e-3,
        )
        softbody_info.append(
            {
                "dim": dim,
                "cell_size": cell_size,
                "position": (px, py, pz),
            }
        )
        print(f"  Softbody {i}: {dim}x{dim} cells at z={pz:.2f}")

    builder.color()

    model = builder.finalize()
    model.soft_contact_ke = 1.0e4
    model.soft_contact_kd = 1e-3
    model.soft_contact_mu = 0.8

    solver = newton.solvers.SolverVBD(
        model,
        iterations=solver_iters,
        particle_enable_self_contact=True,
        particle_self_contact_radius=0.01,
        particle_self_contact_margin=0.02,
        particle_enable_tile_solve=True,
    )

    scene_info = {
        "num_softbodies": num_softbodies,
        "softbodies": softbody_info,
        "props": prop_names,
        "stiffness": stiffness,
        "table_height": table_height,
    }

    return model, solver, scene_info


def run_softbody_sim(
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
    parser = argparse.ArgumentParser(description="Newton Softbody Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-softbodies", type=int, default=2)
    parser.add_argument("--num-props", type=int, default=1)
    parser.add_argument("--stiffness", type=float, default=1e4)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--substeps", type=int, default=10)
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
        sim_name = f"softbody_s{args.seed:06d}"
        sim_dir = output_dir / "softbody" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Softbody (seed={args.seed}) ===")

        model, solver, scene_info = build_softbody_scene(
            config=config,
            seed=args.seed,
            num_softbodies=args.num_softbodies,
            num_props=args.num_props,
            stiffness=args.stiffness,
            table_height=args.table_height,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_softbody_sim(
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
            sim_type="softbody",
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
