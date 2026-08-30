"""
Newton Cloth Pull Simulation
================================
Pulls a cloth off a table by animating one pinned edge away from the surface.
The cloth starts flat on the table and is dragged off.

Usage:
    uv run python scripts/disco/newton_cloth_pull_sim.py
    uv run python scripts/disco/newton_cloth_pull_sim.py --test
    uv run python scripts/disco/newton_cloth_pull_sim.py --seed 42 --pull-speed 0.5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import warp as wp
from newton_sim_utils import (
    CLOTH_PRESETS,
    TEST_FRAMES,
    add_dtc_prop_as_rigid_body,
    add_ground_plane,
    add_table,
    build_solver,
    create_blend_file,
    create_viewer,
    discover_dtc_props,
    finalize_cloth_model,
    load_config,
    write_sim_metadata,
)

import newton


@wp.kernel
def _pull_pinned_particles(
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    pull_velocity: wp.vec3,
    dt: float,
):
    """Move pinned particles (flag == 0) by pull_velocity * dt."""
    i = wp.tid()
    flags = particle_flags[i]
    if flags == 0:
        q = particle_q[i]
        particle_q[i] = q + pull_velocity * dt


def build_cloth_pull_scene(
    config: dict,
    seed: int = 0,
    preset: str = "cotton",
    num_props: int = 2,
    grid_res: int = 40,
    cloth_size: float = 0.8,
    table_height: float = 0.4,
    solver_type: str = "vbd",
    solver_iters: int = 10,
):
    """Build pull scene: cloth on table, left edge pinned, pulled in -X direction."""
    rng = random.Random(seed)

    dtc_props = discover_dtc_props(
        config.get("dtc_dir", ""),
        unique_only=config.get("dtc_unique_only", True),
    )

    selected_props = []
    if dtc_props and num_props > 0:
        n = min(num_props, len(dtc_props))
        selected_props = rng.sample(dtc_props, n)

    cloth_params = CLOTH_PRESETS.get(preset, CLOTH_PRESETS["cotton"])

    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    table_half = (0.5, 0.5, 0.02)
    add_table(builder, position=(0.0, 0.0, table_height), size=table_half)
    table_top_z = table_height + table_half[2]

    # Place props on table (cloth will be pulled off over them)
    prop_names = []
    for i, prop in enumerate(selected_props):
        meta = prop["metadata"]
        dims = meta.get("dimensions_m", [0.1, 0.1, 0.1])
        prop_height = max(dims) / 2.0
        px = rng.uniform(-0.1, 0.3)
        py = rng.uniform(-0.2, 0.2)
        pz = table_top_z + prop_height + 0.01

        add_dtc_prop_as_rigid_body(builder, prop, position=(px, py, pz), static=True)
        prop_names.append(prop["name"])
        print(f"  Prop {i}: {prop['name']}")

    # Cloth starts flat on table, left edge pinned
    cell = cloth_size / grid_res
    cloth_z = table_top_z + 0.01

    builder.add_cloth_grid(
        pos=wp.vec3(-(cloth_size / 2), -(cloth_size / 2), cloth_z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=grid_res,
        dim_y=grid_res,
        cell_x=cell,
        cell_y=cell,
        mass=cloth_params["density"] * cell * cell,
        tri_ke=cloth_params["tri_ke"],
        tri_ka=cloth_params["tri_ka"],
        tri_kd=cloth_params["tri_kd"],
        edge_ke=cloth_params["edge_ke"],
        edge_kd=cloth_params["edge_kd"],
        fix_left=True,
        particle_radius=0.005,
    )

    if solver_type == "vbd":
        builder.color()

    model = finalize_cloth_model(builder)
    solver = build_solver(model, solver_type=solver_type, iterations=solver_iters)

    scene_info = {
        "preset": preset,
        "grid_res": grid_res,
        "cloth_size": cloth_size,
        "props": prop_names,
        "table_height": table_height,
    }

    return model, solver, scene_info


def run_cloth_pull(
    model,
    solver,
    num_frames: int = 300,
    substeps: int = 10,
    dt: float = 1.0 / 60.0,
    pull_speed: float = 0.4,
    pull_frames: int = 180,
    viewer=None,
):
    """Run pull simulation. Pinned edge moves in -X for pull_frames, then settles."""
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.01)
    contacts = collision_pipeline.contacts()

    sub_dt = dt / substeps

    # Pull in -X direction (away from table)
    pull_vel = wp.vec3(-pull_speed, 0.0, 0.0)

    if viewer:
        viewer.set_model(model)
        viewer.begin_frame(0.0)
        viewer.log_state(state_0)
        viewer.end_frame()

    for frame in range(num_frames):
        for _ in range(substeps):
            # Animate pinned edge during pull phase
            if frame < pull_frames and model.particle_count > 0:
                wp.launch(
                    _pull_pinned_particles,
                    dim=model.particle_count,
                    inputs=[
                        state_0.particle_q,
                        model.particle_flags,
                        pull_vel,
                        sub_dt,
                    ],
                )

            state_0.clear_forces()
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
    parser = argparse.ArgumentParser(description="Newton Cloth Pull Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", type=str, default="cotton", choices=list(CLOTH_PRESETS.keys()))
    parser.add_argument("--num-props", type=int, default=2)
    parser.add_argument("--grid-res", type=int, default=40)
    parser.add_argument("--cloth-size", type=float, default=0.8)
    parser.add_argument("--pull-speed", type=float, default=0.4, help="Pull speed [m/s]")
    parser.add_argument("--pull-frames", type=int, default=180, help="Frames to pull before settling")
    parser.add_argument("--num-frames", type=int, default=300)
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
        sim_name = f"cloth_pull_s{args.seed:06d}"
        sim_dir = output_dir / "cloth_pull" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Cloth Pull (seed={args.seed}) ===")

        model, solver, scene_info = build_cloth_pull_scene(
            config=config,
            seed=args.seed,
            preset=args.preset,
            num_props=args.num_props,
            grid_res=args.grid_res,
            cloth_size=args.cloth_size,
            table_height=args.table_height,
            solver_type=args.solver,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_cloth_pull(
            model,
            solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            pull_speed=args.pull_speed,
            pull_frames=args.pull_frames,
            viewer=viewer,
        )

        particle_q = final_state.particle_q.numpy()
        bbox_size = np.linalg.norm(np.max(particle_q, axis=0) - np.min(particle_q, axis=0))

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="cloth_pull",
            seed=args.seed,
            solver=args.solver,
            pull_speed=args.pull_speed,
            pull_frames=args.pull_frames,
            bbox_size=float(bbox_size),
            **scene_info,
        )

        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
