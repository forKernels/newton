"""
Newton Cloth Drag Simulation
===============================
Drags a cloth sheet across a table with DTC props.
One edge of the cloth is pinned and animated horizontally.

Usage:
    uv run python scripts/disco/newton_cloth_drag_sim.py
    uv run python scripts/disco/newton_cloth_drag_sim.py --seed 42 --preset denim --num-props 4
    uv run python scripts/disco/newton_cloth_drag_sim.py --viewer usd --output-dir ./output
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
    CLOTH_PRESETS,
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


@wp.kernel
def _animate_pinned_particles(
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    drag_velocity: wp.vec3,
    dt: float,
):
    """Move pinned particles (flag bit 1 set) by drag_velocity * dt."""
    i = wp.tid()
    flags = particle_flags[i]
    # newton uses PARTICLE_FLAG_ACTIVE = 1; pinned particles have flag == 0
    if flags == 0:
        q = particle_q[i]
        particle_q[i] = q + drag_velocity * dt


def build_cloth_drag_scene(
    config: dict,
    seed: int = 0,
    preset: str = "cotton",
    num_props: int = 3,
    grid_res: int = 40,
    cloth_size: float = 0.8,
    table_height: float = 0.4,
    solver_type: str = "vbd",
    solver_iters: int = 10,
):
    """Build a cloth drag scene: cloth on table, edge pinned, dragged across DTC props.

    Returns (model, solver, scene_info, pinned_mask).
    """
    rng = random.Random(seed)

    dtc_props = discover_dtc_props(
        config.get("dtc_dir", ""),
        unique_only=config.get("dtc_unique_only", True),
    )
    print(f"Found {len(dtc_props)} DTC props")

    selected_props = []
    if dtc_props and num_props > 0:
        n = min(num_props, len(dtc_props))
        selected_props = rng.sample(dtc_props, n)

    cloth_params = CLOTH_PRESETS.get(preset, CLOTH_PRESETS["cotton"])

    # --- Build scene ---
    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    table_pos = (0.0, 0.0, table_height)
    table_half = (0.6, 0.6, 0.02)
    add_table(builder, position=table_pos, size=table_half)

    # Place DTC props on table
    table_top_z = table_height + table_half[2]
    prop_info_list = []
    for i, prop in enumerate(selected_props):
        meta = prop["metadata"]
        dims = meta.get("dimensions_m", [0.1, 0.1, 0.1])
        prop_height = max(dims) / 2.0

        # Scatter in right half of table (cloth drags from left to right)
        px = rng.uniform(0.0, 0.35)
        py = rng.uniform(-0.25, 0.25)
        pz = table_top_z + prop_height + 0.01

        body_idx = add_dtc_prop_as_rigid_body(
            builder, prop,
            position=(px, py, pz),
            static=True,  # props are fixed for drag
        )
        prop_info_list.append({"name": prop["name"], "position": (px, py, pz)})
        print(f"  Prop {i}: {prop['name']} at ({px:.2f}, {py:.2f}, {pz:.2f})")

    # Add cloth grid lying on table, left edge pinned
    cell = cloth_size / grid_res
    cloth_x0 = -(cloth_size / 2)
    cloth_y0 = -(cloth_size / 2)
    cloth_z = table_top_z + 0.01  # just above table

    builder.add_cloth_grid(
        pos=wp.vec3(cloth_x0, cloth_y0, cloth_z),
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
            particle_self_contact_radius=0.01,
            particle_self_contact_margin=0.02,
        )
    elif solver_type == "xpbd":
        solver = newton.solvers.SolverXPBD(model, iterations=solver_iters)
    else:
        raise ValueError(f"Unsupported solver for cloth drag: {solver_type}")

    scene_info = {
        "preset": preset,
        "cloth_params": cloth_params,
        "grid_res": grid_res,
        "cloth_size": cloth_size,
        "props": [p["name"] for p in selected_props],
        "table_height": table_height,
    }

    return model, solver, scene_info


def run_cloth_drag(
    model,
    solver,
    num_frames: int = 300,
    substeps: int = 10,
    dt: float = 1.0 / 60.0,
    drag_speed: float = 0.3,
    viewer=None,
):
    """Run the cloth drag simulation.

    The pinned (left) edge is animated rightward at drag_speed [m/s].
    Returns final state.
    """
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()


    sub_dt = dt / substeps
    drag_vel = wp.vec3(drag_speed, 0.0, 0.0)

    if viewer:
        viewer.set_model(model)
        viewer.begin_frame(0.0)
        viewer.log_state(state_0)
        viewer.end_frame()

    for frame in range(num_frames):
        for _ in range(substeps):
            # Animate pinned particles
            if model.particle_count > 0:
                wp.launch(
                    _animate_pinned_particles,
                    dim=model.particle_count,
                    inputs=[
                        state_0.particle_q,
                        model.particle_flags,
                        drag_vel,
                        sub_dt,
                    ],
                )

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
    parser = argparse.ArgumentParser(description="Newton Cloth Drag Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", type=str, default="cotton",
                        choices=list(CLOTH_PRESETS.keys()))
    parser.add_argument("--num-props", type=int, default=3)
    parser.add_argument("--grid-res", type=int, default=40)
    parser.add_argument("--cloth-size", type=float, default=0.8)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--solver", type=str, default="vbd", choices=["vbd", "xpbd"])
    parser.add_argument("--solver-iters", type=int, default=10)
    parser.add_argument("--drag-speed", type=float, default=0.3,
                        help="Drag speed [m/s]")
    parser.add_argument("--table-height", type=float, default=0.4)
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
        sim_name = f"cloth_drag_s{args.seed:06d}"
        sim_dir = output_dir / "cloth_drag" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Cloth Drag (seed={args.seed}) ===")
        print(f"  Device: {device}")
        print(f"  Output: {sim_dir}")

        model, solver, scene_info = build_cloth_drag_scene(
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

        final_state = run_cloth_drag(
            model, solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            drag_speed=args.drag_speed,
            viewer=viewer,
        )

        particle_q = final_state.particle_q.numpy()
        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)
        print(f"  Final cloth bbox: size={bbox_size:.3f}")

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="cloth_drag",
            seed=args.seed,
            solver=args.solver,
            num_frames=args.num_frames,
            drag_speed=args.drag_speed,
            bbox_size=float(bbox_size),
            **scene_info,
        )

        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
