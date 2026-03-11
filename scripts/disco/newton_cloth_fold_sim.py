"""
Newton Cloth Fold Simulation
===============================
Folds a cloth sheet by animating opposite edges toward each other.

Usage:
    uv run python scripts/disco/newton_cloth_fold_sim.py
    uv run python scripts/disco/newton_cloth_fold_sim.py --seed 42 --preset silk --fold-axis y
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import warp as wp

import newton

from newton_sim_utils import (
    CLOTH_PRESETS,
    TEST_FRAMES,
    add_ground_plane,
    add_table,
    create_blend_file,
    create_viewer,
    load_config,
    write_sim_metadata,
)


@wp.kernel
def _fold_particles(
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    fold_center: float,
    fold_axis: int,
    fold_velocity: float,
    dt: float,
):
    """Move pinned particles inward along fold_axis (0=x, 1=y)."""
    i = wp.tid()
    flags = particle_flags[i]
    if flags == 0:
        q = particle_q[i]
        if fold_axis == 0:
            sign = wp.sign(q[0] - fold_center)
            particle_q[i] = wp.vec3(q[0] - sign * fold_velocity * dt, q[1], q[2])
        else:
            sign = wp.sign(q[1] - fold_center)
            particle_q[i] = wp.vec3(q[0], q[1] - sign * fold_velocity * dt, q[2])


def build_cloth_fold_scene(
    config: dict,
    seed: int = 0,
    preset: str = "cotton",
    grid_res: int = 40,
    cloth_size: float = 0.8,
    table_height: float = 0.4,
    fold_axis: str = "x",
    solver_type: str = "vbd",
    solver_iters: int = 10,
):
    """Build fold scene: cloth on table, opposite edges pinned."""
    cloth_params = CLOTH_PRESETS.get(preset, CLOTH_PRESETS["cotton"])

    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    table_pos = (0.0, 0.0, table_height)
    table_half = (0.6, 0.6, 0.02)
    add_table(builder, position=table_pos, size=table_half)

    table_top_z = table_height + table_half[2]
    cell = cloth_size / grid_res
    cloth_x0 = -(cloth_size / 2)
    cloth_y0 = -(cloth_size / 2)
    cloth_z = table_top_z + 0.01

    # Pin opposite edges based on fold axis
    fix_left = fold_axis == "x"
    fix_right = fold_axis == "x"
    fix_top = fold_axis == "y"
    fix_bottom = fold_axis == "y"

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
        fix_left=fix_left,
        fix_right=fix_right,
        fix_top=fix_top,
        fix_bottom=fix_bottom,
        particle_radius=0.005,
    )

    if solver_type == "vbd":
        builder.color(include_bending=True)

    model = builder.finalize()
    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 1.0e0
    model.soft_contact_mu = 0.5

    if solver_type == "vbd":
        solver = newton.solvers.SolverVBD(
            model,
            iterations=solver_iters,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.01,
            particle_self_contact_margin=0.02,
        )
    else:
        solver = newton.solvers.SolverXPBD(model, iterations=solver_iters)

    scene_info = {
        "preset": preset,
        "fold_axis": fold_axis,
        "grid_res": grid_res,
        "cloth_size": cloth_size,
        "table_height": table_height,
    }

    return model, solver, scene_info


def run_cloth_fold(
    model,
    solver,
    num_frames: int = 250,
    substeps: int = 10,
    dt: float = 1.0 / 60.0,
    fold_speed: float = 0.2,
    fold_axis: str = "x",
    fold_frames: int = 120,
    viewer=None,
):
    """Run fold simulation. Pinned edges move inward for fold_frames, then settle."""
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()


    sub_dt = dt / substeps
    axis_idx = 0 if fold_axis == "x" else 1

    if viewer:
        viewer.set_model(model)
        viewer.begin_frame(0.0)
        viewer.log_state(state_0)
        viewer.end_frame()

    for frame in range(num_frames):
        for _ in range(substeps):
            # Animate fold during fold phase
            if frame < fold_frames and model.particle_count > 0:
                wp.launch(
                    _fold_particles,
                    dim=model.particle_count,
                    inputs=[
                        state_0.particle_q,
                        model.particle_flags,
                        0.0,  # fold center
                        axis_idx,
                        fold_speed,
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
    parser = argparse.ArgumentParser(description="Newton Cloth Fold Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", type=str, default="cotton",
                        choices=list(CLOTH_PRESETS.keys()))
    parser.add_argument("--grid-res", type=int, default=40)
    parser.add_argument("--cloth-size", type=float, default=0.8)
    parser.add_argument("--fold-axis", type=str, default="x", choices=["x", "y"])
    parser.add_argument("--fold-speed", type=float, default=0.2)
    parser.add_argument("--fold-frames", type=int, default=120)
    parser.add_argument("--num-frames", type=int, default=250)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--solver", type=str, default="vbd", choices=["vbd", "xpbd"])
    parser.add_argument("--solver-iters", type=int, default=10)
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
        sim_name = f"cloth_fold_s{args.seed:06d}"
        sim_dir = output_dir / "cloth_fold" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Cloth Fold (seed={args.seed}) ===")

        model, solver, scene_info = build_cloth_fold_scene(
            config=config,
            seed=args.seed,
            preset=args.preset,
            grid_res=args.grid_res,
            cloth_size=args.cloth_size,
            table_height=args.table_height,
            fold_axis=args.fold_axis,
            solver_type=args.solver,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_cloth_fold(
            model, solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            fold_speed=args.fold_speed,
            fold_axis=args.fold_axis,
            fold_frames=args.fold_frames,
            viewer=viewer,
        )

        particle_q = final_state.particle_q.numpy()
        bbox_size = np.linalg.norm(np.max(particle_q, axis=0) - np.min(particle_q, axis=0))
        print(f"  Final cloth bbox size: {bbox_size:.3f}")

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="cloth_fold",
            seed=args.seed,
            solver=args.solver,
            fold_speed=args.fold_speed,
            fold_frames=args.fold_frames,
            bbox_size=float(bbox_size),
            **scene_info,
        )

        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
