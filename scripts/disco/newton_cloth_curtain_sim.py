"""
Newton Cloth Curtain / Wind Simulation
=========================================
Simulates a hanging cloth (curtain) with wind forces.

Usage:
    uv run python scripts/disco/newton_cloth_curtain_sim.py
    uv run python scripts/disco/newton_cloth_curtain_sim.py --seed 42 --wind-strength 5.0
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
    add_ground_plane,
    create_blend_file,
    create_viewer,
    load_config,
    write_sim_metadata,
)


@wp.kernel
def _apply_wind_force(
    particle_f: wp.array(dtype=wp.vec3),
    wind_dir: wp.vec3,
    strength: float,
    time: float,
):
    """Apply oscillating wind force to all particles."""
    i = wp.tid()
    # Oscillate strength with time for natural wind gusts
    gust = strength * (0.5 + 0.5 * wp.sin(time * 3.0 + float(i) * 0.01))
    f = particle_f[i]
    particle_f[i] = f + wind_dir * gust


def build_curtain_scene(
    config: dict,
    seed: int = 0,
    preset: str = "silk",
    grid_res: int = 40,
    curtain_width: float = 1.5,
    curtain_height: float = 1.2,
    hang_height: float = 2.0,
    solver_type: str = "vbd",
    solver_iters: int = 10,
):
    """Build curtain scene: cloth pinned at top, hanging, with wind."""
    cloth_params = CLOTH_PRESETS.get(preset, CLOTH_PRESETS["silk"])

    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    cell_x = curtain_width / grid_res
    cell_y = curtain_height / grid_res

    # Curtain hangs from top edge (fix_top), oriented vertically
    # Rotate so Y-axis of grid points downward (cloth hangs in -Z with grid rotated)
    builder.add_cloth_grid(
        pos=wp.vec3(-curtain_width / 2, 0.0, hang_height),
        rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.pi * 0.5),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=grid_res,
        dim_y=int(grid_res * curtain_height / curtain_width),
        cell_x=cell_x,
        cell_y=cell_y,
        mass=cloth_params["density"] * cell_x * cell_y,
        tri_ke=cloth_params["tri_ke"],
        tri_ka=cloth_params["tri_ka"],
        tri_kd=cloth_params["tri_kd"],
        edge_ke=cloth_params["edge_ke"],
        edge_kd=cloth_params["edge_kd"],
        fix_top=True,
        particle_radius=0.005,
    )

    if solver_type == "vbd":
        builder.color(include_bending=True)

    model = builder.finalize()
    model.soft_contact_ke = 1.0e2
    model.soft_contact_kd = 1.0e0
    model.soft_contact_mu = 0.5

    if solver_type == "vbd":
        solver = newton.solvers.SolverVBD(
            model, iterations=solver_iters,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.01,
            particle_self_contact_margin=0.02,
        )
    else:
        solver = newton.solvers.SolverXPBD(model, iterations=solver_iters)

    scene_info = {
        "preset": preset,
        "grid_res": grid_res,
        "curtain_width": curtain_width,
        "curtain_height": curtain_height,
        "hang_height": hang_height,
    }

    return model, solver, scene_info


def run_curtain_sim(
    model,
    solver,
    num_frames: int = 300,
    substeps: int = 10,
    dt: float = 1.0 / 60.0,
    wind_strength: float = 3.0,
    wind_angle: float = 0.0,
    viewer=None,
):
    """Run curtain simulation with wind. Returns final state."""
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()


    sub_dt = dt / substeps
    wind_dir = wp.vec3(math.cos(wind_angle), math.sin(wind_angle), 0.0)
    sim_time = 0.0

    if viewer:
        viewer.set_model(model)
        viewer.begin_frame(0.0)
        viewer.log_state(state_0)
        viewer.end_frame()

    for frame in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()

            # Apply wind
            if model.particle_count > 0:
                wp.launch(
                    _apply_wind_force,
                    dim=model.particle_count,
                    inputs=[
                        state_0.particle_f,
                        wind_dir,
                        wind_strength,
                        sim_time,
                    ],
                )

            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0
            sim_time += sub_dt

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
    parser = argparse.ArgumentParser(description="Newton Cloth Curtain / Wind Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", type=str, default="silk",
                        choices=list(CLOTH_PRESETS.keys()))
    parser.add_argument("--grid-res", type=int, default=40)
    parser.add_argument("--curtain-width", type=float, default=1.5)
    parser.add_argument("--curtain-height", type=float, default=1.2)
    parser.add_argument("--hang-height", type=float, default=2.0)
    parser.add_argument("--wind-strength", type=float, default=3.0)
    parser.add_argument("--wind-angle", type=float, default=0.0,
                        help="Wind direction [radians]")
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--solver", type=str, default="vbd", choices=["vbd", "xpbd"])
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
        sim_name = f"cloth_curtain_s{args.seed:06d}"
        sim_dir = output_dir / "cloth_curtain" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Cloth Curtain (seed={args.seed}) ===")

        model, solver, scene_info = build_curtain_scene(
            config=config,
            seed=args.seed,
            preset=args.preset,
            grid_res=args.grid_res,
            curtain_width=args.curtain_width,
            curtain_height=args.curtain_height,
            hang_height=args.hang_height,
            solver_type=args.solver,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_curtain_sim(
            model, solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            wind_strength=args.wind_strength,
            wind_angle=args.wind_angle,
            viewer=viewer,
        )

        particle_q = final_state.particle_q.numpy()
        bbox_size = np.linalg.norm(np.max(particle_q, axis=0) - np.min(particle_q, axis=0))

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="cloth_curtain",
            seed=args.seed,
            solver=args.solver,
            wind_strength=args.wind_strength,
            wind_angle=args.wind_angle,
            bbox_size=float(bbox_size),
            **scene_info,
        )

        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
