"""
Newton Cloth Lay Simulation
===============================
Lays a cloth flat just above a surface with DTC props and lets it settle
under gravity. No pinning, no velocity — pure drape and settle.

Usage:
    uv run python scripts/disco/newton_cloth_lay_sim.py
    uv run python scripts/disco/newton_cloth_lay_sim.py --test
    uv run python scripts/disco/newton_cloth_lay_sim.py --seed 42 --preset silk --num-props 3
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
    add_dtc_prop_as_rigid_body,
    add_garment_as_cloth,
    add_ground_plane,
    add_table,
    create_blend_file,
    create_viewer,
    discover_dtc_props,
    discover_garments,
    load_config,
    write_sim_metadata,
)


def build_cloth_lay_scene(
    config: dict,
    seed: int = 0,
    preset: str = "cotton",
    num_props: int = 2,
    use_grid: bool = False,
    grid_res: int = 40,
    garment_name: str | None = None,
    table_height: float = 0.4,
    hover_gap: float = 0.05,
    solver_type: str = "vbd",
    solver_iters: int = 10,
):
    """Build lay scene: cloth hovers just above surface, settles under gravity."""
    rng = random.Random(seed)

    dtc_props = discover_dtc_props(
        config.get("dtc_dir", ""),
        unique_only=config.get("dtc_unique_only", True),
    )
    garments = discover_garments(config.get("garment_dir", ""))

    selected_props = []
    if dtc_props and num_props > 0:
        n = min(num_props, len(dtc_props))
        selected_props = rng.sample(dtc_props, n)

    garment = None
    if not use_grid and garments:
        if garment_name:
            matches = [g for g in garments if g["name"] == garment_name]
            garment = matches[0] if matches else rng.choice(garments)
        else:
            garment = rng.choice(garments)

    cloth_params = CLOTH_PRESETS.get(preset, CLOTH_PRESETS["cotton"])

    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    table_half = (0.6, 0.6, 0.02)
    add_table(builder, position=(0.0, 0.0, table_height), size=table_half)
    table_top_z = table_height + table_half[2]

    # Place props on table
    prop_names = []
    for i, prop in enumerate(selected_props):
        meta = prop["metadata"]
        dims = meta.get("dimensions_m", [0.1, 0.1, 0.1])
        prop_height = max(dims) / 2.0
        px = rng.uniform(-0.3, 0.3)
        py = rng.uniform(-0.3, 0.3)
        pz = table_top_z + prop_height + 0.01

        add_dtc_prop_as_rigid_body(builder, prop, position=(px, py, pz), static=True)
        prop_names.append(prop["name"])
        print(f"  Prop {i}: {prop['name']}")

    # Cloth placed flat, just above table surface
    cloth_z = table_top_z + hover_gap

    if garment and not use_grid:
        print(f"  Garment: {garment['name']} ({garment['category']}) preset={preset}")
        add_garment_as_cloth(
            builder, garment,
            position=(0.0, 0.0, cloth_z),
            **cloth_params,
        )
    else:
        print(f"  Grid cloth: {grid_res}x{grid_res}, preset={preset}")
        size = 0.8
        cell = size / grid_res
        builder.add_cloth_grid(
            pos=wp.vec3(-(size / 2), -(size / 2), cloth_z),
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
            particle_radius=0.005,
        )

    if solver_type == "vbd":
        builder.color(include_bending=True)

    model = builder.finalize()
    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 1.0e0
    model.soft_contact_mu = 0.7

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
        "garment": garment["name"] if garment else f"grid_{grid_res}x{grid_res}",
        "garment_category": garment["category"] if garment else "procedural",
        "preset": preset,
        "props": prop_names,
        "hover_gap": hover_gap,
        "table_height": table_height,
    }

    return model, solver, scene_info


def run_cloth_lay(
    model,
    solver,
    num_frames: int = 150,
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
    parser = argparse.ArgumentParser(description="Newton Cloth Lay Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", type=str, default="cotton",
                        choices=list(CLOTH_PRESETS.keys()))
    parser.add_argument("--num-props", type=int, default=2)
    parser.add_argument("--use-grid", action="store_true")
    parser.add_argument("--grid-res", type=int, default=40)
    parser.add_argument("--garment", type=str, default=None)
    parser.add_argument("--hover-gap", type=float, default=0.05,
                        help="Gap above surface before settling [m]")
    parser.add_argument("--num-frames", type=int, default=150)
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
        sim_name = f"cloth_lay_s{args.seed:06d}"
        sim_dir = output_dir / "cloth_lay" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Cloth Lay (seed={args.seed}) ===")

        model, solver, scene_info = build_cloth_lay_scene(
            config=config,
            seed=args.seed,
            preset=args.preset,
            num_props=args.num_props,
            use_grid=args.use_grid,
            grid_res=args.grid_res,
            garment_name=args.garment,
            hover_gap=args.hover_gap,
            table_height=args.table_height,
            solver_type=args.solver,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_cloth_lay(
            model, solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        particle_q = final_state.particle_q.numpy()
        bbox_size = np.linalg.norm(np.max(particle_q, axis=0) - np.min(particle_q, axis=0))

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="cloth_lay",
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
