"""
Newton Cloth Drop Simulation
==============================
Drops a Maria garment onto a table with DTC props.

Usage:
    uv run python scripts/disco/newton_cloth_drop_sim.py
    uv run python scripts/disco/newton_cloth_drop_sim.py --seed 42 --preset denim
    uv run python scripts/disco/newton_cloth_drop_sim.py --viewer usd --output-dir ./output
    uv run python scripts/disco/newton_cloth_drop_sim.py --use-grid --grid-res 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
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


def build_cloth_drop_scene(
    config: dict,
    seed: int = 0,
    preset: str = "cotton",
    num_props: int = 2,
    use_grid: bool = False,
    grid_res: int = 30,
    garment_name: str | None = None,
    table_height: float = 0.4,
    drop_height: float = 1.0,
    solver_type: str = "vbd",
    solver_iters: int = 10,
):
    """Build a cloth drop scene: garment drops onto table + DTC props.

    Returns (model, solver, scene_info dict).
    """
    rng = random.Random(seed)

    # Discover assets
    dtc_props = discover_dtc_props(
        config.get("dtc_dir", ""),
        unique_only=config.get("dtc_unique_only", True),
    )
    garments = discover_garments(config.get("garment_dir", ""))

    print(f"Found {len(dtc_props)} DTC props, {len(garments)} garments")

    # Pick garment
    garment = None
    if not use_grid and garments:
        if garment_name:
            matches = [g for g in garments if g["name"] == garment_name]
            garment = matches[0] if matches else rng.choice(garments)
        else:
            garment = rng.choice(garments)

    # Pick props
    selected_props = []
    if dtc_props and num_props > 0:
        n = min(num_props, len(dtc_props))
        selected_props = rng.sample(dtc_props, n)

    # Get cloth params from preset
    cloth_params = CLOTH_PRESETS.get(preset, CLOTH_PRESETS["cotton"])

    # --- Build scene ---
    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    # Table
    table_pos = (0.0, 0.0, table_height)
    table_half = (0.5, 0.5, 0.02)
    add_table(builder, position=table_pos, size=table_half)

    # Place DTC props on table
    prop_info_list = []
    table_top_z = table_height + table_half[2]
    for i, prop in enumerate(selected_props):
        meta = prop["metadata"]
        dims = meta.get("dimensions_m", [0.1, 0.1, 0.1])
        prop_height = max(dims) / 2.0

        # Scatter props on table surface
        px = rng.uniform(-0.25, 0.25)
        py = rng.uniform(-0.25, 0.25)
        pz = table_top_z + prop_height + 0.01

        body_idx = add_dtc_prop_as_rigid_body(
            builder, prop,
            position=(px, py, pz),
            static=False,
        )
        prop_info_list.append({
            "name": prop["name"],
            "body_idx": body_idx,
            "position": (px, py, pz),
        })
        print(f"  Prop {i}: {prop['name']} at ({px:.2f}, {py:.2f}, {pz:.2f})")

    # Add cloth (garment mesh or procedural grid)
    cloth_z = table_top_z + drop_height
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

    # Color for VBD solver
    if solver_type == "vbd":
        builder.color(include_bending=True)

    # Finalize
    model = builder.finalize()
    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 1.0e0
    model.soft_contact_mu = 0.8

    # Build solver
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
        raise ValueError(f"Unsupported solver for cloth drop: {solver_type}")

    scene_info = {
        "garment": garment["name"] if garment else f"grid_{grid_res}x{grid_res}",
        "garment_category": garment["category"] if garment else "procedural",
        "preset": preset,
        "cloth_params": cloth_params,
        "props": [p["name"] for p in selected_props],
        "drop_height": drop_height,
        "table_height": table_height,
    }

    return model, solver, scene_info


def run_cloth_drop(
    model,
    solver,
    num_frames: int = 200,
    substeps: int = 10,
    dt: float = 1.0 / 60.0,
    viewer=None,
):
    """Run the cloth drop simulation loop.

    Returns final state.
    """
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
    parser = argparse.ArgumentParser(description="Newton Cloth Drop Simulation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--preset", type=str, default="cotton",
                        choices=list(CLOTH_PRESETS.keys()),
                        help="Cloth material preset")
    parser.add_argument("--num-props", type=int, default=2,
                        help="Number of DTC props on table")
    parser.add_argument("--use-grid", action="store_true",
                        help="Use procedural grid cloth instead of garment")
    parser.add_argument("--grid-res", type=int, default=30,
                        help="Grid cloth resolution (when --use-grid)")
    parser.add_argument("--garment", type=str, default=None,
                        help="Specific garment name to use")
    parser.add_argument("--num-frames", type=int, default=200,
                        help="Number of simulation frames")
    parser.add_argument("--substeps", type=int, default=10,
                        help="Substeps per frame")
    parser.add_argument("--solver", type=str, default="vbd",
                        choices=["vbd", "xpbd"],
                        help="Solver type")
    parser.add_argument("--solver-iters", type=int, default=10,
                        help="Solver iterations")
    parser.add_argument("--drop-height", type=float, default=0.5,
                        help="Drop height above table [m]")
    parser.add_argument("--table-height", type=float, default=0.4,
                        help="Table height [m]")
    parser.add_argument("--viewer", type=str, default="null",
                        choices=["null", "usd", "gl"],
                        help="Viewer type")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to config output_dir)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to sim_config.json")
    parser.add_argument("--device", type=str, default=None,
                        help="Warp device (e.g. cuda:0)")
    parser.add_argument("--test", action="store_true",
                        help="Test run: few frames, GL viewer")
    args = parser.parse_args()

    if args.test:
        args.num_frames = TEST_FRAMES
        args.viewer = args.viewer if args.viewer != "null" else "gl"

    # Init Warp
    wp.init()
    if args.device:
        device = wp.get_device(args.device)
    else:
        device = wp.get_preferred_device()

    with wp.ScopedDevice(device):
        config = load_config(args.config)

        # Output path
        output_dir = Path(args.output_dir) if args.output_dir else Path(config.get("output_dir", "./output"))
        sim_name = f"cloth_drop_s{args.seed:06d}"
        sim_dir = output_dir / "cloth_drop" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Cloth Drop (seed={args.seed}) ===")
        print(f"  Device: {device}")
        print(f"  Output: {sim_dir}")

        # Build scene
        model, solver, scene_info = build_cloth_drop_scene(
            config=config,
            seed=args.seed,
            preset=args.preset,
            num_props=args.num_props,
            use_grid=args.use_grid,
            grid_res=args.grid_res,
            garment_name=args.garment,
            table_height=args.table_height,
            drop_height=args.drop_height,
            solver_type=args.solver,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        # Run simulation
        final_state = run_cloth_drop(
            model, solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        # Validate: cloth should be resting, not exploded
        particle_q = final_state.particle_q.numpy()
        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)
        print(f"  Final cloth bbox: min={min_pos}, max={max_pos}, size={bbox_size:.3f}")

        if bbox_size > 20.0:
            print("  WARNING: Simulation may have exploded!")
        if min_pos[2] < -0.5:
            print("  WARNING: Excessive ground penetration!")

        # Write metadata
        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="cloth_drop",
            seed=args.seed,
            solver=args.solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            bbox_size=float(bbox_size),
            min_z=float(min_pos[2]),
            **scene_info,
        )

        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
