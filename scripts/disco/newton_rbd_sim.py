"""
Newton Rigid Body Dynamics Simulation
========================================
Drops and scatters DTC props onto a table/ground for rigid body dynamics data.

Usage:
    uv run python scripts/disco/newton_rbd_sim.py
    uv run python scripts/disco/newton_rbd_sim.py --seed 42 --num-props 10
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
    add_dtc_prop_as_rigid_body,
    add_ground_plane,
    add_table,
    create_viewer,
    discover_dtc_props,
    load_config,
    write_sim_metadata,
)


def build_rbd_scene(
    config: dict,
    seed: int = 0,
    num_props: int = 8,
    use_table: bool = True,
    table_height: float = 0.4,
    drop_spread: float = 0.3,
    drop_height_range: tuple[float, float] = (0.5, 1.5),
    solver_type: str = "xpbd",
    solver_iters: int = 10,
):
    """Build RBD scene: multiple DTC props dropped onto table/ground."""
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

    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    base_z = 0.0
    if use_table:
        table_half = (0.6, 0.6, 0.02)
        add_table(builder, position=(0.0, 0.0, table_height), size=table_half)
        base_z = table_height + table_half[2]

    prop_names = []
    for i, prop in enumerate(selected_props):
        meta = prop["metadata"]
        dims = meta.get("dimensions_m", [0.1, 0.1, 0.1])
        prop_height = max(dims) / 2.0

        px = rng.uniform(-drop_spread, drop_spread)
        py = rng.uniform(-drop_spread, drop_spread)
        pz = base_z + rng.uniform(*drop_height_range) + prop_height

        # Random rotation
        angle = rng.uniform(0, 2 * math.pi)
        axis = [rng.gauss(0, 1) for _ in range(3)]
        norm = math.sqrt(sum(a * a for a in axis)) or 1.0
        axis = [a / norm for a in axis]
        rot = wp.quat_from_axis_angle(wp.vec3(*axis), angle)
        rotation = (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))

        add_dtc_prop_as_rigid_body(builder, prop, position=(px, py, pz), rotation=rotation)
        prop_names.append(prop["name"])
        print(f"  Prop {i}: {prop['name']} at z={pz:.2f}")

    model = builder.finalize()

    if solver_type == "xpbd":
        solver = newton.solvers.SolverXPBD(model, iterations=solver_iters)
    elif solver_type == "mujoco":
        solver = newton.solvers.SolverMuJoCo(model)
    else:
        raise ValueError(f"Unsupported solver for RBD: {solver_type}")

    scene_info = {
        "props": prop_names,
        "num_props": len(selected_props),
        "use_table": use_table,
        "table_height": table_height,
    }

    return model, solver, scene_info


def run_rbd_sim(
    model,
    solver,
    num_frames: int = 200,
    substeps: int = 10,
    dt: float = 1.0 / 60.0,
    viewer=None,
):
    """Run RBD simulation. Returns final state."""
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, None, state_0)
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
    parser = argparse.ArgumentParser(description="Newton RBD Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-props", type=int, default=8)
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--table-height", type=float, default=0.4)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--solver", type=str, default="xpbd", choices=["xpbd", "mujoco"])
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
        sim_name = f"rbd_s{args.seed:06d}"
        sim_dir = output_dir / "rbd" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton RBD (seed={args.seed}) ===")

        model, solver, scene_info = build_rbd_scene(
            config=config,
            seed=args.seed,
            num_props=args.num_props,
            use_table=not args.no_table,
            table_height=args.table_height,
            solver_type=args.solver,
            solver_iters=args.solver_iters,
        )

        usd_path = sim_dir / f"{sim_name}.usd" if args.viewer == "usd" else None
        viewer = create_viewer(output_path=usd_path, viewer_type=args.viewer)

        final_state = run_rbd_sim(
            model, solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="rbd",
            seed=args.seed,
            solver=args.solver,
            num_frames=args.num_frames,
            **scene_info,
        )

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
