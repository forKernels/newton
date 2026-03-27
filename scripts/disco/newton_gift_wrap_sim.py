"""
Newton Gift Wrap Simulation
===============================
Softbody blocks held together by cloth straps, dropped under gravity.
Multi-physics: volumetric deformables + cloth wrapping.

Based on Newton's falling_gift example.

Usage:
    uv run python scripts/disco/newton_gift_wrap_sim.py
    uv run python scripts/disco/newton_gift_wrap_sim.py --test
    uv run python scripts/disco/newton_gift_wrap_sim.py --seed 42 --num-blocks 4
"""

from __future__ import annotations

import argparse
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


def _cloth_strap_verts_and_faces(
    hx: float,
    hz: float,
    width: float,
    center_y: float,
    nu: int = 60,
    nv: int = 4,
):
    """Generate a closed cloth loop (strap) around a box in X-Z plane."""
    verts = []
    faces = []
    P = 4.0 * (hx + hz)

    for i in range(nu):
        s = (i / nu) * P
        # Walk rectangle perimeter
        if s < 2 * hx:
            x = -hx + s
            z = -hz
        elif s < 2 * hx + 2 * hz:
            x = hx
            z = -hz + (s - 2 * hx)
        elif s < 4 * hx + 2 * hz:
            x = hx - (s - 2 * hx - 2 * hz)
            z = hz
        else:
            x = -hx
            z = hz - (s - 4 * hx - 2 * hz)

        for j in range(nv):
            y = center_y - width / 2 + j * width / (nv - 1)
            verts.append((x, y, z))

    # Faces
    for i in range(nu):
        i_next = (i + 1) % nu
        for j in range(nv - 1):
            v0 = i * nv + j
            v1 = i * nv + j + 1
            v2 = i_next * nv + j + 1
            v3 = i_next * nv + j
            faces.append((v0, v1, v2))
            faces.append((v0, v2, v3))

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def build_gift_wrap_scene(
    seed: int = 0,
    num_blocks: int = 4,
    block_size: float = 0.3,
    num_straps: int = 2,
    drop_height: float = 1.5,
    stiffness: float = 1e4,
    solver_iters: int = 10,
):
    """Build gift wrap scene: stacked soft blocks with cloth straps."""
    rng = random.Random(seed)

    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    # Stack soft blocks
    block_half = block_size / 2
    for i in range(num_blocks):
        px = rng.uniform(-0.02, 0.02)
        py = rng.uniform(-0.02, 0.02)
        pz = drop_height + i * block_size * 1.05

        builder.add_soft_grid(
            pos=wp.vec3(px, py, pz),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4,
            dim_y=4,
            dim_z=4,
            cell_x=block_size / 4,
            cell_y=block_size / 4,
            cell_z=block_size / 4,
            density=500.0,
            k_mu=stiffness,
            k_lambda=stiffness,
            k_damp=1e-3,
        )

    # Add cloth straps around the stack
    stack_height = num_blocks * block_size
    hx = block_half * 1.05
    hz = (stack_height / 2) * 1.05

    for s in range(num_straps):
        y_pos = -block_half + (s + 1) * block_size / (num_straps + 1)
        strap_width = block_size * 0.15

        verts, faces = _cloth_strap_verts_and_faces(
            hx=hx,
            hz=hz,
            width=strap_width,
            center_y=y_pos,
        )
        # Offset to match stack center height
        verts[:, 2] += drop_height + stack_height / 2

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts],
            indices=faces.flatten().tolist(),
            density=0.3,
            tri_ke=5e3,
            tri_ka=5e3,
            tri_kd=1e-2,
            edge_ke=1e-1,
            edge_kd=1e-2,
            particle_radius=0.005,
        )

    builder.color()

    model = builder.finalize()
    model.soft_contact_ke = 1.0e4
    model.soft_contact_kd = 1e-3
    model.soft_contact_mu = 0.5

    solver = newton.solvers.SolverVBD(
        model,
        iterations=solver_iters,
        particle_enable_self_contact=True,
        particle_self_contact_radius=0.01,
        particle_self_contact_margin=0.02,
        particle_enable_tile_solve=True,
    )

    scene_info = {
        "num_blocks": num_blocks,
        "block_size": block_size,
        "num_straps": num_straps,
        "drop_height": drop_height,
        "stiffness": stiffness,
    }

    return model, solver, scene_info


def run_gift_wrap_sim(
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
    parser = argparse.ArgumentParser(description="Newton Gift Wrap Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--block-size", type=float, default=0.3)
    parser.add_argument("--num-straps", type=int, default=2)
    parser.add_argument("--drop-height", type=float, default=1.5)
    parser.add_argument("--stiffness", type=float, default=1e4)
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
        sim_name = f"gift_wrap_s{args.seed:06d}"
        sim_dir = output_dir / "gift_wrap" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Gift Wrap (seed={args.seed}) ===")

        model, solver, scene_info = build_gift_wrap_scene(
            seed=args.seed,
            num_blocks=args.num_blocks,
            block_size=args.block_size,
            num_straps=args.num_straps,
            drop_height=args.drop_height,
            stiffness=args.stiffness,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_gift_wrap_sim(
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
            sim_type="gift_wrap",
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
