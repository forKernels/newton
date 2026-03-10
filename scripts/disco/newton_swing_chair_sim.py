"""
Newton Swinging Chair Simulation
====================================
Loads a hanging chair from USDA (with _Soft / _Static mesh flags),
suspends it, then throws DTC props at it.

Mesh naming convention from Blender export:
  *_Soft   → soft body (add_soft_grid or add_cloth_mesh)
  *_Static → fixed rigid body (body=-1)

Usage:
    uv run python scripts/disco/newton_swing_chair_sim.py
    uv run python scripts/disco/newton_swing_chair_sim.py --test
    uv run python scripts/disco/newton_swing_chair_sim.py --seed 42 --num-projectiles 5
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import warp as wp

import newton

from newton_sim_utils import (
    TEST_FRAMES,
    add_dtc_prop_as_rigid_body,
    add_ground_plane,
    create_blend_file,
    create_viewer,
    discover_dtc_props,
    load_config,
    write_sim_metadata,
)


def _load_usda_meshes(usda_path: str | Path):
    """Parse a USDA and return mesh data grouped by physics flag.

    Returns list of dicts: {name, flag, vertices (Nx3 float32), triangles (Mx3 int32)}
    """
    # Add OpenUSD DLL dirs
    usd_root = os.environ.get("OPENUSD_ROOT", "C:/_tools/OpenUSD/25.08")
    if Path(usd_root).is_dir():
        os.add_dll_directory(str(Path(usd_root) / "bin"))
        os.add_dll_directory(str(Path(usd_root) / "lib"))

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usda_path))
    meshes = []

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        parent_name = prim.GetParent().GetName()

        # Determine physics flag from parent Xform name
        flag = "rigid"  # default
        name_lower = parent_name.lower()
        if name_lower.endswith("_soft"):
            flag = "soft"
        elif name_lower.endswith("_static"):
            flag = "static"
        elif name_lower.endswith("_cloth"):
            flag = "cloth"
        elif name_lower.endswith("_rigid"):
            flag = "rigid"

        # Extract geometry
        points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
        face_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        face_counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)

        # Triangulate quads
        triangles = []
        idx = 0
        for fc in face_counts:
            if fc == 3:
                triangles.append(face_indices[idx:idx + 3])
            elif fc == 4:
                triangles.append(face_indices[[idx, idx + 1, idx + 2]])
                triangles.append(face_indices[[idx, idx + 2, idx + 3]])
            else:
                # Fan triangulation for n-gons
                for j in range(1, fc - 1):
                    triangles.append(np.array([face_indices[idx], face_indices[idx + j], face_indices[idx + j + 1]]))
            idx += fc

        triangles = np.array(triangles, dtype=np.int32)

        meshes.append({
            "name": f"{parent_name}/{prim.GetName()}",
            "flag": flag,
            "vertices": points,
            "triangles": triangles,
        })
        print(f"  Mesh: {parent_name}/{prim.GetName()} -> {flag} ({len(points)} verts, {len(triangles)} tris)")

    return meshes


def build_swing_chair_scene(
    config: dict,
    seed: int = 0,
    chair_usda: str | None = None,
    num_projectiles: int = 3,
    throw_speed: float = 3.0,
    solver_iters: int = 10,
):
    """Build swinging chair scene: chair from USDA + thrown projectiles."""
    rng = random.Random(seed)

    # Find chair USDA
    if chair_usda is None:
        scenes_dir = config.get("scenes_dir", "")
        chair_usda = str(Path(scenes_dir) / "HangingChair_01.usda")

    if not Path(chair_usda).exists():
        raise FileNotFoundError(f"Chair USDA not found: {chair_usda}")

    print(f"  Loading chair: {chair_usda}")
    chair_meshes = _load_usda_meshes(chair_usda)

    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    # Add chair meshes based on flags
    for mesh_data in chair_meshes:
        verts = mesh_data["vertices"]
        tris = mesh_data["triangles"]
        flag = mesh_data["flag"]

        if flag == "static":
            # Fixed rigid body — the metal frame
            nm = newton.Mesh(vertices=verts, indices=tris.flatten())
            cfg = newton.ModelBuilder.ShapeConfig()
            cfg.density = 0.0
            cfg.mu = 0.5
            builder.add_shape_mesh(body=-1, mesh=nm, cfg=cfg)

        elif flag == "soft":
            # Cloth mesh — ropes, leather seat, pillow
            vert_list = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts]
            idx_list = tris.flatten().tolist()
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=vert_list,
                indices=idx_list,
                density=0.5,
                tri_ke=5e3,
                tri_ka=5e3,
                tri_kd=1e-2,
                edge_ke=1e-1,
                edge_kd=1e-2,
                particle_radius=0.005,
            )

    # Discover DTC props for projectiles
    dtc_props = discover_dtc_props(
        config.get("dtc_dir", ""),
        unique_only=config.get("dtc_unique_only", True),
    )

    projectile_info = []
    projectile_bodies = []
    if dtc_props and num_projectiles > 0:
        n = min(num_projectiles, len(dtc_props))
        selected = rng.sample(dtc_props, n)

        for i, prop in enumerate(selected):
            # Random launch position: off to the side, aimed at chair center (~0, 0, 0.6)
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(1.5, 2.5)
            launch_x = dist * math.cos(angle)
            launch_y = dist * math.sin(angle)
            launch_z = rng.uniform(0.4, 1.2)

            body_idx = add_dtc_prop_as_rigid_body(
                builder, prop,
                position=(launch_x, launch_y, launch_z),
                static=False,
            )
            projectile_bodies.append(body_idx)

            # Velocity aimed at chair center with some randomness
            target = np.array([0.0, 0.0, 0.6])
            origin = np.array([launch_x, launch_y, launch_z])
            direction = target - origin
            direction = direction / np.linalg.norm(direction)
            direction += np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05)])
            vel = direction * throw_speed

            projectile_info.append({
                "name": prop["name"],
                "body_idx": body_idx,
                "position": (launch_x, launch_y, launch_z),
                "velocity": vel.tolist(),
            })
            print(f"  Projectile {i}: {prop['name']} from ({launch_x:.2f}, {launch_y:.2f}, {launch_z:.2f})")

    # Color for VBD solver (needed for cloth)
    builder.color(include_bending=True)

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

    # Set projectile velocities
    initial_state = model.state()
    if projectile_bodies:
        body_qd = initial_state.body_qd.numpy()
        for info in projectile_info:
            idx = info["body_idx"]
            if 0 <= idx < len(body_qd):
                vx, vy, vz = info["velocity"]
                body_qd[idx] = [0.0, 0.0, 0.0, vx, vy, vz]
        initial_state.body_qd = wp.array(body_qd, dtype=wp.spatial_vector)

    scene_info = {
        "chair_usda": str(chair_usda),
        "num_meshes": len(chair_meshes),
        "mesh_flags": {m["name"]: m["flag"] for m in chair_meshes},
        "num_projectiles": len(projectile_info),
        "projectiles": [p["name"] for p in projectile_info],
        "throw_speed": throw_speed,
    }

    return model, solver, initial_state, scene_info


def run_swing_chair_sim(
    model,
    solver,
    initial_state,
    num_frames: int = 300,
    substeps: int = 10,
    dt: float = 1.0 / 60.0,
    viewer=None,
):
    state_0 = initial_state
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
    parser = argparse.ArgumentParser(description="Newton Swinging Chair Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chair-usda", type=str, default=None,
                        help="Path to chair USDA (default: scenes_dir/HangingChair_01.usda)")
    parser.add_argument("--num-projectiles", type=int, default=3,
                        help="Number of DTC props to throw at chair")
    parser.add_argument("--throw-speed", type=float, default=3.0,
                        help="Projectile launch speed [m/s]")
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--solver-iters", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--test", action="store_true",
                        help="Test run: few frames")
    args = parser.parse_args()

    if args.test:
        args.num_frames = TEST_FRAMES

    wp.init()
    if args.device:
        device = wp.get_device(args.device)
    else:
        device = wp.get_preferred_device()

    with wp.ScopedDevice(device):
        config = load_config(args.config)

        output_dir = Path(args.output_dir) if args.output_dir else Path(config.get("output_dir", "./output"))
        sim_name = f"swing_chair_s{args.seed:06d}"
        sim_dir = output_dir / "swing_chair" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Swinging Chair (seed={args.seed}) ===")

        model, solver, initial_state, scene_info = build_swing_chair_scene(
            config=config,
            seed=args.seed,
            chair_usda=args.chair_usda,
            num_projectiles=args.num_projectiles,
            throw_speed=args.throw_speed,
            solver_iters=args.solver_iters,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_swing_chair_sim(
            model, solver, initial_state,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="swing_chair",
            seed=args.seed,
            **scene_info,
        )

        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
