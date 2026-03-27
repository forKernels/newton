"""
Newton Beans Spill Simulation
================================
Spills granular particles (beans) from a tilted container using the
MPM (Material Point Method) solver for realistic granular flow.

Usage:
    uv run python scripts/disco/newton_beans_sim.py
    uv run python scripts/disco/newton_beans_sim.py --test
    uv run python scripts/disco/newton_beans_sim.py --seed 42 --num-particles 5000
"""

from __future__ import annotations

import argparse
import math
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
from newton.solvers import SolverImplicitMPM


def build_beans_scene(
    seed: int = 0,
    emit_size: tuple[float, float, float] = (0.3, 0.3, 0.3),
    emit_height: float = 1.0,
    container_tilt_deg: float = 30.0,
    density: float = 1200.0,
    voxel_size: float = 0.02,
    friction: float = 0.5,
    young_modulus: float = 1.0e6,
    use_container: bool = True,
):
    """Build beans spill scene: particle volume + tilted container + ground.

    Uses MPM solver for granular material simulation.
    """
    rng = random.Random(seed)

    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)

    add_ground_plane(builder)

    # Container (tilted box walls — open top)
    if use_container:
        tilt_rad = math.radians(container_tilt_deg + rng.uniform(-5, 5))
        tilt_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt_rad)

        container_cfg = newton.ModelBuilder.ShapeConfig(mu=0.3)
        cx, cy, cz = 0.0, 0.0, emit_height + 0.15
        wall_thickness = 0.02

        # Bottom
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(cx, cy, cz), q=tilt_q),
            hx=0.2,
            hy=0.2,
            hz=wall_thickness,
            cfg=container_cfg,
        )
        # Side walls (4)
        for dx, dy in [(0.2, 0.0), (-0.2, 0.0), (0.0, 0.2), (0.0, -0.2)]:
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(p=wp.vec3(cx + dx, cy + dy, cz + 0.12), q=tilt_q),
                hx=wall_thickness if abs(dx) > 0.1 else 0.2,
                hy=wall_thickness if abs(dy) > 0.1 else 0.2,
                hz=0.12,
                cfg=container_cfg,
            )

    # Optional: a bowl/ramp to catch beans
    ramp_cfg = newton.ModelBuilder.ShapeConfig(mu=0.4)
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(
            p=wp.vec3(0.5, 0.0, 0.15),
            q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.15),
        ),
        hx=0.3,
        hy=0.3,
        hz=0.02,
        cfg=ramp_cfg,
    )

    # Emit particles (bean volume)
    particles_per_cell = 3
    emit_lo = np.array([-emit_size[0] / 2, -emit_size[1] / 2, emit_height])
    emit_hi = np.array([emit_size[0] / 2, emit_size[1] / 2, emit_height + emit_size[2]])

    particle_res = np.array(
        np.ceil(particles_per_cell * (emit_hi - emit_lo) / voxel_size),
        dtype=int,
    )
    cell_size = (emit_hi - emit_lo) / particle_res
    cell_volume = np.prod(cell_size)
    radius = np.max(cell_size) * 0.5
    mass = cell_volume * density

    builder.add_particle_grid(
        pos=wp.vec3(emit_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=int(particle_res[0]) + 1,
        dim_y=int(particle_res[1]) + 1,
        dim_z=int(particle_res[2]) + 1,
        cell_x=float(cell_size[0]),
        cell_y=float(cell_size[1]),
        cell_z=float(cell_size[2]),
        mass=float(mass),
        jitter=2.0 * float(radius),
        radius_mean=float(radius),
    )

    num_particles = (int(particle_res[0]) + 1) * (int(particle_res[1]) + 1) * (int(particle_res[2]) + 1)
    print(f"  Particles: {num_particles}")

    model = builder.finalize()

    # MPM solver options
    mpm_options = SolverImplicitMPM.Options()
    mpm_options.voxel_size = voxel_size
    mpm_options.max_iterations = 100
    mpm_options.tolerance = 1.0e-5

    # Set per-particle material properties
    model.mpm.young_modulus.fill_(young_modulus)
    model.mpm.poisson_ratio.fill_(0.3)
    model.mpm.friction.fill_(friction)
    model.mpm.density.fill_(density)

    solver = SolverImplicitMPM(model, mpm_options)

    scene_info = {
        "num_particles": num_particles,
        "emit_size": list(emit_size),
        "emit_height": emit_height,
        "container_tilt_deg": container_tilt_deg,
        "density": density,
        "friction": friction,
        "voxel_size": voxel_size,
    }

    return model, solver, scene_info


def run_beans_sim(
    model,
    solver,
    num_frames: int = 200,
    substeps: int = 1,
    dt: float = 1.0 / 60.0,
    viewer=None,
):
    """Run beans simulation using MPM step + project_outside."""
    state_0 = model.state()
    state_1 = model.state()

    sub_dt = dt / substeps

    if viewer:
        viewer.set_model(model)
        viewer.show_particles = True
        viewer.begin_frame(0.0)
        viewer.log_state(state_0)
        viewer.end_frame()

    for frame in range(num_frames):
        for _ in range(substeps):
            solver.step(state_0, state_1, None, None, sub_dt)
            solver.project_outside(state_1, state_1, sub_dt)
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
    parser = argparse.ArgumentParser(description="Newton Beans Spill Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--emit-size", type=float, nargs=3, default=[0.3, 0.3, 0.3], help="Bean volume extents [x, y, z] in meters"
    )
    parser.add_argument("--emit-height", type=float, default=1.0)
    parser.add_argument("--container-tilt", type=float, default=30.0, help="Container tilt angle [degrees]")
    parser.add_argument("--density", type=float, default=1200.0, help="Bean density [kg/m3]")
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--young-modulus", type=float, default=1.0e6)
    parser.add_argument("--no-container", action="store_true")
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--substeps", type=int, default=1)
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
        sim_name = f"beans_s{args.seed:06d}"
        sim_dir = output_dir / "beans" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Beans Spill (seed={args.seed}) ===")

        model, solver, scene_info = build_beans_scene(
            seed=args.seed,
            emit_size=tuple(args.emit_size),
            emit_height=args.emit_height,
            container_tilt_deg=args.container_tilt,
            density=args.density,
            friction=args.friction,
            voxel_size=args.voxel_size,
            young_modulus=args.young_modulus,
            use_container=not args.no_container,
        )

        # Always write USDA (the deliverable)
        usd_path = sim_dir / f"{sim_name}.usda"
        viewer = create_viewer(output_path=usd_path, viewer_type="usd")

        final_state = run_beans_sim(
            model,
            solver,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        particle_q = final_state.particle_q.numpy()
        min_z = float(np.min(particle_q[:, 2]))
        max_z = float(np.max(particle_q[:, 2]))
        spread = float(np.linalg.norm(np.max(particle_q, axis=0) - np.min(particle_q, axis=0)))

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="beans",
            seed=args.seed,
            min_z=min_z,
            max_z=max_z,
            spread=spread,
            **scene_info,
        )

        print(f"  Particles settled: z_range=[{min_z:.3f}, {max_z:.3f}], spread={spread:.3f}")
        # Create .blend file referencing the USDA
        blend_path = sim_dir / f"{sim_name}.blend"
        create_blend_file(usd_path, blend_path, blender_exe=config.get("blender_exe", "blender"))

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
