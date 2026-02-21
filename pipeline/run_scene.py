# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point: build a Franka + garment scene, simulate headlessly, export USDA.

Usage:
    uv run python -m pipeline.run_scene \\
        --garment-usd path/to/garment.usd \\
        --garment-prim /Root/GarmentMesh \\
        --output path/to/output.usda \\
        --num-frames 500 \\
        --fps 60
"""

from __future__ import annotations

import argparse
import time

import newton.examples
from newton.viewer import ViewerUSD

from .scene_assembler import SceneAssembler, simulate_frame
from .scene_config import ElementConfig, ElementType, SceneConfig
from .usd_materials import bind_materials


def build_config(args: argparse.Namespace) -> SceneConfig:
    """Create a SceneConfig from CLI arguments."""
    config = SceneConfig(
        output_path=args.output,
        sim=SceneConfig().sim,  # start with defaults
    )
    config.sim.fps = args.fps
    config.sim.num_frames = args.num_frames

    # Resolve garment asset
    garment_path = args.garment_usd
    garment_prim = args.garment_prim
    if garment_path is None:
        # Fall back to Newton's built-in shirt
        garment_path = newton.examples.get_asset("unisex_shirt.usd")
        garment_prim = "/root/shirt"

    config.elements.append(
        ElementConfig(
            element_type=ElementType.CLOTH_USD,
            asset_path=garment_path,
            prim_path=garment_prim,
        )
    )
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Scene Orchestrator — Franka + garment simulation pipeline",
    )
    parser.add_argument(
        "--garment-usd",
        type=str,
        default=None,
        help="Path to garment USD file. Defaults to Newton's built-in unisex_shirt.usd.",
    )
    parser.add_argument(
        "--garment-prim",
        type=str,
        default="/root/shirt",
        help="USD prim path within the garment file (default: /root/shirt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.usda",
        help="Output USDA file path (default: output.usda).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=500,
        help="Number of frames to simulate (default: 500).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second (default: 60).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file. Overrides other arguments if provided.",
    )
    args = parser.parse_args(argv)

    # Build config
    if args.config:
        config = SceneConfig.from_json(args.config)
    else:
        config = build_config(args)

    print("Scene Orchestrator Pipeline")
    print(f"  Output:     {config.output_path}")
    print(f"  Frames:     {config.sim.num_frames}")
    print(f"  FPS:        {config.sim.fps}")
    print(f"  Substeps:   {config.sim.sim_substeps}")
    print(f"  Elements:   {len(config.elements)}")
    print()

    # Assemble scene
    print("Assembling scene...")
    t0 = time.perf_counter()
    assembler = SceneAssembler()
    scene = assembler.assemble(config)
    print(f"  Assembly took {time.perf_counter() - t0:.1f}s")
    print(f"  Bodies:     {scene.model.body_count}")
    print(f"  Particles:  {scene.model.particle_count}")
    print(f"  Joints:     {scene.model.joint_count}")
    print()

    # Set up USD viewer for export
    viewer = ViewerUSD(
        output_path=config.output_path,
        fps=config.sim.fps,
        num_frames=config.sim.num_frames,
    )
    viewer.set_model(scene.model)

    # Simulation loop
    frame_dt = 1.0 / config.sim.fps
    sim_dt = frame_dt / config.sim.sim_substeps
    sim_time = 0.0

    print(f"Simulating {config.sim.num_frames} frames...")
    t0 = time.perf_counter()

    for frame_idx in range(config.sim.num_frames):
        # Update robot control targets
        scene.robot_controller.generate_joint_qd(scene.state_0, sim_time)

        # Step physics
        simulate_frame(scene, sim_dt, config.sim.sim_substeps)

        # Record frame
        viewer.begin_frame(sim_time)
        viewer.log_state(scene.state_0)
        viewer.end_frame()

        sim_time += frame_dt

        # Progress
        if (frame_idx + 1) % 50 == 0 or frame_idx == config.sim.num_frames - 1:
            elapsed = time.perf_counter() - t0
            fps_actual = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_idx + 1}/{config.sim.num_frames}  ({fps_actual:.1f} frames/s)")

    elapsed = time.perf_counter() - t0
    print(f"\nSimulation complete: {elapsed:.1f}s total ({config.sim.num_frames / elapsed:.1f} frames/s)")

    # Save USD file
    viewer.close()

    # Bind PBR materials
    print("Binding materials...")
    bind_materials(config.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
