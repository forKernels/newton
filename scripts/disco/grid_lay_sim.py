"""
Grid Cloth Lay Simulation (Blender)
====================================
Places a grid cloth flat just above furniture and lets it settle naturally
with gravity and cloth physics.  Simplest of the grid modes — no pinning,
no throwing, just drape and settle.

Output pipeline: bake -> .blend -> .abc -> .usda -> .json

Usage (headless):
  blender --background --python scripts/disco/grid_lay_sim.py -- \
      --furniture-dir "D:/_blender/_myBlender/SimulationWork/seedAssets/scenes" \
      --output-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Sims" \
      --cloth-preset Cotton --num-samples 10 \
      --grid-size 1.0 --grid-cuts 30

  # Larger cloth:
  blender --background --python scripts/disco/grid_lay_sim.py -- \
      --grid-size 2.0 --grid-cuts 50 --frames 80
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy
from cloth_sim_utils import (
    add_collision,
    add_floor_plane,
    append_furniture,
    apply_cloth_preset,
    apply_config_defaults,
    bake_simulation,
    clear_scene,
    convert_abc_to_usda,
    create_cloth_grid,
    export_alembic,
    extract_config_path,
    find_furniture,
    furniture_bbox,
    furniture_center,
    load_sim_config,
    parse_blender_args,
    resolve_presets,
    save_blend,
    write_metadata,
)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_FRAMES = 50
DEFAULT_GRID_SIZE = 1.0
DEFAULT_GRID_CUTS = 30
DEFAULT_THICKNESS = 0.001
DEFAULT_SUBDIV = 1
LAY_GAP_M = 0.02  # metres above furniture top


# =============================================================================
# Positioning
# =============================================================================


def position_lay(
    grid: bpy.types.Object,
    furn_min,
    furn_max,
    rng: random.Random,
) -> dict:
    """Position grid flat above furniture with random Z rotation and slight XY offset."""
    center = furniture_center(furn_min, furn_max)

    # Random XY offset within ±30% of furniture extent for variety
    x_span = furn_max.x - furn_min.x
    y_span = furn_max.y - furn_min.y
    x_offset = rng.uniform(-0.3, 0.3) * x_span
    y_offset = rng.uniform(-0.3, 0.3) * y_span

    x = center.x + x_offset
    y = center.y + y_offset
    z = furn_max.z + LAY_GAP_M

    rot_z_deg = rng.uniform(0, 360)

    grid.location = (x, y, z)
    grid.rotation_euler = (0, 0, math.radians(rot_z_deg))
    bpy.context.view_layer.update()

    return {
        "lay_position": [round(x, 4), round(y, 4), round(z, 4)],
        "lay_rotation_z": round(rot_z_deg, 1),
    }


# =============================================================================
# Single simulation run
# =============================================================================


def run_single_sim(
    furniture_path: Path,
    preset_name: str,
    output_dir: Path,
    seed: int,
    sample_idx: int,
    *,
    frame_count: int = DEFAULT_FRAMES,
    grid_size: float = DEFAULT_GRID_SIZE,
    grid_cuts: int = DEFAULT_GRID_CUTS,
    thickness: float = DEFAULT_THICKNESS,
    subdiv_level: int = DEFAULT_SUBDIV,
) -> dict:
    """Execute one grid cloth lay simulation and export results."""
    rng = random.Random(seed)
    furniture_name = furniture_path.stem

    # Output paths
    sim_dir = output_dir / furniture_name / "grid_lay" / preset_name
    sim_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"grid_{grid_size}m_{sample_idx:03d}"
    blend_path = sim_dir / f"{base_name}.blend"
    abc_path = sim_dir / f"{base_name}.abc"
    usda_path = sim_dir / f"{base_name}.usda"
    json_path = sim_dir / f"{base_name}.json"

    if usda_path.exists() and json_path.exists():
        print(f"  SKIP (exists): {usda_path.relative_to(output_dir)}")
        return {"skipped": True}

    print(f"  SIM [grid_lay]: {furniture_name} / {preset_name} #{sample_idx}")

    # 1. Clean scene
    clear_scene()

    # 2. Append furniture
    furniture_objs = append_furniture(str(furniture_path))
    if not furniture_objs:
        print(f"    WARNING: No mesh objects in {furniture_path.name}, skipping")
        return {"error": "no_furniture_meshes"}

    # 3. Create grid cloth
    grid = create_cloth_grid(size=grid_size, cuts=grid_cuts, thickness=thickness, subdiv_level=subdiv_level)

    # 4. Compute furniture bbox + add floor
    furn_min, furn_max = furniture_bbox(furniture_objs)
    add_floor_plane(furn_min.z - 0.01)

    # 5. Position grid flat above furniture
    placement = position_lay(grid, furn_min, furn_max, rng)
    p = placement["lay_position"]
    print(f"    pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) rot_z={placement['lay_rotation_z']:.1f}deg")

    # 6. Add collision to furniture
    add_collision(furniture_objs)

    # 7. Apply cloth preset (no pinning -- pure gravity settle)
    apply_cloth_preset(grid, preset_name)

    # 8. Bake simulation
    t0 = time.time()
    bake_simulation(frame_count)
    bake_time = time.time() - t0
    print(f"    bake: {bake_time:.1f}s ({frame_count} frames)")

    # 9. Save .blend
    save_blend(str(blend_path))
    print(f"    saved: {blend_path.relative_to(output_dir)}")

    # 10. Export Alembic
    export_alembic(str(abc_path))
    print(f"    exported: {abc_path.relative_to(output_dir)}")

    # 11. Convert .abc -> .usda
    convert_abc_to_usda(str(abc_path), str(usda_path))
    print(f"    converted: {usda_path.relative_to(output_dir)}")

    # 12. Write metadata
    metadata = {
        "mode": "grid_lay",
        "furniture": furniture_path.name,
        "cloth_preset": preset_name,
        "grid_size": grid_size,
        "grid_cuts": grid_cuts,
        "thickness": thickness,
        "subdiv_level": subdiv_level,
        "frames": frame_count,
        "seed": seed,
        "bake_time_s": round(bake_time, 1),
        "blend_file": blend_path.name,
        "abc_file": abc_path.name,
        "usda_file": usda_path.name,
        **placement,
    }
    write_metadata(str(json_path), metadata)

    return metadata


# =============================================================================
# Main
# =============================================================================


def main():
    argv = parse_blender_args(sys.argv)

    parser = argparse.ArgumentParser(description="Grid cloth lay simulation (Blender)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--furniture-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--furniture", type=str, default=None, help="Specific furniture .blend filename")
    parser.add_argument("--cloth-preset", type=str, default="Cotton")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--grid-size", type=float, default=DEFAULT_GRID_SIZE, help="Grid side length in metres")
    parser.add_argument("--grid-cuts", type=int, default=DEFAULT_GRID_CUTS, help="Subdivision cuts per axis")
    parser.add_argument("--thickness", type=float, default=DEFAULT_THICKNESS, help="Solidify thickness in metres")
    parser.add_argument("--subdiv-level", type=int, default=DEFAULT_SUBDIV, help="Subdivision Surface level")

    config = load_sim_config(extract_config_path(argv))
    apply_config_defaults(parser, config)
    args = parser.parse_args(argv)

    for name in ("furniture_dir", "output_dir"):
        if not getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is required (or set in config)")

    print("=" * 70)
    print("GRID CLOTH LAY SIMULATION")
    print("=" * 70)

    furniture_files = find_furniture(args.furniture_dir, args.furniture)
    print(f"Furniture: {len(furniture_files)} files")

    presets = resolve_presets(args.cloth_preset)
    print(f"Presets: {presets}")
    print(f"Grid: {args.grid_size}m, {args.grid_cuts} cuts, thickness={args.thickness}m, subdiv={args.subdiv_level}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(furniture_files) * len(presets) * args.num_samples
    print(f"Total simulations: {total}")

    completed = 0
    errors = 0
    skipped = 0
    t_start = time.time()

    for furn_path in furniture_files:
        for preset in presets:
            for sample_i in range(1, args.num_samples + 1):
                seed = args.seed + hash((furn_path.stem, preset, sample_i)) % (2**31)
                result = run_single_sim(
                    furniture_path=furn_path,
                    preset_name=preset,
                    output_dir=output_dir,
                    seed=seed,
                    sample_idx=sample_i,
                    frame_count=args.frames,
                    grid_size=args.grid_size,
                    grid_cuts=args.grid_cuts,
                    thickness=args.thickness,
                    subdiv_level=args.subdiv_level,
                )
                if result.get("skipped"):
                    skipped += 1
                elif result.get("error"):
                    errors += 1
                else:
                    completed += 1

    duration = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"DONE -- {completed} simulations in {duration:.1f}s ({skipped} skipped, {errors} errors)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
