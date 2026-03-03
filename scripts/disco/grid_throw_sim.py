"""
Grid Cloth Throw Simulation (Blender)
======================================
Throw a grid (subdivided plane) cloth onto furniture using the grab-and-throw
mechanism: Empty + Hook modifier + vertex group pinning with animated release.

Reuses throw trajectory logic from cloth_throw_sim.py but replaces garment
import with a procedural grid mesh (Solidify + Subdivision Surface applied).

Output pipeline: bake -> .blend -> .abc -> .usda -> .json

Usage (headless):
  blender --background --python scripts/disco/grid_throw_sim.py -- \
      --furniture-dir "D:/_blender/_myBlender/SimulationWork/seedAssets/scenes" \
      --output-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Sims" \
      --cloth-preset Cotton --num-samples 10 \
      --grid-size 1.0 --grid-cuts 30

  # Specific throw style:
  blender --background --python scripts/disco/grid_throw_sim.py -- \
      --throw-style swing_up --power 0.7 --grid-size 1.5 --grid-cuts 40
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
    garment_height,
    load_sim_config,
    parse_blender_args,
    resolve_presets,
    save_blend,
    write_metadata,
)
from cloth_throw_sim import (
    THROW_STYLES,
    apply_throw_cloth,
    setup_throw,
)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_FRAMES = 50
DEFAULT_GRID_SIZE = 1.0
DEFAULT_GRID_CUTS = 30
DEFAULT_THICKNESS = 0.001
DEFAULT_SUBDIV = 1


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
    throw_style: str = "random",
    power: float | None = None,
) -> dict:
    """Execute one grid cloth throw simulation and export results."""
    rng = random.Random(seed)
    furniture_name = furniture_path.stem

    # Pick style
    style = rng.choice(THROW_STYLES) if throw_style == "random" else throw_style

    # Pick power
    pwr = rng.uniform(0.3, 1.0) if power is None else power

    # Random direction around furniture
    direction_deg = rng.uniform(0, 360)

    # Output paths
    sim_dir = output_dir / furniture_name / "grid_throw" / preset_name
    sim_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"grid_{grid_size}m_{sample_idx:03d}"
    blend_path = sim_dir / f"{base_name}.blend"
    abc_path = sim_dir / f"{base_name}.abc"
    usda_path = sim_dir / f"{base_name}.usda"
    json_path = sim_dir / f"{base_name}.json"

    if usda_path.exists() and json_path.exists():
        print(f"  SKIP (exists): {usda_path.relative_to(output_dir)}")
        return {"skipped": True}

    print(f"  SIM [grid_throw/{style}]: {furniture_name} / {preset_name} #{sample_idx}")

    # 1. Clean scene
    clear_scene()

    # 2. Append furniture
    furniture_objs = append_furniture(str(furniture_path))
    if not furniture_objs:
        print(f"    WARNING: No mesh objects in {furniture_path.name}, skipping")
        return {"error": "no_furniture_meshes"}

    # 3. Create grid cloth
    grid = create_cloth_grid(size=grid_size, cuts=grid_cuts, thickness=thickness, subdiv_level=subdiv_level)

    # 4. Compute furniture bbox
    furn_min, furn_max = furniture_bbox(furniture_objs)

    # 5. Position grid above furniture (will be moved by Hook)
    center = furniture_center(furn_min, furn_max)
    drop_z = furn_max.z + garment_height(grid) + 0.3
    grid.location = (center.x, center.y, drop_z)
    grid.rotation_euler = (0, 0, math.radians(rng.uniform(0, 360)))
    bpy.context.view_layer.update()

    # 6. Add floor plane
    add_floor_plane(furn_min.z - 0.01)

    # 7. Add collision to furniture
    add_collision(furniture_objs)

    # 8. Set up throw mechanism (Empty + Hook + vertex group)
    throw_meta = setup_throw(grid, style, pwr, direction_deg, furn_min, furn_max)
    print(
        f"    style={style} power={pwr:.2f} dir={direction_deg:.0f}deg "
        f"pinned={throw_meta['pinned_vertices']} release_frame={throw_meta['release_frame']}"
    )

    # 9. Apply cloth with pin group and animated release
    apply_throw_cloth(grid, preset_name)

    # 10. Bake simulation
    t0 = time.time()
    bake_simulation(frame_count)
    bake_time = time.time() - t0
    print(f"    bake: {bake_time:.1f}s ({frame_count} frames)")

    # 11. Save .blend
    save_blend(str(blend_path))
    print(f"    saved: {blend_path.relative_to(output_dir)}")

    # 12. Export Alembic
    export_alembic(str(abc_path))
    print(f"    exported: {abc_path.relative_to(output_dir)}")

    # 13. Convert .abc -> .usda
    convert_abc_to_usda(str(abc_path), str(usda_path))
    print(f"    converted: {usda_path.relative_to(output_dir)}")

    # 14. Write metadata
    metadata = {
        "mode": "grid_throw",
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
        **throw_meta,
    }
    write_metadata(str(json_path), metadata)

    return metadata


# =============================================================================
# Main
# =============================================================================


def main():
    argv = parse_blender_args(sys.argv)

    parser = argparse.ArgumentParser(description="Grid cloth throw simulation (Blender)")
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
    parser.add_argument(
        "--throw-style", type=str, default="random", choices=["random"] + THROW_STYLES,
    )
    parser.add_argument("--power", type=float, default=None, help="Throw power 0.3-1.0")

    config = load_sim_config(extract_config_path(argv))
    apply_config_defaults(parser, config)
    args = parser.parse_args(argv)

    for name in ("furniture_dir", "output_dir"):
        if not getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is required (or set in config)")

    print("=" * 70)
    print(f"GRID CLOTH THROW SIMULATION -- style: {args.throw_style}")
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
                    throw_style=args.throw_style,
                    power=args.power,
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
