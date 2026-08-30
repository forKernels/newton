"""
Cloth Stack Simulation (Blender)
================================
Drops multiple garments onto furniture in sequence, creating stacked cloth
piles.  Records the cloth simulation as animated USDA.

Each stack contains 2-5 garments placed at staggered heights above the
furniture.  They all simulate simultaneously -- the lowest garment lands
first, each subsequent one lands on top of the growing pile.

Output structure:
  _Sims/{furniture}/stack/{preset}/stack{N}_{hash}_{idx}.usda

Usage (headless):
  blender --background --python scripts/disco/cloth_stack_sim.py -- \
      --furniture-dir "D:/_blender/_myBlender/SimulationWork/seedAssets/scenes" \
      --garment-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Set" \
      --output-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Sims" \
      --cloth-preset Cotton --num-samples 10

  # Larger stacks, all presets:
  blender --background --python scripts/disco/cloth_stack_sim.py -- \
      --furniture-dir "..." --garment-dir "..." --output-dir "..." \
      --min-stack 3 --max-stack 5 --cloth-preset all --num-samples 20

  # Single furniture + category:
  blender --background --python scripts/disco/cloth_stack_sim.py -- \
      --furniture "Chair_01.blend" \
      --garment-category dress_sleeveless_2550 \
      --cloth-preset Denim --num-samples 5
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
import time
from pathlib import Path

# Blender script -- ensure the sibling module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cloth_sim_utils import (
    BBOX_MARGIN,
    add_collision,
    append_furniture,
    apply_cloth_preset,
    apply_config_defaults,
    bake_simulation,
    clear_scene,
    export_usda,
    extract_config_path,
    find_furniture,
    find_prepped_garments,
    furniture_bbox,
    garment_height,
    import_garment,
    load_sim_config,
    make_seed,
    parse_blender_args,
    resolve_presets,
    write_metadata,
)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_FRAMES = 350  # More frames for stack settling
DROP_GAP_M = 0.3  # metres above furniture top for first garment
STACK_SPACING_M = 0.25  # vertical spacing between staggered garments

MIN_STACK_DEFAULT = 2
MAX_STACK_DEFAULT = 5


# =============================================================================
# Helpers
# =============================================================================


def stack_hash(garment_ids: list[str]) -> str:
    """Short hash of garment IDs for the output filename."""
    h = hashlib.md5("_".join(garment_ids).encode()).hexdigest()[:8]
    return h


# =============================================================================
# Single stack simulation
# =============================================================================


def run_stack_sim(
    furniture_path: Path,
    garments_pool: list[dict],
    preset_name: str,
    output_dir: Path,
    seed: int,
    sample_idx: int,
    *,
    min_stack: int = MIN_STACK_DEFAULT,
    max_stack: int = MAX_STACK_DEFAULT,
    frame_count: int = DEFAULT_FRAMES,
) -> dict:
    """Execute one cloth stack simulation and export results."""
    rng = random.Random(seed)
    furniture_name = furniture_path.stem

    # Determine stack size
    stack_count = rng.randint(min_stack, min(max_stack, len(garments_pool)))

    # Pick random garments for the stack
    selected = rng.sample(garments_pool, stack_count)
    garment_ids = [g["garment_id"] for g in selected]

    # Output paths
    sim_dir = output_dir / furniture_name / "stack" / preset_name
    sim_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"stack{stack_count}_{stack_hash(garment_ids)}_{sample_idx:03d}"
    usda_path = sim_dir / f"{base_name}.usda"
    json_path = sim_dir / f"{base_name}.json"

    if usda_path.exists() and json_path.exists():
        print(f"  SKIP (exists): {usda_path.relative_to(output_dir)}")
        return {"skipped": True}

    print(f"  SIM [stack]: {furniture_name} / {preset_name} / {stack_count} garments #{sample_idx}")

    # 1. Clean scene
    clear_scene()

    # 2. Append furniture
    furniture_objs = append_furniture(str(furniture_path))
    if not furniture_objs:
        print(f"    WARNING: No mesh objects in {furniture_path.name}, skipping")
        return {"error": "no_furniture_meshes"}

    # 3. Compute furniture bbox
    furn_min, furn_max = furniture_bbox(furniture_objs)

    # 4. Add collision to furniture
    add_collision(furniture_objs)

    # 5. Import and position all garments at staggered heights
    garment_objs = []
    placements = []

    for i, garment_info in enumerate(selected):
        garment = import_garment(garment_info["obj_path"])
        garment_objs.append(garment)

        # Random XY within furniture bbox (with margin)
        x_span = furn_max.x - furn_min.x
        y_span = furn_max.y - furn_min.y
        margin_x = x_span * BBOX_MARGIN
        margin_y = y_span * BBOX_MARGIN

        x = rng.uniform(furn_min.x + margin_x, furn_max.x - margin_x)
        y = rng.uniform(furn_min.y + margin_y, furn_max.y - margin_y)

        # Stagger height: each garment higher than the last
        g_height = garment_height(garment)
        z = furn_max.z + g_height + DROP_GAP_M + (i * STACK_SPACING_M)

        rot_z_deg = rng.uniform(0, 360)

        garment.location = (x, y, z)
        garment.rotation_euler = (0, 0, math.radians(rot_z_deg))

        placements.append(
            {
                "garment_id": garment_info["garment_id"],
                "category": garment_info["category"],
                "position": [round(x, 4), round(y, 4), round(z, 4)],
                "rotation_z": round(rot_z_deg, 1),
                "stack_order": i,
            }
        )

        print(
            f"    [{i + 1}/{stack_count}] {garment_info['garment_id']} "
            f"pos=({x:.3f}, {y:.3f}, {z:.3f}) rot_z={rot_z_deg:.1f}deg"
        )

    # 6. Add cloth modifier to all garments
    for garment in garment_objs:
        apply_cloth_preset(garment, preset_name)

    # 7. Bake simulation
    t0 = time.time()
    bake_simulation(frame_count)
    bake_time = time.time() - t0
    print(f"    bake: {bake_time:.1f}s ({frame_count} frames, {stack_count} garments)")

    # 8. Export USDA
    export_usda(str(usda_path))
    print(f"    exported: {usda_path.relative_to(output_dir)}")

    # 9. Write metadata
    metadata = {
        "mode": "stack",
        "furniture": furniture_path.name,
        "cloth_preset": preset_name,
        "stack_count": stack_count,
        "garments": placements,
        "frames": frame_count,
        "seed": seed,
        "bake_time_s": round(bake_time, 1),
    }
    write_metadata(str(json_path), metadata)

    return metadata


# =============================================================================
# Main
# =============================================================================


def main():
    argv = parse_blender_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Cloth stack simulation: multiple garments piling onto furniture (Blender)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to sim_config.json for default paths (portable across machines)",
    )
    parser.add_argument(
        "--furniture-dir",
        type=str,
        default=None,
        help="Directory containing furniture .blend files",
    )
    parser.add_argument(
        "--garment-dir",
        type=str,
        default=None,
        help="Root garment dataset directory (e.g. _Maria_Set)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for simulation results",
    )
    parser.add_argument(
        "--furniture",
        type=str,
        default=None,
        help="Specific furniture .blend filename (e.g. Chair_01.blend)",
    )
    parser.add_argument(
        "--garment-category",
        type=str,
        default=None,
        help="Filter garments to a specific category (e.g. dress_sleeveless_2550)",
    )
    parser.add_argument(
        "--cloth-preset",
        type=str,
        default="Cotton",
        help="Cloth preset: Silk, Cotton, Denim, Leather, Rubber, or 'all'",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of stacks per furniture-preset combo",
    )
    parser.add_argument(
        "--min-stack",
        type=int,
        default=MIN_STACK_DEFAULT,
        help=f"Minimum garments per stack (default: {MIN_STACK_DEFAULT})",
    )
    parser.add_argument(
        "--max-stack",
        type=int,
        default=MAX_STACK_DEFAULT,
        help=f"Maximum garments per stack (default: {MAX_STACK_DEFAULT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"Simulation frame count (default: {DEFAULT_FRAMES})",
    )
    # Apply config defaults before parsing (CLI args override config)
    config = load_sim_config(extract_config_path(argv))
    apply_config_defaults(parser, config)
    args = parser.parse_args(argv)

    # Validate required paths
    for name in ("furniture_dir", "garment_dir", "output_dir"):
        if not getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is required (or set in config)")

    print("=" * 70)
    print("CLOTH STACK SIMULATION (Blender)")
    print("=" * 70)

    # Discover furniture
    furniture_files = find_furniture(args.furniture_dir, args.furniture)
    print(f"Furniture: {len(furniture_files)} files")
    for f in furniture_files:
        print(f"  {f.name}")

    # Discover garments
    garments = find_prepped_garments(args.garment_dir, args.garment_category)
    print(f"Garments pool: {len(garments)} prepped meshes")
    if len(garments) < args.min_stack:
        print(f"ERROR: Need at least {args.min_stack} garments for stacking, found {len(garments)}")
        sys.exit(1)

    # Presets
    presets = resolve_presets(args.cloth_preset)
    print(f"Presets: {presets}")
    print(f"Stack size: {args.min_stack}-{args.max_stack} garments")

    total = len(furniture_files) * len(presets) * args.num_samples
    print(
        f"Total simulations: {total} "
        f"({len(furniture_files)} furniture x {len(presets)} presets "
        f"x {args.num_samples} samples)"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    completed = 0
    skipped = 0
    errors = 0

    for furn_path in furniture_files:
        for preset in presets:
            for sample_i in range(1, args.num_samples + 1):
                seed = make_seed(args.seed, furn_path.stem, preset, "stack", sample_i)

                result = run_stack_sim(
                    furniture_path=furn_path,
                    garments_pool=garments,
                    preset_name=preset,
                    output_dir=output_dir,
                    seed=seed,
                    sample_idx=sample_i,
                    min_stack=args.min_stack,
                    max_stack=args.max_stack,
                    frame_count=args.frames,
                )

                if result.get("skipped"):
                    skipped += 1
                elif result.get("error"):
                    errors += 1
                else:
                    completed += 1

                done = completed + skipped + errors
                if done % 5 == 0 or done == total:
                    elapsed = time.time() - t_start
                    print(
                        f"\n  Progress: {done}/{total} "
                        f"({completed} done, {skipped} skip, {errors} err) "
                        f"[{elapsed:.0f}s]\n"
                    )

    duration = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"DONE -- {completed} stack simulations in {duration:.1f}s")
    print(f"  Completed: {completed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Output:    {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
