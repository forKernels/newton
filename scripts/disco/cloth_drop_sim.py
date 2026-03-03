"""
Cloth Drop Simulation (Blender)
================================
Drops garments from the Maria dataset onto furniture and records the cloth
simulation.  Generates training data by combining:
  every garment x every Blender cloth preset x furniture pieces.

The garment is centered directly above the furniture with only Z rotation
randomized.  Output pipeline: bake -> save .blend -> export .abc -> convert
.abc to .usda -> write .json metadata.

The prepped garments (*_sim_prep.obj) already have solidify applied and are in
metres.  Furniture lives as .blend files in the seedAssets/scenes/ directory.

Usage (headless):
  blender --background --python scripts/disco/cloth_drop_sim.py -- \
      --furniture-dir "D:/_blender/_myBlender/SimulationWork/seedAssets/scenes" \
      --garment-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Set" \
      --output-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Sims" \
      --cloth-preset Silk \
      --num-samples 10

  # All presets:
  blender --background --python scripts/disco/cloth_drop_sim.py -- \
      --cloth-preset all --num-samples 50 ...

  # Specific furniture + garment category:
  blender --background --python scripts/disco/cloth_drop_sim.py -- \
      --furniture "Chair_01.blend" \
      --garment-category dress_sleeveless_2550 \
      --cloth-preset Denim \
      --num-samples 5
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

# Blender script -- ensure the sibling module is importable
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
    export_alembic,
    extract_config_path,
    find_furniture,
    find_prepped_garments,
    furniture_bbox,
    furniture_center,
    garment_bottom_offset,
    import_garment,
    load_sim_config,
    parse_blender_args,
    prepare_garment_mesh,
    resolve_presets,
    run_sim_loop,
    save_blend,
    write_metadata,
)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_FRAMES = 50
DROP_GAP_M = 0.05  # metres above furniture top (low drop for natural drape)


# =============================================================================
# Drop positioning -- garment centred above furniture, Z rotation only
# =============================================================================


def position_drop(
    garment: bpy.types.Object,
    furn_min,
    furn_max,
    rng: random.Random,
) -> dict:
    """Position garment centred above furniture for a gravity drop.

    The garment is tilted 40-80 deg around Y (randomly +/-) so it tumbles
    naturally instead of dropping flat.  Rotation is applied first, then
    the garment is repositioned so its lowest point clears the furniture.
    Returns placement metadata dict.
    """
    center = furniture_center(furn_min, furn_max)

    rot_y_deg = rng.uniform(40, 80) * rng.choice([-1, 1])

    # Apply rotation first so bounding box reflects the tilt
    garment.rotation_euler = (0, math.radians(rot_y_deg), 0)
    garment.location = (0, 0, 0)
    bpy.context.view_layer.update()

    # Now compute bottom offset with the tilted orientation
    x = center.x
    y = center.y
    z = furn_max.z + DROP_GAP_M + garment_bottom_offset(garment)

    garment.location = (x, y, z)

    return {
        "drop_position": [round(x, 4), round(y, 4), round(z, 4)],
        "drop_rotation_y": round(rot_y_deg, 1),
    }


# =============================================================================
# Single simulation run
# =============================================================================


def run_single_sim(
    furniture_path: Path,
    garment_info: dict,
    preset_name: str,
    output_dir: Path,
    seed: int,
    sample_idx: int,
    *,
    frame_count: int = DEFAULT_FRAMES,
) -> dict:
    """Execute one cloth drop simulation and export results.

    Output pipeline: bake -> .blend -> .abc -> .usda -> .json
    """
    rng = random.Random(seed)

    garment_id = garment_info["garment_id"]
    garment_category = garment_info["category"]
    furniture_name = furniture_path.stem

    # Output paths  --  _Sims/{furniture}/drop/{preset}/{garment}_{idx}.*
    sim_dir = output_dir / furniture_name / "drop" / preset_name
    sim_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{garment_id}_{sample_idx:03d}"
    blend_path = sim_dir / f"{base_name}.blend"
    abc_path = sim_dir / f"{base_name}.abc"
    usda_path = sim_dir / f"{base_name}.usda"
    json_path = sim_dir / f"{base_name}.json"

    if usda_path.exists() and json_path.exists():
        print(f"  SKIP (exists): {usda_path.relative_to(output_dir)}")
        return {"skipped": True}

    print(f"  SIM [drop]: {furniture_name} / {preset_name} / {garment_id} #{sample_idx}")

    # 1. Clean scene
    clear_scene()

    # 2. Append furniture
    furniture_objs = append_furniture(str(furniture_path))
    if not furniture_objs:
        print(f"    WARNING: No mesh objects in {furniture_path.name}, skipping")
        return {"error": "no_furniture_meshes"}

    # 3. Import garment + subdivision + shade smooth
    garment = import_garment(garment_info["obj_path"])
    prepare_garment_mesh(garment)

    # 4. Compute furniture bbox + add floor
    furn_min, furn_max = furniture_bbox(furniture_objs)
    add_floor_plane(furn_min.z - 0.01, size=10.0)

    # 5. Position garment centred above furniture
    placement = position_drop(garment, furn_min, furn_max, rng)
    p = placement["drop_position"]
    print(f"    drop pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) rot_y={placement['drop_rotation_y']:.1f}deg")

    # 6. Add collision to furniture
    add_collision(furniture_objs)

    # 7. Add cloth modifier with preset
    apply_cloth_preset(garment, preset_name)

    # 8. Bake simulation
    t0 = time.time()
    bake_simulation(frame_count)
    bake_time = time.time() - t0
    print(f"    bake: {bake_time:.1f}s ({frame_count} frames)")

    # 9. Save .blend (preserves baked cloth cache for GUI playback)
    save_blend(str(blend_path))
    print(f"    saved: {blend_path.relative_to(output_dir)}")

    # 10. Export Alembic (captures cloth deformation)
    export_alembic(str(abc_path))
    print(f"    exported: {abc_path.relative_to(output_dir)}")

    # 11. Convert .abc -> .usda (roundtrip captures deformed mesh)
    convert_abc_to_usda(str(abc_path), str(usda_path))
    print(f"    converted: {usda_path.relative_to(output_dir)}")

    # 12. Write metadata JSON
    metadata = {
        "mode": "drop",
        "furniture": furniture_path.name,
        "garment": garment_id,
        "garment_category": garment_category,
        "cloth_preset": preset_name,
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

    parser = argparse.ArgumentParser(description="Cloth drop simulation: garments onto furniture (Blender)")
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
        help="Number of samples per garment-furniture-preset combo",
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
    print("CLOTH DROP SIMULATION (Blender)")
    print("=" * 70)

    # Discover furniture
    furniture_files = find_furniture(args.furniture_dir, args.furniture)
    print(f"Furniture: {len(furniture_files)} files")
    for f in furniture_files:
        print(f"  {f.name}")

    # Discover garments
    garments = find_prepped_garments(args.garment_dir, args.garment_category)
    print(f"Garments: {len(garments)} prepped meshes")
    if not garments:
        print("ERROR: No *_sim_prep.obj garments found. Run clean_cloth_dataset.py first.")
        sys.exit(1)

    # Presets
    presets = resolve_presets(args.cloth_preset)
    print(f"Presets: {presets}")

    # Build sim_fn with frame_count baked in
    def sim_fn(furniture_path, garment_info, preset_name, output_dir, seed, sample_idx):
        return run_single_sim(
            furniture_path=furniture_path,
            garment_info=garment_info,
            preset_name=preset_name,
            output_dir=output_dir,
            seed=seed,
            sample_idx=sample_idx,
            frame_count=args.frames,
        )

    run_sim_loop(
        furniture_files=furniture_files,
        garments=garments,
        presets=presets,
        num_samples=args.num_samples,
        base_seed=args.seed,
        output_dir=Path(args.output_dir),
        sim_fn=sim_fn,
    )


if __name__ == "__main__":
    main()
