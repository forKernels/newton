"""
Cloth Drop / Throw Simulation (Blender)
========================================
Drops or throws garments from the Maria dataset onto furniture and records the
cloth simulation as animated USDA.  Generates training data by combining:
  every garment x every Blender cloth preset x furniture pieces.

Modes:
  drop  -- garment released from rest above furniture (gravity only)
  throw -- garment launched with initial velocity toward furniture

The prepped garments (*_sim_prep.obj) already have solidify applied and are in
meters.  Furniture lives as .blend files in the seedAssets/scenes/ directory.

Usage (headless):
  # Drop mode (default):
  blender --background --python scripts/disco/cloth_drop_sim.py -- \
      --furniture-dir "D:/_blender/_myBlender/SimulationWork/seedAssets/scenes" \
      --garment-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Set" \
      --output-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Sims" \
      --cloth-preset Silk \
      --num-samples 10

  # Throw mode:
  blender --background --python scripts/disco/cloth_drop_sim.py -- \
      --mode throw \
      --furniture-dir "..." --garment-dir "..." --output-dir "..." \
      --cloth-preset all --num-samples 50 \
      --throw-speed 2.5

  # Run all presets, both modes:
  blender --background --python scripts/disco/cloth_drop_sim.py -- \
      --mode drop --cloth-preset all --num-samples 50 ...
  blender --background --python scripts/disco/cloth_drop_sim.py -- \
      --mode throw --cloth-preset all --num-samples 50 ...

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
    furniture_center,
    garment_height,
    import_garment,
    load_sim_config,
    parse_blender_args,
    resolve_presets,
    run_sim_loop,
    write_metadata,
)
from mathutils import Vector

# =============================================================================
# Constants
# =============================================================================

DEFAULT_FRAMES = 250
DROP_GAP_M = 0.3  # metres above furniture top

# Throw mode
THROW_SPEED_DEFAULT = 2.0  # m/s lateral speed
THROW_WINDUP_FRAMES = 5  # keyframed trajectory frames
THROW_LATERAL_OFFSET = 1.5  # metres away from furniture center
THROW_HEIGHT_EXTRA = 0.5  # extra height above furniture for throw arc


# =============================================================================
# Drop mode -- position garment directly above furniture
# =============================================================================


def position_drop(
    garment: bpy.types.Object,
    furn_min: Vector,
    furn_max: Vector,
    rng: random.Random,
) -> dict:
    """Position garment at rest above furniture for a gravity drop.

    Returns placement metadata dict.
    """
    x_span = furn_max.x - furn_min.x
    y_span = furn_max.y - furn_min.y
    margin_x = x_span * BBOX_MARGIN
    margin_y = y_span * BBOX_MARGIN

    x = rng.uniform(furn_min.x + margin_x, furn_max.x - margin_x)
    y = rng.uniform(furn_min.y + margin_y, furn_max.y - margin_y)
    z = furn_max.z + garment_height(garment) + DROP_GAP_M

    rot_z_deg = rng.uniform(0, 360)

    garment.location = (x, y, z)
    garment.rotation_euler = (0, 0, math.radians(rot_z_deg))

    return {
        "drop_position": [round(x, 4), round(y, 4), round(z, 4)],
        "drop_rotation_z": round(rot_z_deg, 1),
    }


# =============================================================================
# Throw mode -- keyframe garment trajectory to impart initial velocity
# =============================================================================


def position_throw(
    garment: bpy.types.Object,
    furn_min: Vector,
    furn_max: Vector,
    rng: random.Random,
    throw_speed: float,
    fps: int = 24,
) -> dict:
    """Position garment offset from furniture and keyframe a throw trajectory.

    The cloth solver picks up velocity from the mesh displacement between
    keyframed positions.  We keyframe the garment at a lateral offset at
    frame 1 and near the furniture center at frame THROW_WINDUP_FRAMES.
    After those frames the cloth physics takes over.

    Returns placement metadata dict.
    """
    center = furniture_center(furn_min, furn_max)

    # Random azimuth: direction the garment is thrown FROM
    azimuth_deg = rng.uniform(0, 360)
    azimuth_rad = math.radians(azimuth_deg)

    # Random elevation angle for the throw arc (-15 to +30 deg from horizontal)
    elev_deg = rng.uniform(-15, 30)
    elev_rad = math.radians(elev_deg)

    # Starting position: offset from furniture center along the azimuth
    start_x = center.x + THROW_LATERAL_OFFSET * math.cos(azimuth_rad)
    start_y = center.y + THROW_LATERAL_OFFSET * math.sin(azimuth_rad)
    start_z = furn_max.z + garment_height(garment) + THROW_HEIGHT_EXTRA

    # Target position: random point over the furniture top surface (with margin)
    x_span = furn_max.x - furn_min.x
    y_span = furn_max.y - furn_min.y
    margin_x = x_span * BBOX_MARGIN
    margin_y = y_span * BBOX_MARGIN
    target_x = rng.uniform(furn_min.x + margin_x, furn_max.x - margin_x)
    target_y = rng.uniform(furn_min.y + margin_y, furn_max.y - margin_y)

    # Z at target: compute from throw speed + elevation over the windup duration
    windup_seconds = THROW_WINDUP_FRAMES / fps
    # Horizontal displacement is from start to target
    dx = target_x - start_x
    dy = target_y - start_y
    horiz_dist = math.sqrt(dx * dx + dy * dy)
    # Scale speed so the garment covers the horizontal distance in windup time
    actual_speed = horiz_dist / max(windup_seconds, 0.01)
    # Vertical displacement from elevation angle
    vertical_delta = horiz_dist * math.tan(elev_rad)
    target_z = start_z + vertical_delta

    # Random Z rotation
    rot_z_deg = rng.uniform(0, 360)

    # Random tumble: slight rotation around X and Y for realism
    tumble_x_deg = rng.uniform(-20, 20)
    tumble_y_deg = rng.uniform(-20, 20)

    # --- Keyframe the trajectory ---
    scene = bpy.context.scene

    # Frame 1: start position
    scene.frame_set(1)
    garment.location = (start_x, start_y, start_z)
    garment.rotation_euler = (
        math.radians(tumble_x_deg),
        math.radians(tumble_y_deg),
        math.radians(rot_z_deg),
    )
    garment.keyframe_insert(data_path="location", frame=1)
    garment.keyframe_insert(data_path="rotation_euler", frame=1)

    # Frame N: target position (near furniture)
    scene.frame_set(THROW_WINDUP_FRAMES)
    garment.location = (target_x, target_y, target_z)
    # Add some tumble rotation during flight
    garment.rotation_euler = (
        math.radians(tumble_x_deg + rng.uniform(-10, 10)),
        math.radians(tumble_y_deg + rng.uniform(-10, 10)),
        math.radians(rot_z_deg + rng.uniform(-30, 30)),
    )
    garment.keyframe_insert(data_path="location", frame=THROW_WINDUP_FRAMES)
    garment.keyframe_insert(data_path="rotation_euler", frame=THROW_WINDUP_FRAMES)

    # Set interpolation to linear for consistent velocity
    if garment.animation_data and garment.animation_data.action:
        for fcurve in garment.animation_data.action.fcurves:
            for kfp in fcurve.keyframe_points:
                kfp.interpolation = "LINEAR"

    # Reset to frame 1 before simulation
    scene.frame_set(1)

    return {
        "throw_start": [round(start_x, 4), round(start_y, 4), round(start_z, 4)],
        "throw_target": [round(target_x, 4), round(target_y, 4), round(target_z, 4)],
        "throw_azimuth_deg": round(azimuth_deg, 1),
        "throw_elevation_deg": round(elev_deg, 1),
        "throw_speed_mps": round(actual_speed, 2),
        "throw_windup_frames": THROW_WINDUP_FRAMES,
        "throw_rotation_z": round(rot_z_deg, 1),
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
    mode: str = "drop",
    frame_count: int = DEFAULT_FRAMES,
    throw_speed: float = THROW_SPEED_DEFAULT,
) -> dict:
    """Execute one cloth simulation (drop or throw) and export results."""
    rng = random.Random(seed)

    garment_id = garment_info["garment_id"]
    garment_category = garment_info["category"]
    furniture_name = furniture_path.stem

    # Output paths  --  _Sims/{furniture}/{mode}/{preset}/{garment}_{idx}.usda
    sim_dir = output_dir / furniture_name / mode / preset_name
    sim_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{garment_id}_{sample_idx:03d}"
    usda_path = sim_dir / f"{base_name}.usda"
    json_path = sim_dir / f"{base_name}.json"

    if usda_path.exists() and json_path.exists():
        print(f"  SKIP (exists): {usda_path.relative_to(output_dir)}")
        return {"skipped": True}

    print(f"  SIM [{mode}]: {furniture_name} / {preset_name} / {garment_id} #{sample_idx}")

    # 1. Clean scene
    clear_scene()

    # 2. Append furniture
    furniture_objs = append_furniture(str(furniture_path))
    if not furniture_objs:
        print(f"    WARNING: No mesh objects in {furniture_path.name}, skipping")
        return {"error": "no_furniture_meshes"}

    # 3. Import garment
    garment = import_garment(garment_info["obj_path"])

    # 4. Compute furniture bbox
    furn_min, furn_max = furniture_bbox(furniture_objs)

    # 5. Position garment (mode-dependent)
    if mode == "throw":
        placement = position_throw(garment, furn_min, furn_max, rng, throw_speed)
        print(
            f"    throw from=({placement['throw_start'][0]:.2f}, "
            f"{placement['throw_start'][1]:.2f}, {placement['throw_start'][2]:.2f}) "
            f"az={placement['throw_azimuth_deg']:.0f}deg "
            f"speed={placement['throw_speed_mps']:.1f}m/s"
        )
    else:
        placement = position_drop(garment, furn_min, furn_max, rng)
        p = placement["drop_position"]
        print(f"    drop pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) rot_z={placement['drop_rotation_z']:.1f}deg")

    # 6. Add collision to furniture
    add_collision(furniture_objs)

    # 7. Add cloth modifier with preset
    apply_cloth_preset(garment, preset_name)

    # 8. Bake simulation
    t0 = time.time()
    bake_simulation(frame_count)
    bake_time = time.time() - t0
    print(f"    bake: {bake_time:.1f}s ({frame_count} frames)")

    # 9. Export USDA
    export_usda(str(usda_path))
    print(f"    exported: {usda_path.relative_to(output_dir)}")

    # 10. Write metadata JSON
    metadata = {
        "mode": mode,
        "furniture": furniture_path.name,
        "garment": garment_id,
        "garment_category": garment_category,
        "cloth_preset": preset_name,
        "frames": frame_count,
        "seed": seed,
        "bake_time_s": round(bake_time, 1),
        **placement,
    }
    write_metadata(str(json_path), metadata)

    return metadata


# =============================================================================
# Main
# =============================================================================


def main():
    argv = parse_blender_args(sys.argv)

    parser = argparse.ArgumentParser(description="Cloth drop/throw simulation: garments onto furniture (Blender)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to sim_config.json for default paths (portable across machines)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="drop",
        choices=["drop", "throw"],
        help="Simulation mode: 'drop' (gravity) or 'throw' (initial velocity)",
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
    # Throw-specific
    parser.add_argument(
        "--throw-speed",
        type=float,
        default=THROW_SPEED_DEFAULT,
        help=f"Throw speed in m/s (default: {THROW_SPEED_DEFAULT}, throw mode only)",
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
    print(f"CLOTH SIMULATION (Blender) -- mode: {args.mode.upper()}")
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

    # Build sim_fn with mode/frames/throw_speed baked in
    def sim_fn(furniture_path, garment_info, preset_name, output_dir, seed, sample_idx):
        return run_single_sim(
            furniture_path=furniture_path,
            garment_info=garment_info,
            preset_name=preset_name,
            output_dir=output_dir,
            seed=seed,
            sample_idx=sample_idx,
            mode=args.mode,
            frame_count=args.frames,
            throw_speed=args.throw_speed,
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
