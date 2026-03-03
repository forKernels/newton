"""
Lego Drop / Crash Simulation (Blender)
=======================================
Drops Lego models from height, letting bricks scatter with rigid body dynamics.
Each brick is separated and given active rigid body physics.  Generates training
data by combining: every Lego model x material preset x drop samples.

Modes:
  drop  -- model released from rest above surface (gravity only)
  crash -- model dropped from higher with random tilt for dramatic breakup

The .blend files should contain Lego-style models where bricks are either
separate objects or a single mesh with disconnected parts (separated
automatically by loose parts).

Usage (headless):
  # Drop mode (default) onto ground:
  blender --background --python scripts/disco/lego_drop_sim.py -- \
      --model-dir "path/to/lego_models" \
      --output-dir "path/to/output" \
      --material ABS \
      --num-samples 10

  # Crash mode:
  blender --background --python scripts/disco/lego_drop_sim.py -- \
      --mode crash \
      --model-dir "path/to/lego_models" \
      --output-dir "path/to/output" \
      --material all --num-samples 50

  # Drop onto furniture:
  blender --background --python scripts/disco/lego_drop_sim.py -- \
      --model-dir "path/to/lego_models" \
      --furniture-dir "path/to/furniture" \
      --output-dir "path/to/output" \
      --num-samples 5

  # Specific model + furniture:
  blender --background --python scripts/disco/lego_drop_sim.py -- \
      --model-dir "path/to/lego_models" \
      --model "spaceship.blend" \
      --furniture "Table_01.blend" \
      --furniture-dir "path/to/furniture" \
      --output-dir "path/to/output" \
      --material ABS --num-samples 5
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
    bake_simulation,
    clear_scene,
    export_usda,
    extract_config_path,
    find_furniture,
    furniture_bbox,
    load_sim_config,
    make_seed,
    parse_blender_args,
    write_metadata,
)
from lego_sim_utils import (
    LEGO_MATERIALS,
    add_rigid_body_active,
    add_rigid_body_passive,
    apply_transforms,
    ensure_rigid_body_world,
    estimate_brick_mass,
    find_lego_models,
    import_lego_model,
    model_bbox,
    model_height,
    move_bricks,
    resolve_lego_materials,
    separate_bricks,
    setup_ground_plane,
)
from mathutils import Vector

# =============================================================================
# Constants
# =============================================================================

DEFAULT_FRAMES = 250
DROP_GAP_M = 0.5  # metres above surface for drop mode
CRASH_GAP_M = 1.5  # metres above surface for crash mode (higher = more dramatic)
CRASH_TILT_MIN = 15  # degrees of tilt for crash mode
CRASH_TILT_MAX = 45

# Sentinel path for "use ground plane instead of furniture"
GROUND_SENTINEL = Path("__ground__")


# =============================================================================
# Drop mode -- position model directly above surface
# =============================================================================


def position_drop(
    bricks: list[bpy.types.Object],
    surface_z: float,
    rng: random.Random,
    *,
    surface_min: Vector | None = None,
    surface_max: Vector | None = None,
) -> dict:
    """Position Lego model at rest above surface for a gravity drop.

    If surface_min/surface_max are provided (furniture), places within
    the furniture bounds.  Otherwise places near the origin (ground mode).

    Returns placement metadata dict.
    """
    bb_min, bb_max = model_bbox(bricks)
    height = bb_max.z - bb_min.z

    # Target XY
    if surface_min is not None and surface_max is not None:
        x_span = surface_max.x - surface_min.x
        y_span = surface_max.y - surface_min.y
        margin_x = x_span * BBOX_MARGIN
        margin_y = y_span * BBOX_MARGIN
        target_x = rng.uniform(surface_min.x + margin_x, surface_max.x - margin_x)
        target_y = rng.uniform(surface_min.y + margin_y, surface_max.y - margin_y)
    else:
        target_x = rng.uniform(-0.3, 0.3)
        target_y = rng.uniform(-0.3, 0.3)

    target_z = surface_z + height + DROP_GAP_M

    # Compute offset from current model position to target
    center = (bb_min + bb_max) / 2.0
    offset = Vector((target_x - center.x, target_y - center.y, target_z - bb_min.z))

    # Random Z rotation
    rot_z_deg = rng.uniform(0, 360)
    rot_z_rad = math.radians(rot_z_deg)

    move_bricks(bricks, offset, rotation_euler=(0, 0, rot_z_rad))

    return {
        "drop_position": [round(target_x, 4), round(target_y, 4), round(target_z, 4)],
        "drop_rotation_z": round(rot_z_deg, 1),
    }


# =============================================================================
# Crash mode -- higher drop with tilt for dramatic breakup
# =============================================================================


def position_crash(
    bricks: list[bpy.types.Object],
    surface_z: float,
    rng: random.Random,
    *,
    surface_min: Vector | None = None,
    surface_max: Vector | None = None,
) -> dict:
    """Position Lego model above surface with random tilt for a crash landing.

    Higher drop height and random tilt (around X and Y) causes the model
    to hit at an angle, producing a more dramatic breakup.

    Returns placement metadata dict.
    """
    bb_min, bb_max = model_bbox(bricks)
    height = bb_max.z - bb_min.z

    # Target XY (same logic as drop)
    if surface_min is not None and surface_max is not None:
        x_span = surface_max.x - surface_min.x
        y_span = surface_max.y - surface_min.y
        margin_x = x_span * BBOX_MARGIN
        margin_y = y_span * BBOX_MARGIN
        target_x = rng.uniform(surface_min.x + margin_x, surface_max.x - margin_x)
        target_y = rng.uniform(surface_min.y + margin_y, surface_max.y - margin_y)
    else:
        target_x = rng.uniform(-0.3, 0.3)
        target_y = rng.uniform(-0.3, 0.3)

    target_z = surface_z + height + CRASH_GAP_M

    # Compute offset
    center = (bb_min + bb_max) / 2.0
    offset = Vector((target_x - center.x, target_y - center.y, target_z - bb_min.z))

    # Random tilt around X and Y, plus Z rotation
    tilt_x_deg = rng.uniform(CRASH_TILT_MIN, CRASH_TILT_MAX) * rng.choice([-1, 1])
    tilt_y_deg = rng.uniform(CRASH_TILT_MIN, CRASH_TILT_MAX) * rng.choice([-1, 1])
    rot_z_deg = rng.uniform(0, 360)

    move_bricks(
        bricks,
        offset,
        rotation_euler=(
            math.radians(tilt_x_deg),
            math.radians(tilt_y_deg),
            math.radians(rot_z_deg),
        ),
    )

    return {
        "crash_position": [round(target_x, 4), round(target_y, 4), round(target_z, 4)],
        "crash_tilt_x": round(tilt_x_deg, 1),
        "crash_tilt_y": round(tilt_y_deg, 1),
        "crash_rotation_z": round(rot_z_deg, 1),
    }


# =============================================================================
# Single simulation run
# =============================================================================


def run_single_sim(
    model_info: dict,
    surface_path: Path,
    material_name: str,
    output_dir: Path,
    seed: int,
    sample_idx: int,
    *,
    mode: str = "drop",
    frame_count: int = DEFAULT_FRAMES,
) -> dict:
    """Execute one Lego RBD simulation (drop or crash) and export results."""
    rng = random.Random(seed)

    model_id = model_info["model_id"]
    blend_path = model_info["blend_path"]
    surface_name = "ground" if surface_path == GROUND_SENTINEL else surface_path.stem
    mat_props = LEGO_MATERIALS[material_name]

    # Output paths -- {output}/{surface}/{mode}/{material}/{model}_{idx}.usda
    sim_dir = output_dir / surface_name / mode / material_name
    sim_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{model_id}_{sample_idx:03d}"
    usda_path = sim_dir / f"{base_name}.usda"
    json_path = sim_dir / f"{base_name}.json"

    if usda_path.exists() and json_path.exists():
        print(f"  SKIP (exists): {usda_path.relative_to(output_dir)}")
        return {"skipped": True}

    print(f"  SIM [{mode}]: {surface_name} / {material_name} / {model_id} #{sample_idx}")

    # 1. Clean scene
    clear_scene()

    # 2. Import furniture or set up ground
    surface_z = 0.0
    surface_min = None
    surface_max = None
    furniture_objs = []

    if surface_path != GROUND_SENTINEL:
        furniture_objs = append_furniture(str(surface_path))
        if not furniture_objs:
            print(f"    WARNING: No mesh objects in {surface_path.name}, skipping")
            return {"error": "no_furniture_meshes"}
        furn_min, furn_max = furniture_bbox(furniture_objs)
        surface_z = furn_max.z
        surface_min = furn_min
        surface_max = furn_max

    # Always create ground plane (catches fallen bricks)
    ground = setup_ground_plane()

    # 3. Import Lego model
    raw_objects = import_lego_model(blend_path)
    if not raw_objects:
        print(f"    WARNING: No mesh objects in {model_id}, skipping")
        return {"error": "no_model_meshes"}

    # 4. Apply rotation/scale transforms, then separate into bricks
    apply_transforms(raw_objects)
    bricks = separate_bricks(raw_objects)
    num_bricks = len(bricks)
    print(f"    bricks: {num_bricks}")

    if not bricks:
        print(f"    WARNING: No bricks after separation for {model_id}, skipping")
        return {"error": "no_bricks"}

    # 5. Position model above surface (mode-dependent)
    if mode == "crash":
        placement = position_crash(
            bricks,
            surface_z,
            rng,
            surface_min=surface_min,
            surface_max=surface_max,
        )
        p = placement["crash_position"]
        print(
            f"    crash pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) "
            f"tilt=({placement['crash_tilt_x']:.1f}, {placement['crash_tilt_y']:.1f}) "
            f"rot_z={placement['crash_rotation_z']:.1f}deg"
        )
    else:
        placement = position_drop(
            bricks,
            surface_z,
            rng,
            surface_min=surface_min,
            surface_max=surface_max,
        )
        p = placement["drop_position"]
        print(f"    drop pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) rot_z={placement['drop_rotation_z']:.1f}deg")

    # 6. Add rigid body physics
    # Active RBD on each brick (mass estimated from volume + material density)
    for brick in bricks:
        mass = estimate_brick_mass(brick, density=mat_props["density"])
        add_rigid_body_active(
            brick,
            mass=mass,
            friction=mat_props["friction"],
            restitution=mat_props["restitution"],
        )

    # Passive RBD on furniture (collision for cloth was already different)
    for fobj in furniture_objs:
        add_rigid_body_passive(fobj)

    # 7. Configure rigid body world
    ensure_rigid_body_world(frame_count)

    # 8. Bake simulation
    t0 = time.time()
    bake_simulation(frame_count)
    bake_time = time.time() - t0
    print(f"    bake: {bake_time:.1f}s ({frame_count} frames, {num_bricks} bricks)")

    # 9. Export USDA
    export_usda(str(usda_path))
    print(f"    exported: {usda_path.relative_to(output_dir)}")

    # 10. Write metadata JSON
    metadata = {
        "mode": mode,
        "surface": surface_name,
        "model": model_id,
        "material": material_name,
        "num_bricks": num_bricks,
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


def _apply_lego_config_defaults(parser, config: dict):
    """Set argparse defaults from config for Lego-specific keys."""
    defaults = {}
    for cfg_key, arg_dest in [
        ("lego_model_dir", "model_dir"),
        ("furniture_dir", "furniture_dir"),
        ("output_dir", "output_dir"),
    ]:
        if cfg_key in config:
            defaults[arg_dest] = config[cfg_key]
    if defaults:
        parser.set_defaults(**defaults)


def main():
    argv = parse_blender_args(sys.argv)

    parser = argparse.ArgumentParser(description="Lego drop/crash simulation: rigid body brick scatter (Blender)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to sim_config.json for default paths",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="drop",
        choices=["drop", "crash"],
        help="Simulation mode: 'drop' (gravity) or 'crash' (tilted dramatic breakup)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing Lego .blend files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model .blend filename (e.g. spaceship.blend)",
    )
    parser.add_argument(
        "--furniture-dir",
        type=str,
        default=None,
        help="Directory containing furniture .blend files (optional, uses ground if omitted)",
    )
    parser.add_argument(
        "--furniture",
        type=str,
        default=None,
        help="Specific furniture .blend filename (e.g. Table_01.blend)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for simulation results",
    )
    parser.add_argument(
        "--material",
        type=str,
        default="ABS",
        help="Material preset: ABS, Metal, Rubber, Wood, Glass, or 'all'",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples per model-surface-material combo",
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

    # Apply config defaults
    config = load_sim_config(extract_config_path(argv))
    _apply_lego_config_defaults(parser, config)
    args = parser.parse_args(argv)

    # Validate required paths
    if not args.model_dir:
        parser.error("--model-dir is required (or set lego_model_dir in config)")
    if not args.output_dir:
        parser.error("--output-dir is required (or set output_dir in config)")

    print("=" * 70)
    print(f"LEGO SIMULATION (Blender) -- mode: {args.mode.upper()}")
    print("=" * 70)

    # Discover models
    models = find_lego_models(args.model_dir, args.model)
    print(f"Models: {len(models)} files")
    for m in models:
        print(f"  {m['model_id']}")

    # Discover surfaces
    if args.furniture_dir:
        surfaces = find_furniture(args.furniture_dir, args.furniture)
        print(f"Surfaces: {len(surfaces)} furniture files")
        for s in surfaces:
            print(f"  {s.name}")
    else:
        surfaces = [GROUND_SENTINEL]
        print("Surfaces: ground plane")

    # Materials
    materials = resolve_lego_materials(args.material)
    print(f"Materials: {materials}")

    # Simulation loop
    output_dir = Path(args.output_dir)
    total = len(surfaces) * len(models) * len(materials) * args.num_samples
    print(
        f"Total simulations: {total} "
        f"({len(surfaces)} surfaces x {len(models)} models "
        f"x {len(materials)} materials x {args.num_samples} samples)"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    completed = 0
    skipped = 0
    errors = 0

    for surface_path in surfaces:
        for model_info in models:
            for material in materials:
                for sample_i in range(1, args.num_samples + 1):
                    seed = make_seed(
                        args.seed,
                        "ground" if surface_path == GROUND_SENTINEL else surface_path.stem,
                        model_info["model_id"],
                        material,
                        sample_i,
                    )

                    result = run_single_sim(
                        model_info=model_info,
                        surface_path=surface_path,
                        material_name=material,
                        output_dir=output_dir,
                        seed=seed,
                        sample_idx=sample_i,
                        mode=args.mode,
                        frame_count=args.frames,
                    )

                    if result.get("skipped"):
                        skipped += 1
                    elif result.get("error"):
                        errors += 1
                    else:
                        completed += 1

                    done = completed + skipped + errors
                    if done % 10 == 0 or done == total:
                        elapsed = time.time() - t_start
                        print(
                            f"\n  Progress: {done}/{total} "
                            f"({completed} done, {skipped} skip, {errors} err) "
                            f"[{elapsed:.0f}s]\n"
                        )

    duration = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"DONE -- {completed} simulations in {duration:.1f}s")
    print(f"  Completed: {completed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Output:    {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
