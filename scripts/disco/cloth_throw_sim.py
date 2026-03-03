"""
Cloth Throw Simulation (Blender)
=================================
Grab-and-throw pipeline: an Empty + Hook modifier pins a portion of the garment,
animates a throw trajectory, then releases so the cloth flies free and settles
on furniture or the floor.

4 throw styles:
  swing_up   -- start below furniture, arc upward over it
  swing_side -- horizontal sweep from one side across to the other
  swing_over -- behind -> overhead -> forward arc
  toss       -- quick upward flick from near the furniture

Output pipeline: bake -> .blend -> .abc -> .usda -> .json

Usage (headless):
  blender --background --python scripts/disco/cloth_throw_sim.py -- \
      --furniture-dir "D:/_blender/_myBlender/SimulationWork/seedAssets/scenes" \
      --garment-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Set" \
      --output-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Sims" \
      --cloth-preset Cotton --num-samples 10

  # Specific throw style + power:
  blender --background --python scripts/disco/cloth_throw_sim.py -- \
      --throw-style swing_up --power 0.7 --cloth-preset Cotton

  # Random style (default):
  blender --background --python scripts/disco/cloth_throw_sim.py -- \
      --throw-style random --cloth-preset all --num-samples 20
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
    create_vertex_group,
    export_alembic,
    extract_config_path,
    find_furniture,
    find_prepped_garments,
    furniture_bbox,
    furniture_center,
    garment_height,
    import_garment,
    load_sim_config,
    parse_blender_args,
    prepare_garment_mesh,
    resolve_presets,
    run_sim_loop,
    save_blend,
    write_metadata,
)
from mathutils import Vector

# =============================================================================
# Constants
# =============================================================================

DEFAULT_FRAMES = 250
THROW_STYLES = ["swing_up", "swing_side", "swing_over", "toss"]
GRAB_GROUP = "grab"
GRAB_RADIUS = 0.08  # metres
TOP_PERCENT = 0.20  # top 20% of vertices for grab centroid

# Frame timing (at 24 fps)
HOLD_END = 5  # frames 1-5: garment hangs from grab
WINDUP_END = 15  # frames 6-15: Empty moves to start position
THROW_END = 23  # frames 16-23: throw arc
RELEASE_FRAME = 20  # frame ~20: pin_stiffness -> 0


# =============================================================================
# Grab point selection
# =============================================================================


def find_grab_centroid(garment: bpy.types.Object) -> Vector:
    """Return the centroid of the top *TOP_PERCENT* vertices (shoulder area)."""
    mesh = garment.data
    world_coords = [garment.matrix_world @ v.co for v in mesh.vertices]
    if not world_coords:
        return Vector((0, 0, 0))

    z_vals = sorted((co.z for co in world_coords), reverse=True)
    cutoff_idx = max(1, int(len(z_vals) * TOP_PERCENT))
    z_threshold = z_vals[cutoff_idx - 1]

    top_verts = [co for co in world_coords if co.z >= z_threshold]
    cx = sum(co.x for co in top_verts) / len(top_verts)
    cy = sum(co.y for co in top_verts) / len(top_verts)
    cz = sum(co.z for co in top_verts) / len(top_verts)
    return Vector((cx, cy, cz))


# =============================================================================
# Throw trajectory generation
# =============================================================================


def _throw_swing_up(
    grab_pos: Vector,
    furn_center: Vector,
    furn_max_z: float,
    power: float,
    direction_rad: float,
) -> list[tuple[int, Vector]]:
    """swing_up: start below furniture, arc upward over it."""
    offset = 0.8 + power * 0.6
    arc_height = 0.5 + power * 1.0

    start = Vector((
        furn_center.x + offset * math.cos(direction_rad),
        furn_center.y + offset * math.sin(direction_rad),
        furn_max_z - 0.3,
    ))
    mid = Vector((
        furn_center.x + 0.3 * math.cos(direction_rad),
        furn_center.y + 0.3 * math.sin(direction_rad),
        furn_max_z + arc_height,
    ))
    end = Vector((
        furn_center.x - 0.2 * math.cos(direction_rad),
        furn_center.y - 0.2 * math.sin(direction_rad),
        furn_max_z + arc_height * 0.5,
    ))
    return [
        (1, grab_pos),
        (HOLD_END, grab_pos),
        (WINDUP_END, start),
        ((WINDUP_END + THROW_END) // 2, mid),
        (THROW_END, end),
    ]


def _throw_swing_side(
    grab_pos: Vector,
    furn_center: Vector,
    furn_max_z: float,
    power: float,
    direction_rad: float,
) -> list[tuple[int, Vector]]:
    """swing_side: horizontal sweep from one side across to the other."""
    sweep = 1.0 + power * 0.8
    height = furn_max_z + 0.3 + power * 0.3
    perp = direction_rad + math.pi / 2

    start = Vector((
        furn_center.x + sweep * math.cos(perp),
        furn_center.y + sweep * math.sin(perp),
        height,
    ))
    end = Vector((
        furn_center.x - sweep * math.cos(perp),
        furn_center.y - sweep * math.sin(perp),
        height + 0.2,
    ))
    return [
        (1, grab_pos),
        (HOLD_END, grab_pos),
        (WINDUP_END, start),
        (THROW_END, end),
    ]


def _throw_swing_over(
    grab_pos: Vector,
    furn_center: Vector,
    furn_max_z: float,
    power: float,
    direction_rad: float,
) -> list[tuple[int, Vector]]:
    """swing_over: behind -> overhead -> forward arc."""
    behind_dist = 0.8 + power * 0.5
    overhead_h = 1.0 + power * 1.0
    forward_dist = 0.4 + power * 0.4

    behind = Vector((
        furn_center.x + behind_dist * math.cos(direction_rad),
        furn_center.y + behind_dist * math.sin(direction_rad),
        furn_max_z + 0.2,
    ))
    overhead = Vector((
        furn_center.x,
        furn_center.y,
        furn_max_z + overhead_h,
    ))
    forward = Vector((
        furn_center.x - forward_dist * math.cos(direction_rad),
        furn_center.y - forward_dist * math.sin(direction_rad),
        furn_max_z + overhead_h * 0.4,
    ))
    return [
        (1, grab_pos),
        (HOLD_END, grab_pos),
        (WINDUP_END, behind),
        ((WINDUP_END + THROW_END) // 2, overhead),
        (THROW_END, forward),
    ]


def _throw_toss(
    grab_pos: Vector,
    furn_center: Vector,
    furn_max_z: float,
    power: float,
    direction_rad: float,
) -> list[tuple[int, Vector]]:
    """toss: quick upward flick from near the furniture."""
    near_dist = 0.3 + power * 0.2
    toss_height = 0.4 + power * 0.6

    near = Vector((
        furn_center.x + near_dist * math.cos(direction_rad),
        furn_center.y + near_dist * math.sin(direction_rad),
        furn_max_z + 0.1,
    ))
    up = Vector((
        furn_center.x + near_dist * 0.5 * math.cos(direction_rad),
        furn_center.y + near_dist * 0.5 * math.sin(direction_rad),
        furn_max_z + toss_height,
    ))
    return [
        (1, grab_pos),
        (HOLD_END, grab_pos),
        (WINDUP_END, near),
        (THROW_END, up),
    ]


STYLE_FN = {
    "swing_up": _throw_swing_up,
    "swing_side": _throw_swing_side,
    "swing_over": _throw_swing_over,
    "toss": _throw_toss,
}


# =============================================================================
# Hook + throw setup
# =============================================================================


def setup_throw(
    garment: bpy.types.Object,
    style: str,
    power: float,
    direction_deg: float,
    furn_min: Vector,
    furn_max: Vector,
) -> dict:
    """Set up the grab-and-throw mechanism on *garment*.

    1. Find grab centroid (top vertices).
    2. Create vertex group + Empty + Hook modifier.
    3. Animate the Empty along the style trajectory.
    4. Keyframe pin_stiffness release.

    Returns metadata dict.
    """
    scene = bpy.context.scene
    center = furniture_center(furn_min, furn_max)
    direction_rad = math.radians(direction_deg)

    # --- Grab point ---
    grab_pos = find_grab_centroid(garment)

    # --- Vertex group ---
    vg, n_pinned = create_vertex_group(garment, GRAB_GROUP, grab_pos, GRAB_RADIUS)
    if n_pinned == 0:
        # Retry with larger radius
        vg, n_pinned = create_vertex_group(garment, GRAB_GROUP, grab_pos, GRAB_RADIUS * 3)

    # --- Empty at grab point ---
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=grab_pos)
    empty = bpy.context.active_object
    empty.name = "Throw_Empty"

    # --- Hook modifier (before Cloth) ---
    bpy.context.view_layer.objects.active = garment
    garment.select_set(True)
    hook_mod = garment.modifiers.new("Hook_Grab", "HOOK")
    hook_mod.object = empty
    hook_mod.vertex_group = GRAB_GROUP
    garment.select_set(False)

    # --- Generate trajectory keyframes ---
    fn = STYLE_FN[style]
    keyframes = fn(grab_pos, center, furn_max.z, power, direction_rad)

    for frame, pos in keyframes:
        scene.frame_set(frame)
        empty.location = pos
        empty.keyframe_insert(data_path="location", frame=frame)

    # Linear interpolation for consistent motion
    if empty.animation_data and empty.animation_data.action:
        for fcurve in empty.animation_data.action.fcurves:
            for kfp in fcurve.keyframe_points:
                kfp.interpolation = "LINEAR"

    scene.frame_set(1)

    return {
        "throw_style": style,
        "throw_power": round(power, 2),
        "throw_direction_deg": round(direction_deg, 1),
        "grab_point": [round(grab_pos.x, 4), round(grab_pos.y, 4), round(grab_pos.z, 4)],
        "grab_radius": GRAB_RADIUS,
        "pinned_vertices": n_pinned,
        "release_frame": RELEASE_FRAME,
    }


def apply_throw_cloth(garment: bpy.types.Object, preset_name: str):
    """Apply cloth modifier with pin group and animated release.

    The Hook modifier must already exist on *garment*.  The Cloth modifier
    is placed after Hook in the modifier stack.
    """
    apply_cloth_preset(garment, preset_name)
    cloth_mod = garment.modifiers.get("Cloth")
    if not cloth_mod:
        return

    # Pin the grab group
    cloth_mod.settings.vertex_group_mass = GRAB_GROUP

    # Ensure Hook is above Cloth in modifier stack
    # Move Cloth to the end (it was just added, so it is already last)
    # Move Hook before Cloth by moving it up
    hook_mod = garment.modifiers.get("Hook_Grab")
    if hook_mod:
        # In Blender 4.x, use modifier_move_to_index
        hook_idx = list(garment.modifiers).index(hook_mod)
        cloth_idx = list(garment.modifiers).index(cloth_mod)
        if hook_idx > cloth_idx:
            bpy.context.view_layer.objects.active = garment
            bpy.ops.object.modifier_move_to_index(modifier=hook_mod.name, index=cloth_idx)

    # Keyframe pin_stiffness: pinned until release, then free
    scene = bpy.context.scene
    scene.frame_set(RELEASE_FRAME - 1)
    cloth_mod.settings.pin_stiffness = 1.0
    cloth_mod.settings.keyframe_insert("pin_stiffness", frame=RELEASE_FRAME - 1)

    scene.frame_set(RELEASE_FRAME)
    cloth_mod.settings.pin_stiffness = 0.0
    cloth_mod.settings.keyframe_insert("pin_stiffness", frame=RELEASE_FRAME)

    scene.frame_set(1)


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
    throw_style: str = "random",
    power: float | None = None,
) -> dict:
    """Execute one cloth throw simulation and export results."""
    rng = random.Random(seed)

    garment_id = garment_info["garment_id"]
    garment_category = garment_info["category"]
    furniture_name = furniture_path.stem

    # Pick style
    if throw_style == "random":
        style = rng.choice(THROW_STYLES)
    else:
        style = throw_style

    # Pick power
    if power is None:
        pwr = rng.uniform(0.3, 1.0)
    else:
        pwr = power

    # Random direction around furniture
    direction_deg = rng.uniform(0, 360)

    # Output paths
    sim_dir = output_dir / furniture_name / "throw" / preset_name
    sim_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{garment_id}_{sample_idx:03d}"
    blend_path = sim_dir / f"{base_name}.blend"
    abc_path = sim_dir / f"{base_name}.abc"
    usda_path = sim_dir / f"{base_name}.usda"
    json_path = sim_dir / f"{base_name}.json"

    if usda_path.exists() and json_path.exists():
        print(f"  SKIP (exists): {usda_path.relative_to(output_dir)}")
        return {"skipped": True}

    print(f"  SIM [throw/{style}]: {furniture_name} / {preset_name} / {garment_id} #{sample_idx}")

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

    # 4. Compute furniture bbox
    furn_min, furn_max = furniture_bbox(furniture_objs)

    # 5. Position garment above furniture initially (will be moved by Hook)
    center = furniture_center(furn_min, furn_max)
    drop_z = furn_max.z + garment_height(garment) + 0.3
    garment.location = (center.x, center.y, drop_z)
    garment.rotation_euler = (0, 0, math.radians(rng.uniform(0, 360)))

    # 6. Add floor plane (catches garments that miss furniture)
    add_floor_plane(furn_min.z - 0.01)

    # 7. Add collision to furniture
    add_collision(furniture_objs)

    # 8. Set up throw mechanism (Empty + Hook + vertex group)
    throw_meta = setup_throw(garment, style, pwr, direction_deg, furn_min, furn_max)
    print(
        f"    style={style} power={pwr:.2f} dir={direction_deg:.0f}deg "
        f"pinned={throw_meta['pinned_vertices']} release_frame={RELEASE_FRAME}"
    )

    # 9. Apply cloth with pin group and animated release
    apply_throw_cloth(garment, preset_name)

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
        "mode": "throw",
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
        **throw_meta,
    }
    write_metadata(str(json_path), metadata)

    return metadata


# =============================================================================
# Main
# =============================================================================


def main():
    argv = parse_blender_args(sys.argv)

    parser = argparse.ArgumentParser(description="Cloth throw simulation: grab-and-throw garments at furniture (Blender)")
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
    parser.add_argument(
        "--throw-style",
        type=str,
        default="random",
        choices=["random"] + THROW_STYLES,
        help="Throw style: random, swing_up, swing_side, swing_over, toss",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=None,
        help="Throw power 0.3-1.0 (default: random)",
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
    print(f"CLOTH THROW SIMULATION (Blender) -- style: {args.throw_style}")
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

    # Build sim_fn with throw params baked in
    def sim_fn(furniture_path, garment_info, preset_name, output_dir, seed, sample_idx):
        return run_single_sim(
            furniture_path=furniture_path,
            garment_info=garment_info,
            preset_name=preset_name,
            output_dir=output_dir,
            seed=seed,
            sample_idx=sample_idx,
            frame_count=args.frames,
            throw_style=args.throw_style,
            power=args.power,
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
