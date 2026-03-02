"""
Cloth Drag Simulation (Blender)
================================
Simulates dragging garments across furniture surfaces.  A portion of the
garment is "grabbed" (pinned vertices) and keyframed to slide across the
surface while the rest drags behind with cloth physics.

Generates training data for cloth manipulation tasks where a robot grabs
part of a garment and drags it to a target position.

Output structure:
  _Sims/{furniture}/drag/{preset}/{garment_id}_{idx}.usda

Usage (headless):
  blender --background --python scripts/disco/cloth_drag_sim.py -- \
      --furniture-dir "D:/_blender/_myBlender/SimulationWork/seedAssets/scenes" \
      --garment-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Set" \
      --output-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Sims" \
      --cloth-preset Cotton --num-samples 10

  # Larger grab radius (grabs more of the garment):
  blender --background --python scripts/disco/cloth_drag_sim.py -- \
      --furniture-dir "..." --garment-dir "..." --output-dir "..." \
      --cloth-preset Silk --grab-radius 0.15 --num-samples 20

  # Specific furniture + garment category:
  blender --background --python scripts/disco/cloth_drag_sim.py -- \
      --furniture "Table_01.blend" \
      --garment-category tee_2300 \
      --cloth-preset Denim --num-samples 5
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

DEFAULT_FRAMES = 200
GRAB_RADIUS_DEFAULT = 0.10  # metres -- radius around grab point for pinned verts
DRAG_HEIGHT_M = 0.05  # hover height above furniture during drag
SETTLE_FRAMES = 30  # frames for garment to settle before drag starts
PIN_GROUP_NAME = "drag_pin"


# =============================================================================
# Pin group creation
# =============================================================================


def create_pin_group(
    garment: bpy.types.Object,
    grab_point: Vector,
    grab_radius: float,
) -> int:
    """Create a vertex group for vertices near the grab point.

    Returns the number of pinned vertices.
    """
    bpy.context.view_layer.objects.active = garment
    garment.select_set(True)

    # Remove existing pin group if present
    if PIN_GROUP_NAME in garment.vertex_groups:
        garment.vertex_groups.remove(garment.vertex_groups[PIN_GROUP_NAME])
    pin_group = garment.vertex_groups.new(name=PIN_GROUP_NAME)

    # Find vertices near grab point (in world space)
    mesh = garment.data
    pinned = []
    for v in mesh.vertices:
        world_co = garment.matrix_world @ v.co
        dist = (world_co - grab_point).length
        if dist <= grab_radius:
            pinned.append(v.index)

    if pinned:
        pin_group.add(pinned, 1.0, "REPLACE")

    garment.select_set(False)
    return len(pinned)


# =============================================================================
# Positioning and drag planning
# =============================================================================


def position_garment_flat(
    garment: bpy.types.Object,
    furn_min: Vector,
    furn_max: Vector,
    rng: random.Random,
) -> dict:
    """Position garment flat on the furniture surface.

    Returns placement metadata dict.
    """
    # Center on furniture surface
    cx = (furn_min.x + furn_max.x) / 2
    cy = (furn_min.y + furn_max.y) / 2
    z = furn_max.z + 0.01  # just above surface

    rot_z_deg = rng.uniform(0, 360)
    garment.location = (cx, cy, z)
    garment.rotation_euler = (0, 0, math.radians(rot_z_deg))

    return {
        "start_position": [round(cx, 4), round(cy, 4), round(z, 4)],
        "start_rotation_z": round(rot_z_deg, 1),
    }


def pick_grab_and_drag(
    garment: bpy.types.Object,
    furn_min: Vector,
    furn_max: Vector,
    rng: random.Random,
) -> tuple[Vector, Vector]:
    """Pick a grab point on the garment edge and a drag target.

    Returns (grab_point, drag_target) as world-space Vectors.
    """
    # Get garment bounding box in world space
    corners = [garment.matrix_world @ Vector(c) for c in garment.bound_box]
    g_min = Vector(
        (min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners))
    )
    g_max = Vector(
        (max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners))
    )
    g_cx = (g_min.x + g_max.x) / 2
    g_cy = (g_min.y + g_max.y) / 2
    g_z = (g_min.z + g_max.z) / 2

    # Pick a random edge direction (0=+X, 1=-X, 2=+Y, 3=-Y)
    edge = rng.randint(0, 3)

    if edge == 0:  # grab from +X edge, drag toward -X
        grab_x = g_max.x - 0.02
        grab_y = g_cy + rng.uniform(-0.05, 0.05)
        drag_dir = (-1, rng.uniform(-0.3, 0.3))
    elif edge == 1:  # grab from -X edge, drag toward +X
        grab_x = g_min.x + 0.02
        grab_y = g_cy + rng.uniform(-0.05, 0.05)
        drag_dir = (1, rng.uniform(-0.3, 0.3))
    elif edge == 2:  # grab from +Y edge, drag toward -Y
        grab_x = g_cx + rng.uniform(-0.05, 0.05)
        grab_y = g_max.y - 0.02
        drag_dir = (rng.uniform(-0.3, 0.3), -1)
    else:  # grab from -Y edge, drag toward +Y
        grab_x = g_cx + rng.uniform(-0.05, 0.05)
        grab_y = g_min.y + 0.02
        drag_dir = (rng.uniform(-0.3, 0.3), 1)

    grab_point = Vector((grab_x, grab_y, g_z))

    # Drag distance: 30-80% of garment extent in drag direction
    g_extent = max(g_max.x - g_min.x, g_max.y - g_min.y)
    drag_dist = rng.uniform(0.3, 0.8) * g_extent

    # Normalize drag direction and compute target
    drag_len = math.sqrt(drag_dir[0] ** 2 + drag_dir[1] ** 2)
    drag_dx = drag_dir[0] / drag_len * drag_dist
    drag_dy = drag_dir[1] / drag_len * drag_dist

    drag_target = Vector(
        (grab_x + drag_dx, grab_y + drag_dy, furn_max.z + DRAG_HEIGHT_M)
    )

    # Clamp target within furniture bounds (with margin)
    x_span = furn_max.x - furn_min.x
    y_span = furn_max.y - furn_min.y
    margin_x = x_span * BBOX_MARGIN
    margin_y = y_span * BBOX_MARGIN
    drag_target.x = max(
        furn_min.x + margin_x, min(furn_max.x - margin_x, drag_target.x)
    )
    drag_target.y = max(
        furn_min.y + margin_y, min(furn_max.y - margin_y, drag_target.y)
    )

    return grab_point, drag_target


# =============================================================================
# Keyframing
# =============================================================================


def keyframe_drag(
    garment: bpy.types.Object,
    grab_point: Vector,
    drag_target: Vector,
    settle_frames: int,
    total_frames: int,
):
    """Keyframe the garment location to create a drag motion.

    The garment stays still for settle_frames (cloth settles on surface),
    then moves from the grab point toward the drag target.
    """
    scene = bpy.context.scene
    start_loc = Vector(garment.location)

    # Compute the displacement from grab to target
    drag_delta = drag_target - grab_point

    # Frame 1: starting position (hold still for settling)
    scene.frame_set(1)
    garment.keyframe_insert(data_path="location", frame=1)

    # Frame settle_frames: still at start (cloth has settled)
    scene.frame_set(settle_frames)
    garment.keyframe_insert(data_path="location", frame=settle_frames)

    # Frame total_frames: moved by drag_delta
    scene.frame_set(total_frames)
    garment.location = (
        start_loc.x + drag_delta.x,
        start_loc.y + drag_delta.y,
        start_loc.z + drag_delta.z,
    )
    garment.keyframe_insert(data_path="location", frame=total_frames)

    # Set interpolation to linear for consistent drag speed
    if garment.animation_data and garment.animation_data.action:
        for fcurve in garment.animation_data.action.fcurves:
            for kfp in fcurve.keyframe_points:
                kfp.interpolation = "LINEAR"

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
    grab_radius: float = GRAB_RADIUS_DEFAULT,
) -> dict:
    """Execute one cloth drag simulation and export results."""
    rng = random.Random(seed)

    garment_id = garment_info["garment_id"]
    garment_category = garment_info["category"]
    furniture_name = furniture_path.stem

    # Output paths
    sim_dir = output_dir / furniture_name / "drag" / preset_name
    sim_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{garment_id}_{sample_idx:03d}"
    usda_path = sim_dir / f"{base_name}.usda"
    json_path = sim_dir / f"{base_name}.json"

    if usda_path.exists() and json_path.exists():
        print(f"  SKIP (exists): {usda_path.relative_to(output_dir)}")
        return {"skipped": True}

    print(
        f"  SIM [drag]: {furniture_name} / {preset_name} / "
        f"{garment_id} #{sample_idx}"
    )

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

    # 5. Position garment flat on furniture
    placement = position_garment_flat(garment, furn_min, furn_max, rng)

    # 6. Pick grab point and drag target
    grab_point, drag_target = pick_grab_and_drag(
        garment, furn_min, furn_max, rng
    )

    # 7. Create pin vertex group
    n_pinned = create_pin_group(garment, grab_point, grab_radius)
    if n_pinned == 0:
        # Increase radius and retry
        n_pinned = create_pin_group(garment, grab_point, grab_radius * 3)
        if n_pinned == 0:
            print(f"    WARNING: No vertices pinned for {garment_id}, skipping")
            return {"error": "no_pinned_vertices"}

    print(
        f"    grab at ({grab_point.x:.3f}, {grab_point.y:.3f}, "
        f"{grab_point.z:.3f})"
    )
    print(
        f"    drag to ({drag_target.x:.3f}, {drag_target.y:.3f}, "
        f"{drag_target.z:.3f})"
    )
    print(f"    pinned vertices: {n_pinned}")

    # 8. Add collision to furniture
    add_collision(furniture_objs)

    # 9. Add cloth modifier with preset and pin group
    apply_cloth_preset(garment, preset_name)
    cloth_mod = garment.modifiers.get("Cloth")
    if cloth_mod:
        cloth_mod.settings.vertex_group_mass = PIN_GROUP_NAME

    # 10. Keyframe drag motion
    keyframe_drag(
        garment, grab_point, drag_target, SETTLE_FRAMES, frame_count
    )

    # 11. Bake simulation
    t0 = time.time()
    bake_simulation(frame_count)
    bake_time = time.time() - t0
    print(f"    bake: {bake_time:.1f}s ({frame_count} frames)")

    # 12. Export USDA
    export_usda(str(usda_path))
    print(f"    exported: {usda_path.relative_to(output_dir)}")

    # 13. Write metadata
    metadata = {
        "mode": "drag",
        "furniture": furniture_path.name,
        "garment": garment_id,
        "garment_category": garment_category,
        "cloth_preset": preset_name,
        "frames": frame_count,
        "seed": seed,
        "bake_time_s": round(bake_time, 1),
        "grab_point": [
            round(grab_point.x, 4),
            round(grab_point.y, 4),
            round(grab_point.z, 4),
        ],
        "drag_target": [
            round(drag_target.x, 4),
            round(drag_target.y, 4),
            round(drag_target.z, 4),
        ],
        "grab_radius": grab_radius,
        "pinned_vertices": n_pinned,
        "settle_frames": SETTLE_FRAMES,
        **placement,
    }
    write_metadata(str(json_path), metadata)

    return metadata


# =============================================================================
# Main
# =============================================================================


def main():
    argv = parse_blender_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Cloth drag simulation: drag garments across furniture (Blender)"
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
        help="Specific furniture .blend filename (e.g. Table_01.blend)",
    )
    parser.add_argument(
        "--garment-category",
        type=str,
        default=None,
        help="Filter garments to a specific category (e.g. tee_2300)",
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
        "--grab-radius",
        type=float,
        default=GRAB_RADIUS_DEFAULT,
        help=f"Radius of grab area in metres (default: {GRAB_RADIUS_DEFAULT})",
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
    print("CLOTH DRAG SIMULATION (Blender)")
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
        print(
            "ERROR: No *_sim_prep.obj garments found. "
            "Run clean_cloth_dataset.py first."
        )
        sys.exit(1)

    # Presets
    presets = resolve_presets(args.cloth_preset)
    print(f"Presets: {presets}")

    # Build sim_fn with frame_count/grab_radius baked in
    def sim_fn(
        furniture_path, garment_info, preset_name, output_dir, seed, sample_idx
    ):
        return run_single_sim(
            furniture_path=furniture_path,
            garment_info=garment_info,
            preset_name=preset_name,
            output_dir=output_dir,
            seed=seed,
            sample_idx=sample_idx,
            frame_count=args.frames,
            grab_radius=args.grab_radius,
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
