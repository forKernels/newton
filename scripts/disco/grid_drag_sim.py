"""
Grid Cloth Drag Simulation (Blender)
=====================================
Places a grid cloth flat on furniture, pins one edge, and drags it across
the surface.  The pinned vertices slide while the rest of the cloth drags
behind with cloth physics.

Output pipeline: bake -> .blend -> .abc -> .usda -> .json

Usage (headless):
  blender --background --python scripts/disco/grid_drag_sim.py -- \
      --furniture-dir "D:/_blender/_myBlender/SimulationWork/seedAssets/scenes" \
      --output-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Sims" \
      --cloth-preset Cotton --num-samples 10 \
      --grid-size 1.0 --grid-cuts 30

  # Larger grab + longer drag:
  blender --background --python scripts/disco/grid_drag_sim.py -- \
      --grab-radius 0.12 --frames 100 --grid-size 1.5
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
    BBOX_MARGIN,
    add_collision,
    add_floor_plane,
    append_furniture,
    apply_cloth_preset,
    apply_config_defaults,
    bake_simulation,
    clear_scene,
    convert_abc_to_usda,
    create_cloth_grid,
    create_vertex_group,
    export_alembic,
    extract_config_path,
    find_furniture,
    furniture_bbox,
    load_sim_config,
    parse_blender_args,
    resolve_presets,
    save_blend,
    write_metadata,
)
from mathutils import Vector

# =============================================================================
# Constants
# =============================================================================

DEFAULT_FRAMES = 50
DEFAULT_GRID_SIZE = 1.0
DEFAULT_GRID_CUTS = 30
DEFAULT_THICKNESS = 0.001
DEFAULT_SUBDIV = 1
GRAB_RADIUS_DEFAULT = 0.08
SETTLE_FRAMES_DEFAULT = 20
DRAG_HEIGHT_M = 0.05  # hover height above surface during drag
PIN_GROUP_NAME = "drag_pin"


# =============================================================================
# Positioning and drag planning
# =============================================================================


def position_grid_flat(
    grid: bpy.types.Object,
    furn_min: Vector,
    furn_max: Vector,
    rng: random.Random,
) -> dict:
    """Position grid flat on the furniture surface with random Z rotation."""
    cx = (furn_min.x + furn_max.x) / 2
    cy = (furn_min.y + furn_max.y) / 2
    z = furn_max.z + 0.01

    rot_z_deg = rng.uniform(0, 360)
    grid.location = (cx, cy, z)
    grid.rotation_euler = (0, 0, math.radians(rot_z_deg))
    bpy.context.view_layer.update()

    return {
        "start_position": [round(cx, 4), round(cy, 4), round(z, 4)],
        "start_rotation_z": round(rot_z_deg, 1),
    }


def pick_grab_and_drag(
    grid: bpy.types.Object,
    furn_min: Vector,
    furn_max: Vector,
    rng: random.Random,
) -> tuple[Vector, Vector]:
    """Pick a grab point on the grid edge and a drag target.

    Returns (grab_point, drag_target) as world-space Vectors.
    """
    corners = [grid.matrix_world @ Vector(c) for c in grid.bound_box]
    g_min = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    g_max = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    g_cx = (g_min.x + g_max.x) / 2
    g_cy = (g_min.y + g_max.y) / 2
    g_z = (g_min.z + g_max.z) / 2

    # Pick a random edge direction (0=+X, 1=-X, 2=+Y, 3=-Y)
    edge = rng.randint(0, 3)

    if edge == 0:
        grab_x = g_max.x - 0.02
        grab_y = g_cy + rng.uniform(-0.05, 0.05)
        drag_dir = (-1, rng.uniform(-0.3, 0.3))
    elif edge == 1:
        grab_x = g_min.x + 0.02
        grab_y = g_cy + rng.uniform(-0.05, 0.05)
        drag_dir = (1, rng.uniform(-0.3, 0.3))
    elif edge == 2:
        grab_x = g_cx + rng.uniform(-0.05, 0.05)
        grab_y = g_max.y - 0.02
        drag_dir = (rng.uniform(-0.3, 0.3), -1)
    else:
        grab_x = g_cx + rng.uniform(-0.05, 0.05)
        grab_y = g_min.y + 0.02
        drag_dir = (rng.uniform(-0.3, 0.3), 1)

    grab_point = Vector((grab_x, grab_y, g_z))

    # Drag distance: 50-80% of grid extent
    g_extent = max(g_max.x - g_min.x, g_max.y - g_min.y)
    drag_dist = rng.uniform(0.5, 0.8) * g_extent

    drag_len = math.sqrt(drag_dir[0] ** 2 + drag_dir[1] ** 2)
    drag_dx = drag_dir[0] / drag_len * drag_dist
    drag_dy = drag_dir[1] / drag_len * drag_dist

    drag_target = Vector((grab_x + drag_dx, grab_y + drag_dy, furn_max.z + DRAG_HEIGHT_M))

    # Clamp within furniture bounds
    x_span = furn_max.x - furn_min.x
    y_span = furn_max.y - furn_min.y
    margin_x = x_span * BBOX_MARGIN
    margin_y = y_span * BBOX_MARGIN
    drag_target.x = max(furn_min.x + margin_x, min(furn_max.x - margin_x, drag_target.x))
    drag_target.y = max(furn_min.y + margin_y, min(furn_max.y - margin_y, drag_target.y))

    return grab_point, drag_target


def keyframe_drag(
    empty: bpy.types.Object,
    grab_point: Vector,
    drag_target: Vector,
    settle_frames: int,
    total_frames: int,
):
    """Keyframe the Empty to create a drag motion.

    Holds still for settle_frames (cloth settles), then slides to drag_target.
    """
    scene = bpy.context.scene

    # Frame 1: at grab point
    scene.frame_set(1)
    empty.location = grab_point
    empty.keyframe_insert(data_path="location", frame=1)

    # Hold still through settle
    scene.frame_set(settle_frames)
    empty.location = grab_point
    empty.keyframe_insert(data_path="location", frame=settle_frames)

    # Drag to target
    scene.frame_set(total_frames)
    empty.location = drag_target
    empty.keyframe_insert(data_path="location", frame=total_frames)

    # Linear interpolation
    if empty.animation_data and empty.animation_data.action:
        for fcurve in empty.animation_data.action.fcurves:
            for kfp in fcurve.keyframe_points:
                kfp.interpolation = "LINEAR"

    scene.frame_set(1)


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
    grab_radius: float = GRAB_RADIUS_DEFAULT,
    settle_frames: int = SETTLE_FRAMES_DEFAULT,
) -> dict:
    """Execute one grid cloth drag simulation and export results."""
    rng = random.Random(seed)
    furniture_name = furniture_path.stem

    # Output paths
    sim_dir = output_dir / furniture_name / "grid_drag" / preset_name
    sim_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"grid_{grid_size}m_{sample_idx:03d}"
    blend_path = sim_dir / f"{base_name}.blend"
    abc_path = sim_dir / f"{base_name}.abc"
    usda_path = sim_dir / f"{base_name}.usda"
    json_path = sim_dir / f"{base_name}.json"

    if usda_path.exists() and json_path.exists():
        print(f"  SKIP (exists): {usda_path.relative_to(output_dir)}")
        return {"skipped": True}

    print(f"  SIM [grid_drag]: {furniture_name} / {preset_name} #{sample_idx}")

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

    # 5. Position grid flat on furniture
    placement = position_grid_flat(grid, furn_min, furn_max, rng)

    # 6. Add collision to furniture
    add_collision(furniture_objs)

    # 7. Pick grab point and drag target
    grab_point, drag_target = pick_grab_and_drag(grid, furn_min, furn_max, rng)

    # 8. Create pin vertex group
    vg, n_pinned = create_vertex_group(grid, PIN_GROUP_NAME, grab_point, grab_radius)
    if n_pinned == 0:
        vg, n_pinned = create_vertex_group(grid, PIN_GROUP_NAME, grab_point, grab_radius * 3)
        if n_pinned == 0:
            print("    WARNING: No vertices pinned, skipping")
            return {"error": "no_pinned_vertices"}

    print(f"    grab at ({grab_point.x:.3f}, {grab_point.y:.3f}, {grab_point.z:.3f})")
    print(f"    drag to ({drag_target.x:.3f}, {drag_target.y:.3f}, {drag_target.z:.3f})")
    print(f"    pinned vertices: {n_pinned}")

    # 9. Create Empty at grab point for Hook modifier
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=grab_point)
    empty = bpy.context.active_object
    empty.name = "Drag_Empty"

    # 10. Add Hook modifier to grid (before Cloth)
    bpy.context.view_layer.objects.active = grid
    grid.select_set(True)
    hook_mod = grid.modifiers.new("Hook_Drag", "HOOK")
    hook_mod.object = empty
    hook_mod.vertex_group = PIN_GROUP_NAME
    grid.select_set(False)

    # 11. Apply cloth preset with pin group
    apply_cloth_preset(grid, preset_name)
    cloth_mod = grid.modifiers.get("Cloth")
    if cloth_mod:
        cloth_mod.settings.vertex_group_mass = PIN_GROUP_NAME
        # Ensure Hook is before Cloth
        hook_mod = grid.modifiers.get("Hook_Drag")
        if hook_mod:
            hook_idx = list(grid.modifiers).index(hook_mod)
            cloth_idx = list(grid.modifiers).index(cloth_mod)
            if hook_idx > cloth_idx:
                bpy.context.view_layer.objects.active = grid
                bpy.ops.object.modifier_move_to_index(modifier=hook_mod.name, index=cloth_idx)

    # 12. Keyframe drag motion on the Empty
    keyframe_drag(empty, grab_point, drag_target, settle_frames, frame_count)

    # 13. Bake simulation
    t0 = time.time()
    bake_simulation(frame_count)
    bake_time = time.time() - t0
    print(f"    bake: {bake_time:.1f}s ({frame_count} frames)")

    # 14. Save .blend
    save_blend(str(blend_path))
    print(f"    saved: {blend_path.relative_to(output_dir)}")

    # 15. Export Alembic
    export_alembic(str(abc_path))
    print(f"    exported: {abc_path.relative_to(output_dir)}")

    # 16. Convert .abc -> .usda
    convert_abc_to_usda(str(abc_path), str(usda_path))
    print(f"    converted: {usda_path.relative_to(output_dir)}")

    # 17. Write metadata
    metadata = {
        "mode": "grid_drag",
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
        "grab_point": [round(grab_point.x, 4), round(grab_point.y, 4), round(grab_point.z, 4)],
        "drag_target": [round(drag_target.x, 4), round(drag_target.y, 4), round(drag_target.z, 4)],
        "grab_radius": grab_radius,
        "pinned_vertices": n_pinned,
        "settle_frames": settle_frames,
        **placement,
    }
    write_metadata(str(json_path), metadata)

    return metadata


# =============================================================================
# Main
# =============================================================================


def main():
    argv = parse_blender_args(sys.argv)

    parser = argparse.ArgumentParser(description="Grid cloth drag simulation (Blender)")
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
    parser.add_argument("--grab-radius", type=float, default=GRAB_RADIUS_DEFAULT, help="Grab area radius in metres")
    parser.add_argument("--settle-frames", type=int, default=SETTLE_FRAMES_DEFAULT, help="Frames to settle before drag")

    config = load_sim_config(extract_config_path(argv))
    apply_config_defaults(parser, config)
    args = parser.parse_args(argv)

    for name in ("furniture_dir", "output_dir"):
        if not getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is required (or set in config)")

    print("=" * 70)
    print("GRID CLOTH DRAG SIMULATION")
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
                    grab_radius=args.grab_radius,
                    settle_frames=args.settle_frames,
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
