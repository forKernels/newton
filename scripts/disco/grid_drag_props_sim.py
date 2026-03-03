"""
Grid Cloth Drag Over Props Simulation (Blender)
=================================================
Places props (scanned objects) on the floor or on furniture, then drags a
grid cloth across them.  The cloth catches and drapes over props as it passes.

Props are loaded from a directory of OBJ/USD/blend files (e.g. MuJoCo scanned
objects).  Optionally a furniture piece (table, chair) is placed as the base
surface.

Output pipeline: bake -> .blend -> .abc -> .usda -> .json

Usage (headless):
  blender --background --python scripts/disco/grid_drag_props_sim.py -- \
      --props-dir "C:/_git/mujoco_scanned_objects/models" \
      --output-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Sims" \
      --cloth-preset Cotton --num-samples 5 --num-props 3

  # With furniture as base surface:
  blender --background --python scripts/disco/grid_drag_props_sim.py -- \
      --props-dir "C:/_git/mujoco_scanned_objects/models" \
      --furniture-dir "D:/_blender/_myBlender/SimulationWork/seedAssets/scenes" \
      --furniture Chair_01.blend \
      --num-props 2 --grid-size 1.5

  # Floor only (no furniture):
  blender --background --python scripts/disco/grid_drag_props_sim.py -- \
      --props-dir "C:/_git/mujoco_scanned_objects/models" \
      --surface floor --num-props 4
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
    furniture_center,
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

DEFAULT_FRAMES = 60
DEFAULT_GRID_SIZE = 1.0
DEFAULT_GRID_CUTS = 30
DEFAULT_THICKNESS = 0.001
DEFAULT_SUBDIV = 1
GRAB_RADIUS_DEFAULT = 0.08
SETTLE_FRAMES_DEFAULT = 10
DRAG_FRAC = 0.6  # fraction of total frames spent dragging (rest is settle at end)
RELEASE_CHANCE = 0.5  # probability that the cloth gets unhooked after dragging
DRAG_HEIGHT_M = 0.03  # hover height above highest point during drag
PIN_GROUP_NAME = "drag_pin"
DEFAULT_NUM_PROPS = 3
PROP_SCALE_RANGE = (0.5, 1.5)  # random scale multiplier for props
SCATTER_FRAMES = 40  # rigid body pre-sim frames for physics scatter
SCATTER_LIFT_PROPS = 0.3  # metres to lift props before dropping
SCATTER_LIFT_SCENES = 0.15  # metres to lift furniture before dropping


# =============================================================================
# Prop discovery and import
# =============================================================================


def find_props(props_dir: str, limit: int | None = None) -> list[Path]:
    """Find importable prop files (.obj, .usd, .usda, .usdc, .blend).

    For mujoco_scanned_objects layout: looks for models/*/model.obj
    Also supports flat directories of mixed formats.
    """
    root = Path(props_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Props directory not found: {props_dir}")

    props = []

    # mujoco layout: models/*/model.obj
    for model_obj in sorted(root.glob("*/model.obj")):
        props.append(model_obj)

    # Also pick up loose files
    for ext in ("*.obj", "*.usd", "*.usda", "*.usdc", "*.blend"):
        for f in sorted(root.glob(ext)):
            if f not in props:
                props.append(f)

    if limit and len(props) > limit:
        props = props[:limit]

    return props


def import_prop(filepath: Path) -> list[bpy.types.Object]:
    """Import a prop file and return the imported mesh objects."""
    ext = filepath.suffix.lower()
    before = set(bpy.data.objects)

    if ext == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=str(filepath))
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=str(filepath))
    elif ext in (".usd", ".usda", ".usdc"):
        bpy.ops.wm.usd_import(filepath=str(filepath))
    elif ext == ".blend":
        with bpy.data.libraries.load(str(filepath), link=False) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.scene.collection.objects.link(obj)
    else:
        print(f"    WARNING: Unsupported format {ext}: {filepath.name}")
        return []

    after = set(bpy.data.objects)
    new_objs = [o for o in (after - before) if o.type == "MESH"]

    # Apply transforms
    for obj in new_objs:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(rotation=True, scale=True, location=False)
        obj.select_set(False)

    return new_objs


def place_props_on_surface(
    prop_paths: list[Path],
    surface_z: float,
    surface_center: Vector,
    spread: float,
    rng: random.Random,
) -> tuple[list[bpy.types.Object], list[dict]]:
    """Import and scatter props on a surface.

    Returns (all_prop_objects, prop_metadata_list).
    """
    all_objs = []
    meta_list = []

    for i, prop_path in enumerate(prop_paths):
        objs = import_prop(prop_path)
        if not objs:
            continue

        # Random position within spread radius
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(0, spread)
        x = surface_center.x + dist * math.cos(angle)
        y = surface_center.y + dist * math.sin(angle)

        # Random scale
        scale = rng.uniform(*PROP_SCALE_RANGE)

        # Random Z rotation
        rot_z = rng.uniform(0, 360)

        for obj in objs:
            # Scale first
            obj.scale = (scale, scale, scale)
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.transform_apply(scale=True)

            # Position on surface: find prop's bottom and place it at surface_z
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            bottom = min(c.z for c in corners)
            offset_z = obj.location.z - bottom

            obj.location = (x, y, surface_z + offset_z)
            obj.rotation_euler = (0, 0, math.radians(rot_z))
            obj.select_set(False)

        all_objs.extend(objs)
        meta_list.append({
            "prop": prop_path.parent.name if prop_path.name == "model.obj" else prop_path.stem,
            "position": [round(x, 4), round(y, 4), round(surface_z, 4)],
            "scale": round(scale, 3),
            "rotation_z": round(rot_z, 1),
        })

        print(f"    prop[{i}]: {meta_list[-1]['prop']} scale={scale:.2f} at ({x:.3f}, {y:.3f})")

    return all_objs, meta_list


# =============================================================================
# Physics scatter (rigid body pre-sim)
# =============================================================================


def physics_scatter(
    scatter_objs: list[bpy.types.Object],
    ground_objs: list[bpy.types.Object],
    lift: float = 0.3,
    frames: int = SCATTER_FRAMES,
    add_rotation: bool = True,
    rng: random.Random | None = None,
) -> None:
    """Drop objects with rigid body physics and apply settled positions.

    Runs a short rigid body simulation so objects tumble and settle naturally,
    then bakes the final positions back into the object transforms and removes
    all rigid body data (so the cloth sim starts clean).

    Args:
        scatter_objs: Objects to drop (active rigid bodies).
        ground_objs: Objects that stay fixed (passive rigid bodies, e.g. floor/furniture).
        lift: Metres to lift scatter objects before dropping.
        frames: Number of rigid body sim frames.
        add_rotation: Add random rotation to scatter objects before dropping.
        rng: Random number generator for rotation offsets.
    """
    if not scatter_objs:
        return

    scene = bpy.context.scene

    # Ensure rigid body world exists
    if not scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()

    # Add passive rigid body to ground objects
    for obj in ground_objs:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.rigidbody.object_add(type="PASSIVE")
        obj.rigid_body.friction = 0.8
        obj.select_set(False)

    # Lift and add active rigid body to scatter objects
    for obj in scatter_objs:
        obj.location.z += lift
        if add_rotation and rng:
            obj.rotation_euler.x += math.radians(rng.uniform(-20, 20))
            obj.rotation_euler.y += math.radians(rng.uniform(-20, 20))

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.rigidbody.object_add(type="ACTIVE")
        obj.rigid_body.mass = 0.5
        obj.rigid_body.friction = 0.8
        obj.rigid_body.restitution = 0.1
        obj.select_set(False)

    bpy.context.view_layer.update()

    # Set frame range for scatter sim
    scene.frame_start = 1
    scene.frame_end = frames
    scene.rigidbody_world.point_cache.frame_start = 1
    scene.rigidbody_world.point_cache.frame_end = frames

    # Bake rigid body sim
    bpy.ops.ptcache.bake_all(bake=True)
    scene.frame_set(frames)

    # Apply visual transforms (settled positions become real transforms)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in scatter_objs:
        obj.select_set(True)
    if scatter_objs:
        bpy.context.view_layer.objects.active = scatter_objs[0]
        bpy.ops.object.visual_transform_apply()
    for obj in scatter_objs:
        obj.select_set(False)

    # Remove rigid body from all objects
    all_rb_objs = scatter_objs + ground_objs
    for obj in all_rb_objs:
        if obj.rigid_body:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.rigidbody.object_remove()
            obj.select_set(False)

    # Free all baked caches and remove rigid body world
    bpy.ops.ptcache.free_bake_all()
    if scene.rigidbody_world:
        bpy.ops.rigidbody.world_remove()

    # Reset frame
    scene.frame_set(1)

    print(f"    physics scatter: {len(scatter_objs)} objects settled over {frames} frames")


# =============================================================================
# Drag planning (reused from grid_drag_sim)
# =============================================================================


def pick_drag_across_surface(
    grid: bpy.types.Object,
    surface_min: Vector,
    surface_max: Vector,
    rng: random.Random,
) -> tuple[Vector, Vector]:
    """Pick grab point on grid edge and drag target across the surface."""
    corners = [grid.matrix_world @ Vector(c) for c in grid.bound_box]
    g_min = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    g_max = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    g_cx = (g_min.x + g_max.x) / 2
    g_cy = (g_min.y + g_max.y) / 2
    g_z = (g_min.z + g_max.z) / 2

    edge = rng.randint(0, 3)
    if edge == 0:
        grab_x, grab_y = g_max.x - 0.02, g_cy + rng.uniform(-0.05, 0.05)
        drag_dir = (-1, rng.uniform(-0.3, 0.3))
    elif edge == 1:
        grab_x, grab_y = g_min.x + 0.02, g_cy + rng.uniform(-0.05, 0.05)
        drag_dir = (1, rng.uniform(-0.3, 0.3))
    elif edge == 2:
        grab_x, grab_y = g_cx + rng.uniform(-0.05, 0.05), g_max.y - 0.02
        drag_dir = (rng.uniform(-0.3, 0.3), -1)
    else:
        grab_x, grab_y = g_cx + rng.uniform(-0.05, 0.05), g_min.y + 0.02
        drag_dir = (rng.uniform(-0.3, 0.3), 1)

    grab_point = Vector((grab_x, grab_y, g_z))

    g_extent = max(g_max.x - g_min.x, g_max.y - g_min.y)
    drag_dist = rng.uniform(0.5, 0.9) * g_extent

    drag_len = math.sqrt(drag_dir[0] ** 2 + drag_dir[1] ** 2)
    drag_dx = drag_dir[0] / drag_len * drag_dist
    drag_dy = drag_dir[1] / drag_len * drag_dist

    drag_target = Vector((grab_x + drag_dx, grab_y + drag_dy, surface_max.z + DRAG_HEIGHT_M))

    # Clamp within surface bounds
    x_span = surface_max.x - surface_min.x
    y_span = surface_max.y - surface_min.y
    margin_x = x_span * BBOX_MARGIN
    margin_y = y_span * BBOX_MARGIN
    drag_target.x = max(surface_min.x + margin_x, min(surface_max.x - margin_x, drag_target.x))
    drag_target.y = max(surface_min.y + margin_y, min(surface_max.y - margin_y, drag_target.y))

    return grab_point, drag_target


def keyframe_drag(
    empty: bpy.types.Object,
    grab_point: Vector,
    drag_target: Vector,
    settle_frames: int,
    drag_end_frame: int,
):
    """Keyframe the Empty: hold for settle, then slide to drag target.

    The drag ends at *drag_end_frame*; remaining frames are for rest/settle.
    """
    scene = bpy.context.scene

    # Frame 1: at grab point
    scene.frame_set(1)
    empty.location = grab_point
    empty.keyframe_insert(data_path="location", frame=1)

    # Hold through settle period
    scene.frame_set(settle_frames)
    empty.location = grab_point
    empty.keyframe_insert(data_path="location", frame=settle_frames)

    # Drag to target (ends before total_frames so cloth can rest)
    scene.frame_set(drag_end_frame)
    empty.location = drag_target
    empty.keyframe_insert(data_path="location", frame=drag_end_frame)

    # Hold at drag target for remaining frames
    # (pin will be released so cloth falls free)

    if empty.animation_data and empty.animation_data.action:
        for fcurve in empty.animation_data.action.fcurves:
            for kfp in fcurve.keyframe_points:
                kfp.interpolation = "LINEAR"

    scene.frame_set(1)


# =============================================================================
# Single simulation run
# =============================================================================


def run_single_sim(
    furniture_path: Path | None,
    prop_paths: list[Path],
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
    scatter_props: bool = False,
    scatter_scenes: bool = False,
) -> dict:
    """Execute one grid cloth drag-over-props simulation."""
    rng = random.Random(seed)
    surface_name = furniture_path.stem if furniture_path else "floor"

    # Output paths
    sim_dir = output_dir / surface_name / "grid_drag_props" / preset_name
    sim_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"grid_{grid_size}m_{sample_idx:03d}"
    blend_path = sim_dir / f"{base_name}.blend"
    abc_path = sim_dir / f"{base_name}.abc"
    usda_path = sim_dir / f"{base_name}.usda"
    json_path = sim_dir / f"{base_name}.json"

    if usda_path.exists() and json_path.exists():
        print(f"  SKIP (exists): {usda_path.relative_to(output_dir)}")
        return {"skipped": True}

    print(f"  SIM [grid_drag_props]: {surface_name} / {preset_name} #{sample_idx} ({len(prop_paths)} props)")

    # 1. Clean scene
    clear_scene()

    # 2. Place furniture first (if any)
    furniture_objs = []
    if furniture_path:
        furniture_objs = append_furniture(str(furniture_path))
        if not furniture_objs:
            print(f"    WARNING: No mesh in {furniture_path.name}, skipping")
            return {"error": "no_furniture_meshes"}

    # 3. Floor at half the chair height (so legs are visible, chair sits on floor)
    if furniture_path:
        furn_min, furn_max = furniture_bbox(furniture_objs)
        chair_height = furn_max.z - furn_min.z
        floor_z = furn_max.z - chair_height / 2
    else:
        floor_z = -0.01
    floor_obj = add_floor_plane(floor_z, size=10.0)

    # 4. Physics scatter furniture if requested
    if scatter_scenes and furniture_objs:
        print("    scattering furniture...")
        physics_scatter(
            scatter_objs=furniture_objs,
            ground_objs=[floor_obj],
            lift=SCATTER_LIFT_SCENES,
            add_rotation=False,  # keep furniture upright
            rng=rng,
        )
        # Recompute bbox after scatter (furniture may have shifted)
        furn_min, furn_max = furniture_bbox(furniture_objs)

    if furniture_path:
        surface_z = furn_max.z
        surface_center = furniture_center(furn_min, furn_max)
        add_collision(furniture_objs)
    else:
        surface_z = 0.0
        surface_center = Vector((0, 0, 0))
        furn_min = Vector((-2, -2, -0.01))
        furn_max = Vector((2, 2, 0))

    # 5. Import and place props ON TOP of the surface
    spread = grid_size * 0.4
    prop_objs, prop_meta = place_props_on_surface(
        prop_paths, surface_z, surface_center, spread, rng,
    )

    # 6. Physics scatter props if requested
    if scatter_props and prop_objs:
        print("    scattering props...")
        ground = [floor_obj] + furniture_objs
        physics_scatter(
            scatter_objs=prop_objs,
            ground_objs=ground,
            lift=SCATTER_LIFT_PROPS,
            add_rotation=True,
            rng=rng,
        )

    if prop_objs:
        add_collision(prop_objs)

    # 7. Compute the highest point of all props + furniture
    scene_top_z = surface_z
    if prop_objs:
        for obj in prop_objs:
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            top = max(c.z for c in corners)
            scene_top_z = max(scene_top_z, top)
    print(f"    scene top Z: {scene_top_z:.3f} (surface={surface_z:.3f})")

    # 8. Create grid cloth
    grid = create_cloth_grid(size=grid_size, cuts=grid_cuts, thickness=thickness, subdiv_level=subdiv_level)

    # 9. Position grid ABOVE everything (above highest prop + gap)
    cloth_start_z = scene_top_z + 0.03
    rot_z_deg = rng.uniform(0, 360)
    grid.location = (surface_center.x, surface_center.y, cloth_start_z)
    grid.rotation_euler = (0, 0, math.radians(rot_z_deg))
    bpy.context.view_layer.update()

    # 10. Surface bounds for drag planning (use scene_top_z for drag height)
    surface_min = Vector((surface_center.x - grid_size, surface_center.y - grid_size, surface_z))
    surface_max = Vector((surface_center.x + grid_size, surface_center.y + grid_size, scene_top_z))

    # 11. Pick grab point and drag target
    grab_point, drag_target = pick_drag_across_surface(grid, surface_min, surface_max, rng)

    # 12. Create pin vertex group
    vg, n_pinned = create_vertex_group(grid, PIN_GROUP_NAME, grab_point, grab_radius)
    if n_pinned == 0:
        vg, n_pinned = create_vertex_group(grid, PIN_GROUP_NAME, grab_point, grab_radius * 3)
        if n_pinned == 0:
            print(f"    WARNING: No vertices pinned, skipping")
            return {"error": "no_pinned_vertices"}

    print(f"    grab at ({grab_point.x:.3f}, {grab_point.y:.3f}, {grab_point.z:.3f})")
    print(f"    drag to ({drag_target.x:.3f}, {drag_target.y:.3f}, {drag_target.z:.3f})")
    print(f"    pinned vertices: {n_pinned}")

    # 13. Create Empty at grab point for Hook modifier
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=grab_point)
    empty = bpy.context.active_object
    empty.name = "Drag_Empty"

    # 14. Add Hook modifier
    bpy.context.view_layer.objects.active = grid
    grid.select_set(True)
    hook_mod = grid.modifiers.new("Hook_Drag", "HOOK")
    hook_mod.object = empty
    hook_mod.vertex_group = PIN_GROUP_NAME
    grid.select_set(False)

    # 15. Apply cloth preset with pin group
    apply_cloth_preset(grid, preset_name)
    cloth_mod = grid.modifiers.get("Cloth")
    if cloth_mod:
        cloth_mod.settings.vertex_group_mass = PIN_GROUP_NAME
        hook_mod = grid.modifiers.get("Hook_Drag")
        if hook_mod:
            hook_idx = list(grid.modifiers).index(hook_mod)
            cloth_idx = list(grid.modifiers).index(cloth_mod)
            if hook_idx > cloth_idx:
                bpy.context.view_layer.objects.active = grid
                bpy.ops.object.modifier_move_to_index(modifier=hook_mod.name, index=cloth_idx)

    # 16. Compute drag/release timing
    #   settle_frames: initial hold
    #   drag phase: DRAG_FRAC of remaining frames
    #   rest phase: cloth released, settles freely
    drag_end_frame = settle_frames + int((frame_count - settle_frames) * DRAG_FRAC)
    release_frame = drag_end_frame + 1
    print(f"    timing: settle=1-{settle_frames} drag={settle_frames}-{drag_end_frame} "
          f"release={release_frame} rest={release_frame}-{frame_count}")

    # 17. Keyframe drag motion (drag ends at drag_end_frame, not total_frames)
    keyframe_drag(empty, grab_point, drag_target, settle_frames, drag_end_frame)

    # 18. Randomly release cloth after dragging (50% chance)
    do_release = rng.random() < RELEASE_CHANCE
    if cloth_mod and do_release:
        scene = bpy.context.scene

        # Pin stiffness: 1.0 until release, then 0.0
        scene.frame_set(release_frame - 1)
        cloth_mod.settings.pin_stiffness = 1.0
        cloth_mod.settings.keyframe_insert("pin_stiffness", frame=release_frame - 1)

        scene.frame_set(release_frame)
        cloth_mod.settings.pin_stiffness = 0.0
        cloth_mod.settings.keyframe_insert("pin_stiffness", frame=release_frame)

        # Disable Hook so Empty stops pulling vertices
        hook_mod = grid.modifiers.get("Hook_Drag")
        if hook_mod:
            scene.frame_set(release_frame - 1)
            hook_mod.show_viewport = True
            hook_mod.show_render = True
            hook_mod.keyframe_insert("show_viewport", frame=release_frame - 1)
            hook_mod.keyframe_insert("show_render", frame=release_frame - 1)

            scene.frame_set(release_frame)
            hook_mod.show_viewport = False
            hook_mod.show_render = False
            hook_mod.keyframe_insert("show_viewport", frame=release_frame)
            hook_mod.keyframe_insert("show_render", frame=release_frame)

        scene.frame_set(1)
        print(f"    release: YES (frame {release_frame})")
    else:
        print(f"    release: NO (cloth stays pinned)")

    # 19. Bake simulation
    t0 = time.time()
    bake_simulation(frame_count)
    bake_time = time.time() - t0
    print(f"    bake: {bake_time:.1f}s ({frame_count} frames)")

    # 20. Save .blend
    save_blend(str(blend_path))
    print(f"    saved: {blend_path.relative_to(output_dir)}")

    # 21. Export Alembic
    export_alembic(str(abc_path))
    print(f"    exported: {abc_path.relative_to(output_dir)}")

    # 22. Convert .abc -> .usda
    convert_abc_to_usda(str(abc_path), str(usda_path))
    print(f"    converted: {usda_path.relative_to(output_dir)}")

    # 23. Write metadata
    metadata = {
        "mode": "grid_drag_props",
        "surface": surface_name,
        "furniture": furniture_path.name if furniture_path else None,
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
        "props": prop_meta,
        "num_props": len(prop_meta),
        "grab_point": [round(grab_point.x, 4), round(grab_point.y, 4), round(grab_point.z, 4)],
        "drag_target": [round(drag_target.x, 4), round(drag_target.y, 4), round(drag_target.z, 4)],
        "grab_radius": grab_radius,
        "pinned_vertices": n_pinned,
        "settle_frames": settle_frames,
        "drag_end_frame": drag_end_frame,
        "release_frame": release_frame,
        "drag_frac": DRAG_FRAC,
        "floor_z": round(floor_z, 4),
        "scene_top_z": round(scene_top_z, 4),
        "released": do_release,
        "scatter_props": scatter_props,
        "scatter_scenes": scatter_scenes,
    }
    write_metadata(str(json_path), metadata)

    return metadata


# =============================================================================
# Main
# =============================================================================


def main():
    argv = parse_blender_args(sys.argv)

    parser = argparse.ArgumentParser(description="Grid cloth drag over props simulation (Blender)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--props-dir", type=str, default=None, help="Directory of prop models (OBJ/USD/blend)")
    parser.add_argument("--furniture-dir", type=str, default=None, help="Furniture .blend directory (optional)")
    parser.add_argument("--furniture", type=str, default=None, help="Specific furniture .blend (optional)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cloth-preset", type=str, default="Cotton")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-props", type=int, default=DEFAULT_NUM_PROPS, help="Number of props per sim")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--grid-size", type=float, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--grid-cuts", type=int, default=DEFAULT_GRID_CUTS)
    parser.add_argument("--thickness", type=float, default=DEFAULT_THICKNESS)
    parser.add_argument("--subdiv-level", type=int, default=DEFAULT_SUBDIV)
    parser.add_argument("--grab-radius", type=float, default=GRAB_RADIUS_DEFAULT)
    parser.add_argument("--settle-frames", type=int, default=SETTLE_FRAMES_DEFAULT)
    parser.add_argument("--scatter-props", action="store_true", help="Drop props with rigid body physics before cloth sim")
    parser.add_argument("--scatter-scenes", action="store_true", help="Drop furniture with rigid body physics before cloth sim")
    parser.add_argument("--scatter-all", action="store_true", help="Shortcut: scatter both props and scenes")

    config = load_sim_config(extract_config_path(argv))
    apply_config_defaults(parser, config)
    args = parser.parse_args(argv)

    if not args.props_dir:
        parser.error("--props-dir is required")
    if not args.output_dir:
        parser.error("--output-dir is required (or set in config)")

    # Resolve --scatter-all shortcut
    if args.scatter_all:
        args.scatter_props = True
        args.scatter_scenes = True

    print("=" * 70)
    print("GRID CLOTH DRAG OVER PROPS SIMULATION")
    print("=" * 70)

    # Discover props
    all_props = find_props(args.props_dir)
    print(f"Props available: {len(all_props)}")

    # Furniture (optional)
    furniture_path = None
    if args.furniture_dir and args.furniture:
        furniture_files = find_furniture(args.furniture_dir, args.furniture)
        if furniture_files:
            furniture_path = furniture_files[0]
            print(f"Furniture: {furniture_path.name}")
    elif args.furniture_dir:
        furniture_files = find_furniture(args.furniture_dir)
        print(f"Furniture pool: {len(furniture_files)} files")

    presets = resolve_presets(args.cloth_preset)
    print(f"Presets: {presets}")
    print(f"Grid: {args.grid_size}m, {args.grid_cuts} cuts")
    print(f"Props per sim: {args.num_props}")
    if args.scatter_props or args.scatter_scenes:
        parts = []
        if args.scatter_scenes:
            parts.append("scenes")
        if args.scatter_props:
            parts.append("props")
        print(f"Physics scatter: {' + '.join(parts)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If we have a furniture pool, iterate; otherwise just one surface
    if args.furniture_dir and not args.furniture:
        furniture_list = find_furniture(args.furniture_dir)
    elif furniture_path:
        furniture_list = [furniture_path]
    else:
        furniture_list = [None]  # floor only

    total = len(furniture_list) * len(presets) * args.num_samples
    print(f"Total simulations: {total}")

    completed = 0
    errors = 0
    skipped = 0
    t_start = time.time()

    for furn_path in furniture_list:
        for preset in presets:
            for sample_i in range(1, args.num_samples + 1):
                seed = args.seed + hash((
                    furn_path.stem if furn_path else "floor",
                    preset,
                    sample_i,
                )) % (2**31)

                # Pick random props for this sim
                sim_rng = random.Random(seed)
                n = min(args.num_props, len(all_props))
                prop_paths = sim_rng.sample(all_props, n)

                result = run_single_sim(
                    furniture_path=furn_path,
                    prop_paths=prop_paths,
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
                    scatter_props=args.scatter_props,
                    scatter_scenes=args.scatter_scenes,
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
