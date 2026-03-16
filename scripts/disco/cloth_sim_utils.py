"""
Cloth Simulation Utilities (Blender)
=====================================
Shared helpers for all cloth simulation scripts (drop, throw, stack, drag, ...).

Provides: scene management, furniture/garment discovery, collision + cloth
modifier setup, simulation baking, USDA export, and config loading.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import bpy
from mathutils import Vector

# =============================================================================
# Config loading
# =============================================================================

_CONFIG_FILE = Path(__file__).resolve().parent / "sim_config.json"


def load_sim_config(config_path: str | Path | None = None) -> dict:
    """Load simulation config from JSON.  Falls back to sim_config.json next
    to this file.  Returns empty dict if no config found."""
    p = Path(config_path) if config_path else _CONFIG_FILE
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def apply_config_defaults(parser, config: dict):
    """Set argparse defaults from a config dict.

    Maps config keys to CLI arg names:
      dtc_dir        ->  --dtc-dir
      scenes_dir     ->  --scenes-dir
      props_dir      ->  --props-dir
      garment_dir    ->  --garment-dir
      output_dir     ->  --output-dir
      lego_model_dir ->  --lego-model-dir
    """
    mapping = {
        "dtc_dir": "dtc_dir",
        "scenes_dir": "scenes_dir",
        "furniture_dir": "furniture_dir",
        "props_dir": "props_dir",
        "garment_dir": "garment_dir",
        "output_dir": "output_dir",
        "lego_model_dir": "lego_model_dir",
    }
    defaults = {}
    for cfg_key, arg_dest in mapping.items():
        if cfg_key in config:
            defaults[arg_dest] = config[cfg_key]
    if defaults:
        parser.set_defaults(**defaults)


def extract_config_path(argv: list[str]) -> str | None:
    """Extract --config value from argv before argparse runs."""
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
    return None


# =============================================================================
# Constants
# =============================================================================

CLOTH_PRESETS = ["Silk", "Cotton", "Denim", "Leather", "Rubber"]

# Default Blender preset dir -- can be overridden via sim_config.json
_cfg = load_sim_config()
BLENDER_PRESET_DIR = Path(
    _cfg.get(
        "blender_preset_dir",
        "C:/Program Files/Blender Foundation/Blender 4.5/4.5/scripts/presets/cloth",
    )
)

COLLISION_THICKNESS = 0.0001  # object collision distance [m]
COLLISION_FRICTION = 200.0
COLLISION_DAMPING = 1.0  # collision damping on furniture
BBOX_MARGIN = 0.10  # 10% inward margin for XY placement

# Furniture stems to skip (not droppable surfaces)
FURNITURE_EXCLUDE = {"PoolBalls_Set", "PoolTable_01", "PoolTable_02", "Chess_Table"}


# =============================================================================
# Scene helpers
# =============================================================================


def clear_scene():
    """Remove all objects, meshes, and orphan data from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        bpy.data.textures.remove(block)
    for block in bpy.data.images:
        bpy.data.images.remove(block)


def append_furniture(blend_path: str) -> list[bpy.types.Object]:
    """Append all mesh objects from a .blend file into the current scene."""
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects

    appended = []
    scene = bpy.context.scene
    for obj in data_to.objects:
        if obj is not None:
            scene.collection.objects.link(obj)
            if obj.type == "MESH":
                appended.append(obj)

    return appended


def import_garment(obj_path: str) -> bpy.types.Object:
    """Import a garment OBJ and return the object.

    Applies the OBJ import rotation (Y-up → Z-up) directly to the mesh data
    so that ``rotation_euler`` can be freely set later without losing the
    upright orientation.
    """
    try:
        bpy.ops.wm.obj_import(filepath=obj_path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=obj_path)

    garment = bpy.context.selected_objects[0]

    # Bake the importer's Y-up → Z-up rotation into the mesh vertices
    bpy.context.view_layer.objects.active = garment
    garment.select_set(True)
    bpy.ops.object.transform_apply(rotation=True, scale=True, location=False)

    # Centre the origin on the geometry so rotations pivot correctly
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

    garment.select_set(False)
    return garment


# =============================================================================
# Bounding box helpers
# =============================================================================


def furniture_bbox(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    """Compute the combined world-space bounding box of furniture objects.

    Returns (bb_min, bb_max).
    """
    all_min = Vector((float("inf"), float("inf"), float("inf")))
    all_max = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in objects:
        for corner in obj.bound_box:
            world_co = obj.matrix_world @ Vector(corner)
            all_min.x = min(all_min.x, world_co.x)
            all_min.y = min(all_min.y, world_co.y)
            all_min.z = min(all_min.z, world_co.z)
            all_max.x = max(all_max.x, world_co.x)
            all_max.y = max(all_max.y, world_co.y)
            all_max.z = max(all_max.z, world_co.z)

    return all_min, all_max


def furniture_center(furn_min: Vector, furn_max: Vector) -> Vector:
    """Return the XYZ center of the furniture bounding box."""
    return (furn_min + furn_max) / 2.0


def garment_height(obj: bpy.types.Object) -> float:
    """Return the Z extent (height) of a garment in world space."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return max(v.z for v in corners) - min(v.z for v in corners)


def garment_bottom_offset(obj: bpy.types.Object) -> float:
    """Return how far below the object origin the garment's lowest point is.

    Use this to place the garment so its bottom edge sits at a target Z::

        garment.location.z = target_z + garment_bottom_offset(garment)
    """
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    bottom_z = min(v.z for v in corners)
    return obj.location.z - bottom_z


# =============================================================================
# Collision and cloth setup
# =============================================================================


def add_collision(objects: list[bpy.types.Object]):
    """Add Collision modifier to each mesh object with high friction/damping."""
    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.modifier_add(type="COLLISION")
        obj.collision.thickness_outer = COLLISION_THICKNESS
        obj.collision.cloth_friction = COLLISION_FRICTION
        obj.collision.damping = COLLISION_DAMPING
        obj.collision.use_culling = False
        obj.select_set(False)


CLOTH_QUALITY_STEPS = 30  # cloth simulation substeps per frame
CLOTH_COLLISION_QUALITY = 8  # collision solver iterations
CLOTH_MASS = 0.04  # per-vertex mass [kg] (range 0.03–0.08 works well)
SELF_COLLISION_DISTANCE = 0.0001  # self-collision distance [m]


def prepare_garment_mesh(garment: bpy.types.Object) -> bpy.types.Object:
    """Set Shade Smooth on the garment."""
    bpy.context.view_layer.objects.active = garment
    garment.select_set(True)
    bpy.ops.object.shade_smooth()
    garment.select_set(False)
    return garment


def apply_cloth_preset(garment: bpy.types.Object, preset_name: str):
    """Add Cloth modifier and apply a Blender built-in cloth preset.

    Blender's built-in preset files reference ``bpy.context.cloth`` which is
    only available when the cloth modifier is selected in the UI properties
    panel.  In background (headless) mode this attribute doesn't exist, so we
    rewrite the preset code to access the modifier directly via a local
    variable instead.
    """
    bpy.context.view_layer.objects.active = garment
    garment.select_set(True)
    bpy.ops.object.modifier_add(type="CLOTH")

    preset_file = BLENDER_PRESET_DIR / f"{preset_name}.py"
    if preset_file.exists():
        # Replace UI-only bpy.context.cloth with the actual modifier object
        code = preset_file.read_text()
        code = code.replace("bpy.context.cloth", "_cloth_mod")
        cloth_mod = garment.modifiers["Cloth"]
        exec(compile(code, str(preset_file), "exec"), {"bpy": bpy, "_cloth_mod": cloth_mod})
    else:
        print(f"  WARNING: Preset not found: {preset_file}, using Blender defaults")

    # Override preset values -- applied AFTER preset exec
    cloth_mod = garment.modifiers.get("Cloth")
    if cloth_mod:
        s = cloth_mod.settings
        s.quality = CLOTH_QUALITY_STEPS
        s.mass = CLOTH_MASS

        # Stiffness
        s.tension_stiffness = 10.0
        s.compression_stiffness = 10.0
        s.shear_stiffness = 10.0
        s.bending_stiffness = 0.5

        # Damping -- high values so cloth settles quickly after landing
        s.tension_damping = 10.0
        s.compression_damping = 10.0
        s.shear_damping = 10.0
        s.bending_damping = 0.5

        s.use_pressure = False

        # Object collision
        cloth_mod.collision_settings.use_collision = True
        cloth_mod.collision_settings.distance_min = 0.0001
        cloth_mod.collision_settings.collision_quality = CLOTH_COLLISION_QUALITY
        cloth_mod.collision_settings.impulse_clamp = 5.0

        # Self-collision
        cloth_mod.collision_settings.use_self_collision = True
        cloth_mod.collision_settings.self_distance_min = SELF_COLLISION_DISTANCE
        cloth_mod.collision_settings.self_impulse_clamp = 5.0

        print(
            f"    cloth: quality={s.quality} mass={s.mass}kg "
            f"stiffness T={s.tension_stiffness} C={s.compression_stiffness} "
            f"S={s.shear_stiffness} B={s.bending_stiffness}"
        )
        print(
            f"    cloth damping: T={s.tension_damping} C={s.compression_damping} "
            f"S={s.shear_damping} B={s.bending_damping} air={s.air_damping}"
        )
        print(
            f"    cloth collision: quality={cloth_mod.collision_settings.collision_quality} "
            f"obj_dist={cloth_mod.collision_settings.distance_min} "
            f"self_dist={cloth_mod.collision_settings.self_distance_min}"
        )




# =============================================================================
# Simulation and export
# =============================================================================


def bake_simulation(frame_count: int, fps: int | None = None):
    """Bake all physics caches for the given number of frames.

    Sets the frame range on the scene **and** on every individual point cache
    (cloth, rigid body, soft body, ...) so the bake doesn't overshoot or
    fall back to Blender's default 250-frame range.

    If *fps* is given, the scene render fps is set accordingly.
    """
    scene = bpy.context.scene
    if fps is not None:
        scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = frame_count

    # Align per-modifier point caches (cloth, soft body, …)
    for obj in scene.objects:
        for mod in obj.modifiers:
            if hasattr(mod, "point_cache"):
                mod.point_cache.frame_start = 1
                mod.point_cache.frame_end = frame_count

    # Align rigid body world cache
    if scene.rigidbody_world and scene.rigidbody_world.point_cache:
        scene.rigidbody_world.point_cache.frame_start = 1
        scene.rigidbody_world.point_cache.frame_end = frame_count

    bpy.ops.ptcache.bake_all(bake=True)
    scene.frame_set(frame_count)


def export_alembic(filepath: str):
    """Export the scene as Alembic (.abc) with deformed mesh animation.

    Alembic natively handles deforming meshes from cloth/physics caches,
    unlike Blender's USD exporter which writes only the rest mesh.
    """
    scene = bpy.context.scene
    bpy.ops.wm.alembic_export(
        filepath=filepath,
        start=scene.frame_start,
        end=scene.frame_end,
        evaluation_mode="RENDER",
        export_hair=False,
        export_particles=False,
        export_custom_properties=True,
        flatten=False,
    )


def export_usda(filepath: str):
    """Export the scene as USDA with animation.

    Note: Blender's USD exporter does not reliably capture cloth/physics
    deformation.  Prefer ``export_alembic()`` for simulation data.
    The scene frame range (frame_start/frame_end) must be set before calling.
    """
    bpy.ops.wm.usd_export(
        filepath=filepath,
        export_animation=True,
        export_mesh_colors=False,
        selected_objects_only=False,
    )


# =============================================================================
# Discovery
# =============================================================================


def find_prepped_garments(
    garment_dir: str,
    category: str | None = None,
) -> list[dict]:
    """Find all *_sim_prep.obj garment meshes.

    Returns list of dicts with keys: category, garment_id, obj_path.
    """
    root = Path(garment_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Garment directory not found: {garment_dir}")

    if category:
        cat_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith(category)]
    else:
        cat_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.endswith(".zip"))

    items = []
    for cat_dir in cat_dirs:
        for garment_subdir in sorted(cat_dir.iterdir()):
            if not garment_subdir.is_dir():
                continue
            preps = list(garment_subdir.glob("*_sim_prep.obj"))
            if preps:
                items.append(
                    {
                        "category": cat_dir.name,
                        "garment_id": garment_subdir.name,
                        "obj_path": str(preps[0]),
                    }
                )
    return items


def find_furniture(furniture_dir: str, specific: str | None = None) -> list[Path]:
    """Find furniture .blend files, searching subdirectories recursively.

    Scans top-level .blend files and subdirectories (Chair/, Table/, Sofa/, etc.)
    for .blend files.  Excludes assets in FURNITURE_EXCLUDE.

    If *specific* is given, searches recursively for a .blend file with that name.
    """
    root = Path(furniture_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Furniture directory not found: {furniture_dir}")

    if specific:
        # Search recursively for the specific file
        matches = list(root.rglob(specific))
        if not matches:
            # Try as a stem (without .blend extension)
            matches = list(root.rglob(f"{specific}.blend")) if not specific.endswith(".blend") else []
        if not matches:
            raise FileNotFoundError(f"Furniture not found: {specific} in {furniture_dir}")
        return [matches[0]]

    # Collect all .blend files recursively
    blends = sorted(root.rglob("*.blend"))
    # Exclude backup files (.blend1, .blend2) and non-droppable assets
    return [
        b for b in blends
        if b.suffix == ".blend"
        and b.stem not in FURNITURE_EXCLUDE
        and not b.name.endswith((".blend1", ".blend2"))
    ]


# =============================================================================
# CLI helpers
# =============================================================================


def parse_blender_args(argv: list[str]) -> list[str]:
    """Strip Blender's own args, return everything after '--'."""
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return []


def resolve_presets(preset_arg: str) -> list[str]:
    """Convert a --cloth-preset arg into a list of preset names."""
    if preset_arg.lower() == "all":
        return list(CLOTH_PRESETS)
    if preset_arg not in CLOTH_PRESETS:
        raise ValueError(f"Unknown preset '{preset_arg}'. Choose from: {CLOTH_PRESETS}")
    return [preset_arg]


def make_seed(base_seed: int, *parts) -> int:
    """Deterministic seed from a base seed and hashable components."""
    return base_seed + hash(parts) % (2**31)


def write_metadata(json_path: str, metadata: dict):
    """Write a JSON sidecar file."""
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)


# =============================================================================
# Floor plane, vertex group, blend save, ABC→USDA conversion
# =============================================================================


def add_floor_plane(z_position: float, size: float = 20.0) -> bpy.types.Object:
    """Create a large collision plane at *z_position*.

    Uses the same thickness and friction as ``add_collision()``.
    Returns the created plane object.
    """
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, z_position))
    floor = bpy.context.active_object
    floor.name = "Floor"

    bpy.ops.object.modifier_add(type="COLLISION")
    floor.collision.thickness_outer = COLLISION_THICKNESS
    floor.collision.cloth_friction = COLLISION_FRICTION

    floor.select_set(False)
    return floor


def save_blend(filepath: str):
    """Save the current scene as a .blend file."""
    bpy.ops.wm.save_as_mainfile(filepath=str(filepath))


def create_vertex_group(
    garment: bpy.types.Object,
    group_name: str,
    center: Vector,
    radius: float,
) -> tuple:
    """Create a vertex group for vertices near *center* (world space).

    Returns ``(vertex_group, pinned_count)``.
    """
    bpy.context.view_layer.objects.active = garment
    garment.select_set(True)

    if group_name in garment.vertex_groups:
        garment.vertex_groups.remove(garment.vertex_groups[group_name])
    vg = garment.vertex_groups.new(name=group_name)

    mesh = garment.data
    pinned = []
    for v in mesh.vertices:
        world_co = garment.matrix_world @ v.co
        if (world_co - center).length <= radius:
            pinned.append(v.index)

    if pinned:
        vg.add(pinned, 1.0, "REPLACE")

    garment.select_set(False)
    return vg, len(pinned)


def create_cloth_grid(
    size: float = 1.0,
    cuts: int = 30,
    thickness: float = 0.001,
    subdiv_level: int = 1,
) -> bpy.types.Object:
    """Create a subdivided grid mesh with thickness for cloth simulation.

    Adds Solidify + Subdivision Surface modifiers, applies them so the
    final mesh is baked, then sets Shade Smooth and centres the origin.

    Args:
        size: Grid side length in metres.
        cuts: Subdivision cuts per axis.
        thickness: Solidify thickness in metres.
        subdiv_level: Subdivision Surface viewport level.
    """
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=cuts, y_subdivisions=cuts, size=size)
    grid = bpy.context.active_object
    grid.name = "ClothGrid"

    bpy.context.view_layer.objects.active = grid
    grid.select_set(True)

    # Solidify for thickness
    sol = grid.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = thickness
    sol.offset = 0.0

    # Subdivision Surface for smoother deformation
    sub = grid.modifiers.new("Subdiv", "SUBSURF")
    sub.levels = subdiv_level
    sub.render_levels = subdiv_level + 1

    # Apply both modifiers so the mesh is baked before cloth sim
    bpy.ops.object.modifier_apply(modifier="Solidify")
    bpy.ops.object.modifier_apply(modifier="Subdiv")

    # Shade Smooth by Angle (Blender 4.x: adds Smooth by Angle modifier)
    try:
        bpy.ops.object.shade_auto_smooth()
    except AttributeError:
        bpy.ops.object.shade_smooth()

    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

    grid.select_set(False)
    return grid


def convert_abc_to_usda(abc_path: str, usda_path: str):
    """Import an Alembic file and re-export as USDA.

    This roundtrip captures per-frame deformed mesh data that Blender's
    direct USD exporter misses for cloth/physics simulations.
    The scene frame range is preserved from the prior bake.
    """
    clear_scene()

    bpy.ops.wm.alembic_import(filepath=str(abc_path))

    bpy.ops.wm.usd_export(
        filepath=str(usda_path),
        export_animation=True,
        export_mesh_colors=False,
        selected_objects_only=False,
    )


# =============================================================================
# Main loop runner
# =============================================================================


def run_sim_loop(
    furniture_files: list[Path],
    garments: list[dict],
    presets: list[str],
    num_samples: int,
    base_seed: int,
    output_dir: Path,
    sim_fn,
):
    """Generic simulation loop.  Calls sim_fn for each combination.

    sim_fn signature:
        sim_fn(furniture_path, garment_info, preset_name, output_dir, seed, sample_idx)
        -> dict  (must contain 'skipped' or 'error' keys on non-success)
    """
    total = len(furniture_files) * len(garments) * len(presets) * num_samples
    print(
        f"Total simulations: {total} "
        f"({len(furniture_files)} furniture x {len(garments)} garments "
        f"x {len(presets)} presets x {num_samples} samples)"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    completed = 0
    skipped = 0
    errors = 0

    for furn_path in furniture_files:
        for garment_info in garments:
            for preset in presets:
                for sample_i in range(1, num_samples + 1):
                    seed = make_seed(
                        base_seed,
                        furn_path.stem,
                        garment_info["garment_id"],
                        preset,
                        sample_i,
                    )

                    result = sim_fn(
                        furniture_path=furn_path,
                        garment_info=garment_info,
                        preset_name=preset,
                        output_dir=output_dir,
                        seed=seed,
                        sample_idx=sample_i,
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
