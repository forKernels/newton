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

    Maps config keys to CLI arg names (underscores become hyphens):
      furniture_dir  ->  --furniture-dir
      garment_dir    ->  --garment-dir
      output_dir     ->  --output-dir
    """
    mapping = {
        "furniture_dir": "furniture_dir",
        "garment_dir": "garment_dir",
        "output_dir": "output_dir",
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

COLLISION_THICKNESS = 0.002
COLLISION_FRICTION = 5.0
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
    """Import a garment OBJ and return the object."""
    try:
        bpy.ops.wm.obj_import(filepath=obj_path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=obj_path)

    return bpy.context.selected_objects[0]


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


# =============================================================================
# Collision and cloth setup
# =============================================================================


def add_collision(objects: list[bpy.types.Object]):
    """Add Collision modifier to each mesh object."""
    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.modifier_add(type="COLLISION")
        obj.collision.thickness_outer = COLLISION_THICKNESS
        obj.collision.cloth_friction = COLLISION_FRICTION
        obj.select_set(False)


def apply_cloth_preset(garment: bpy.types.Object, preset_name: str):
    """Add Cloth modifier and apply a Blender built-in cloth preset."""
    bpy.context.view_layer.objects.active = garment
    garment.select_set(True)
    bpy.ops.object.modifier_add(type="CLOTH")

    preset_file = BLENDER_PRESET_DIR / f"{preset_name}.py"
    if preset_file.exists():
        exec(compile(preset_file.read_text(), str(preset_file), "exec"))
    else:
        print(f"  WARNING: Preset not found: {preset_file}, using Blender defaults")


# =============================================================================
# Simulation and export
# =============================================================================


def bake_simulation(frame_count: int):
    """Bake the cloth simulation for the given number of frames."""
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count

    bpy.ops.ptcache.bake_all(bake=True)
    scene.frame_set(frame_count)


def export_usda(filepath: str):
    """Export the scene as USDA with animation."""
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
    """Find furniture .blend files, excluding non-droppable assets."""
    root = Path(furniture_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Furniture directory not found: {furniture_dir}")

    if specific:
        p = root / specific
        if not p.exists():
            raise FileNotFoundError(f"Furniture not found: {p}")
        return [p]

    blends = sorted(root.glob("*.blend"))
    return [b for b in blends if b.stem not in FURNITURE_EXCLUDE]


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
