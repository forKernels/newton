"""Debug: check if cloth solver is actually deforming vertices."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy
from cloth_sim_utils import (
    add_collision,
    append_furniture,
    apply_cloth_preset,
    clear_scene,
    furniture_bbox,
    furniture_center,
    garment_height,
    import_garment,
)
from mathutils import Vector

FURNITURE = "D:/_blender/_myBlender/SimulationWork/seedAssets/scenes/Chair_01.blend"
GARMENT_DIR = Path("D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Set")

garment_subdir = GARMENT_DIR / "dress_sleeveless_2550" / "dress_sleeveless_000YCTJ9HS"
obj_file = str(next(garment_subdir.glob("*_sim_prep.obj")))

# 1. Setup
clear_scene()
furniture_objs = append_furniture(FURNITURE)
furn_min, furn_max = furniture_bbox(furniture_objs)
center = furniture_center(furn_min, furn_max)
garment = import_garment(obj_file)
bpy.context.view_layer.update()

# 2. Position
drop_z = furn_max.z + garment_height(garment) + 0.3
garment.location = (center.x, center.y, drop_z)
garment.rotation_euler = (0, 0, 0)
bpy.context.view_layer.update()

print(f"\nChair bbox: ({furn_min.x:.3f},{furn_min.y:.3f},{furn_min.z:.3f}) to ({furn_max.x:.3f},{furn_max.y:.3f},{furn_max.z:.3f})")
print(f"Garment location: {garment.location[:]}")
print(f"Garment height: {garment_height(garment):.3f}m")

# 3. Collision + Cloth
add_collision(furniture_objs)
apply_cloth_preset(garment, "Cotton")

# 4. Check modifiers
print("\n=== MODIFIERS ===")
for obj in bpy.context.scene.objects:
    mods = [f"{m.name}({m.type})" for m in obj.modifiers]
    if mods:
        print(f"  {obj.name}: {mods}")

# 5. Check cloth settings
for mod in garment.modifiers:
    if mod.type == "CLOTH":
        s = mod.settings
        print(f"\n=== CLOTH SETTINGS ===")
        print(f"  quality: {s.quality}")
        print(f"  mass: {s.mass}")
        print(f"  tension_stiffness: {s.tension_stiffness}")
        print(f"  compression_stiffness: {s.compression_stiffness}")
        print(f"  bending_stiffness: {s.bending_stiffness}")
        print(f"  use_gravity: {s.effector_weights.gravity}")
        pc = mod.point_cache
        print(f"  cache frame_start: {pc.frame_start}")
        print(f"  cache frame_end: {pc.frame_end}")

# 6. Set frame range
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 60

for obj in scene.objects:
    for mod in obj.modifiers:
        if hasattr(mod, "point_cache"):
            mod.point_cache.frame_start = 1
            mod.point_cache.frame_end = 60

# 7. Step through frames and check vertex positions
print("\n=== STEPPING THROUGH FRAMES ===")
depsgraph = bpy.context.evaluated_depsgraph_get()

for frame in [1, 5, 10, 20, 30, 60]:
    scene.frame_set(frame)
    depsgraph.update()

    # Raw mesh verts (no modifiers)
    raw_v0 = garment.data.vertices[0].co.copy()

    # Evaluated mesh (with cloth modifier applied)
    eval_obj = garment.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    eval_v0 = eval_mesh.vertices[0].co.copy()
    eval_v_mid = eval_mesh.vertices[len(eval_mesh.vertices) // 2].co.copy()

    # World space
    w_v0 = eval_obj.matrix_world @ eval_v0
    w_v_mid = eval_obj.matrix_world @ eval_v_mid

    # Bbox of evaluated mesh in world
    ws = [eval_obj.matrix_world @ v.co for v in eval_mesh.vertices]
    zmin = min(v.z for v in ws)
    zmax = max(v.z for v in ws)
    xmin = min(v.x for v in ws)
    xmax = max(v.x for v in ws)

    eval_obj.to_mesh_clear()

    deformed = "SAME" if (eval_v0 - raw_v0).length < 0.0001 else "DIFFERENT"
    print(f"  Frame {frame:3d}: eval_v0=({eval_v0.x:.4f},{eval_v0.y:.4f},{eval_v0.z:.4f})  "
          f"raw_v0=({raw_v0.x:.4f},{raw_v0.y:.4f},{raw_v0.z:.4f})  [{deformed}]  "
          f"world_Z=[{zmin:.3f},{zmax:.3f}]  world_X=[{xmin:.3f},{xmax:.3f}]")

OUTPUT = Path("D:/_blender/_myBlender/SimulationWork/ClothDataset/_TestSims")
blend_path = str(OUTPUT / "debug_cloth_verts.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print(f"\nSaved: {blend_path}")
print("Done.")
