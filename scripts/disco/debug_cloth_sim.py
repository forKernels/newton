"""Debug: set up cloth scene, bake, save .blend + .usda for inspection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy
from cloth_sim_utils import (
    add_collision,
    append_furniture,
    apply_cloth_preset,
    bake_simulation,
    clear_scene,
    export_usda,
    furniture_bbox,
    furniture_center,
    garment_height,
    import_garment,
)
from mathutils import Vector

FURNITURE = "D:/_blender/_myBlender/SimulationWork/seedAssets/scenes/Chair_01.blend"
GARMENT_DIR = Path("D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Set")
OUTPUT = Path("D:/_blender/_myBlender/SimulationWork/ClothDataset/_TestSims")

garment_subdir = GARMENT_DIR / "dress_sleeveless_2550" / "dress_sleeveless_000YCTJ9HS"
obj_file = str(next(garment_subdir.glob("*_sim_prep.obj")))

# 1. Clean
clear_scene()

# 2. Import furniture
furniture_objs = append_furniture(FURNITURE)
furn_min, furn_max = furniture_bbox(furniture_objs)
center = furniture_center(furn_min, furn_max)
print(f"Chair center: ({center.x:.3f}, {center.y:.3f}, {center.z:.3f})  top Z: {furn_max.z:.3f}")

# 3. Import garment (now applies Y-up → Z-up rotation to mesh data)
garment = import_garment(obj_file)
bpy.context.view_layer.update()

# Check that local and world bboxes now match (rotation baked in)
g_local = [Vector(c) for c in garment.bound_box]
g_world = [garment.matrix_world @ Vector(c) for c in garment.bound_box]
local_z = max(v.z for v in g_local) - min(v.z for v in g_local)
world_z = max(v.z for v in g_world) - min(v.z for v in g_world)
print(f"Garment height — local Z: {local_z:.3f}m  world Z: {world_z:.3f}m  (should match ~1.037)")
print(f"garment_height(): {garment_height(garment):.3f}m")
print(f"rotation_euler: {garment.rotation_euler[:]}")

# 4. Position: garment centered above furniture, 30cm gap
drop_x = center.x
drop_y = center.y
drop_z = furn_max.z + garment_height(garment) + 0.3

garment.location = (drop_x, drop_y, drop_z)
garment.rotation_euler = (0, 0, 0)
bpy.context.view_layer.update()

# Verify
g_world2 = [garment.matrix_world @ Vector(c) for c in garment.bound_box]
wmin = Vector((min(v.x for v in g_world2), min(v.y for v in g_world2), min(v.z for v in g_world2)))
wmax = Vector((max(v.x for v in g_world2), max(v.y for v in g_world2), max(v.z for v in g_world2)))
print(f"\nPositioned garment world bbox:")
print(f"  X: [{wmin.x:.3f}, {wmax.x:.3f}]  chair: [{furn_min.x:.3f}, {furn_max.x:.3f}]")
print(f"  Y: [{wmin.y:.3f}, {wmax.y:.3f}]  chair: [{furn_min.y:.3f}, {furn_max.y:.3f}]")
print(f"  Z: [{wmin.z:.3f}, {wmax.z:.3f}]  chair top: {furn_max.z:.3f}")

# 5. Add collision + cloth
add_collision(furniture_objs)
apply_cloth_preset(garment, "Cotton")

# 6. Bake
print("\nBaking 60 frames...")
bake_simulation(60)
print("Bake complete.")

# 7. Save
blend_path = str(OUTPUT / "debug_cloth_baked.blend")
usda_path = str(OUTPUT / "debug_cloth_baked.usda")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
export_usda(usda_path)
print(f"\n.blend: {blend_path}")
print(f".usda:  {usda_path}")
