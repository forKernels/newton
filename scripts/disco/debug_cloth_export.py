"""Debug: verify cloth deformation is visible in evaluated mesh during bake."""

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

# 3. Import garment
garment = import_garment(obj_file)
bpy.context.view_layer.update()

# 4. Position above chair
drop_z = furn_max.z + garment_height(garment) + 0.3
garment.location = (center.x, center.y, drop_z)
garment.rotation_euler = (0, 0, 0)
bpy.context.view_layer.update()

# 5. Add collision + cloth
add_collision(furniture_objs)
apply_cloth_preset(garment, "Cotton")

# 6. Check cloth modifier is present
print("\n=== MODIFIER CHECK ===")
for mod in garment.modifiers:
    print(f"  Modifier: {mod.name}  type: {mod.type}")
    if mod.type == "CLOTH":
        print(f"    settings.mass: {mod.settings.mass}")
        print(f"    settings.tension_stiffness: {mod.settings.tension_stiffness}")
        print(f"    point_cache.frame_start: {mod.point_cache.frame_start}")
        print(f"    point_cache.frame_end: {mod.point_cache.frame_end}")

# 7. Bake
print("\nBaking 60 frames...")
bake_simulation(60)
print("Bake complete.")

# 8. Check cloth cache status
print("\n=== POST-BAKE CACHE STATUS ===")
for mod in garment.modifiers:
    if mod.type == "CLOTH":
        pc = mod.point_cache
        print(f"  point_cache.is_baked: {pc.is_baked}")
        print(f"  point_cache.frame_start: {pc.frame_start}")
        print(f"  point_cache.frame_end: {pc.frame_end}")
        print(f"  point_cache.is_outdated: {pc.is_outdated}")

# 9. Sample vertex positions at different frames to check deformation
print("\n=== VERTEX POSITION CHECK (vertex 0) ===")
depsgraph = bpy.context.evaluated_depsgraph_get()

for frame in [1, 10, 30, 60]:
    bpy.context.scene.frame_set(frame)
    depsgraph.update()

    eval_obj = garment.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    if mesh and len(mesh.vertices) > 0:
        v0 = mesh.vertices[0].co.copy()
        # Also check a few more vertices
        v_count = len(mesh.vertices)
        v_mid = mesh.vertices[v_count // 2].co.copy()
        v_last = mesh.vertices[-1].co.copy()

        # World space
        w0 = eval_obj.matrix_world @ v0
        w_mid = eval_obj.matrix_world @ v_mid
        w_last = eval_obj.matrix_world @ v_last

        print(f"  Frame {frame:3d}: vert[0]     local=({v0.x:.4f}, {v0.y:.4f}, {v0.z:.4f})  world=({w0.x:.4f}, {w0.y:.4f}, {w0.z:.4f})")
        print(f"             vert[{v_count//2}] local=({v_mid.x:.4f}, {v_mid.y:.4f}, {v_mid.z:.4f})  world=({w_mid.x:.4f}, {w_mid.y:.4f}, {w_mid.z:.4f})")
        print(f"             vert[{v_count-1}] local=({v_last.x:.4f}, {v_last.y:.4f}, {v_last.z:.4f})  world=({w_last.x:.4f}, {w_last.y:.4f}, {w_last.z:.4f})")
    eval_obj.to_mesh_clear()

# 10. Now try export with depsgraph evaluation
print("\n=== EXPORT TEST ===")
usda_path = str(OUTPUT / "debug_cloth_deform.usda")
blend_path = str(OUTPUT / "debug_cloth_deform.blend")

# Save blend first
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print(f"Saved: {blend_path}")

# Export USDA
bpy.ops.wm.usd_export(
    filepath=usda_path,
    export_animation=True,
    export_mesh_colors=False,
    selected_objects_only=False,
    evaluation_mode='RENDER',
)
print(f"Exported: {usda_path}")
print("\nDone.")
