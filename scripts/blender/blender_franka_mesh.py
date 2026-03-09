"""
Blender Franka FR3 — Mesh Import Only
=======================================
By David Clabaugh with Claude Code

Imports all Franka FR3 visual meshes at their correct zero-config positions,
plus a garment from the Maria dataset laid flat on the table.
Self-contained URDF parsing — does not require robot_blender_utils for meshes.

NO armature is created — run blender_franka_armature.py separately.
"""

import math
import os
import sys
import xml.etree.ElementTree as ET

import bpy
from mathutils import Euler, Matrix, Vector

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_ROOT = r"D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Set"
GARMENT_OBJ = "dress_sleeveless_2550/dress_sleeveless_000YCTJ9HS/dress_sleeveless_000YCTJ9HS_sim.obj"
CLOTH_SCALE = 0.01

FRANKA_ASSET_DIR = (
    r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache"
    r"\newton-assets_franka_emika_panda_c9e36853\franka_emika_panda"
)
FRANKA_URDF = os.path.join(FRANKA_ASSET_DIR, "urdf", "fr3_franka_hand.urdf")

NEWTON_SCENE = {
    "robot_base": (-0.5, -0.5, -0.1),
    "platform_center": (0.0, -0.5, 0.1),
    "platform_size": (0.8, 0.8, 0.2),
    "platform_top_z": 0.2,
}


def parse_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = []
    for j in root.findall("joint"):
        origin = j.find("origin")
        xyz_str = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        rpy_str = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        xyz = [float(v) for v in xyz_str.split()]
        rpy = [float(v) for v in rpy_str.split()]
        axis_elem = j.find("axis")
        axis = [float(v) for v in axis_elem.get("xyz").split()] if axis_elem is not None else [0, 0, 1]
        lim = j.find("limit")
        lower = float(lim.get("lower", "0")) if lim is not None else 0
        upper = float(lim.get("upper", "0")) if lim is not None else 0
        joints.append({
            "name": j.get("name"), "type": j.get("type"),
            "parent_link": j.find("parent").get("link"),
            "child_link": j.find("child").get("link"),
            "xyz": xyz, "rpy": rpy, "axis": axis, "lower": lower, "upper": upper,
        })
    link_meshes = {}
    for link in root.findall("link"):
        name = link.get("name")
        vis = link.find("visual")
        if vis is not None:
            geom = vis.find("geometry")
            if geom is not None:
                mesh_el = geom.find("mesh")
                if mesh_el is not None:
                    raw = mesh_el.get("filename", "")
                    if "package://" in raw:
                        rel = raw.split("package://franka_emika_panda/", 1)[-1]
                    else:
                        rel = raw
                    link_meshes[name] = os.path.join(FRANKA_ASSET_DIR, rel)
    return joints, link_meshes


def compute_link_transforms(joints):
    transforms = {"base": Matrix.Identity(4)}
    for j in joints:
        parent_T = transforms.get(j["parent_link"], Matrix.Identity(4))
        T = Matrix.Translation(Vector(j["xyz"]))
        R = Euler((j["rpy"][0], j["rpy"][1], j["rpy"][2]), "XYZ").to_matrix().to_4x4()
        transforms[j["child_link"]] = parent_T @ T @ R
    return transforms


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in [bpy.data.meshes, bpy.data.materials, bpy.data.armatures, bpy.data.actions]:
        for item in coll:
            coll.remove(item)


def create_table(garment_bbox=None):
    cx, cy, cz = NEWTON_SCENE["platform_center"]
    sx, sy, sz = NEWTON_SCENE["platform_size"]
    if garment_bbox is not None:
        margin = 0.10
        sx = max(sx, garment_bbox[0] + 2 * margin)
        sy = max(sy, garment_bbox[1] + 2 * margin)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, cz))
    table = bpy.context.active_object
    table.name = "WorkTable"
    table.scale = (sx / 2, sy / 2, sz / 2)
    mat = bpy.data.materials.new(name="TableMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.4, 0.35, 0.3, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.7
    table.data.materials.append(mat)


def create_ground():
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0))
    g = bpy.context.active_object
    g.name = "Ground"
    mat = bpy.data.materials.new(name="GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.2, 0.2, 0.2, 1.0)
    g.data.materials.append(mat)


def parse_dae_node_transform(filepath):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"
        for matrix_el in root.iter(f"{ns}matrix"):
            text = matrix_el.text.strip()
            vals = [float(v) for v in text.split()]
            if len(vals) == 16:
                return Matrix((vals[0:4], vals[4:8], vals[8:12], vals[12:16]))
    except Exception as e:
        print(f"  WARNING: could not parse DAE transform: {e}")
    return Matrix.Identity(4)


def import_dae(filepath, name):
    if not os.path.isfile(filepath):
        print(f"  WARNING: mesh not found: {filepath}")
        return None
    dae_transform = parse_dae_node_transform(filepath)
    before = set(bpy.data.objects)
    bpy.ops.wm.collada_import(filepath=filepath)
    new_objs = [o for o in bpy.data.objects if o not in before]
    if not new_objs:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for o in new_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new_objs[0]
    if len(new_objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = name
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    obj.scale = (1, 1, 1)
    obj.data.transform(dae_transform)
    obj.data.update()
    return obj


def import_garment(obj_path, color=(0.9, 0.5, 0.6, 1.0)):
    if not os.path.isfile(obj_path):
        print(f"ERROR: garment not found: {obj_path}")
        return None
    is_clean = "_clean.obj" in obj_path
    try:
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis="NEGATIVE_Y", up_axis="Z")
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward="-Y", axis_up="Z")
    cloth = bpy.context.selected_objects[0]
    cloth.name = "Garment"
    cloth.location = (0, 0, 0)
    cloth.rotation_euler = (0, 0, 0)
    cloth.scale = (1, 1, 1)
    if not is_clean:
        cloth.data.transform(Matrix.Scale(CLOTH_SCALE, 4))
        cloth.data.transform(Matrix.Rotation(math.pi, 4, "Y"))
        verts = [Vector(v.co) for v in cloth.data.vertices]
        if verts:
            xs, ys, zs = [v.x for v in verts], [v.y for v in verts], [v.z for v in verts]
            cloth.data.transform(Matrix.Translation(Vector((-(min(xs)+max(xs))/2, -(min(ys)+max(ys))/2, -min(zs)))))
        cloth.data.update()
    cloth.data.update()
    bpy.ops.object.select_all(action="DESELECT")
    cloth.select_set(True)
    bpy.context.view_layer.objects.active = cloth
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    verts = [Vector(v.co) for v in cloth.data.vertices]
    if not verts:
        return cloth
    xs, ys, zs = [v.x for v in verts], [v.y for v in verts], [v.z for v in verts]
    bbox_w, bbox_d, bbox_h = max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)
    cloth.location = Vector((NEWTON_SCENE["platform_center"][0], NEWTON_SCENE["platform_center"][1], NEWTON_SCENE["platform_top_z"] + 0.005))
    mat = bpy.data.materials.new(name="GarmentMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.85
    if cloth.data.materials:
        cloth.data.materials[0] = mat
    else:
        cloth.data.materials.append(mat)
    cloth["bbox_size"] = [bbox_w, bbox_d, bbox_h]
    return cloth


def import_franka_meshes():
    base_pos = Vector(NEWTON_SCENE["robot_base"])
    base_matrix = Matrix.Translation(base_pos)
    joints, link_meshes = parse_urdf(FRANKA_URDF)
    link_transforms = compute_link_transforms(joints)
    bpy.ops.object.empty_add(type="ARROWS", location=base_pos, radius=0.08)
    root_empty = bpy.context.active_object
    root_empty.name = "FrankaRoot"
    for link_name, mesh_path in link_meshes.items():
        mesh = import_dae(mesh_path, f"{link_name}_visual")
        if mesh is None:
            continue
        T = link_transforms.get(link_name, Matrix.Identity(4))
        mesh.matrix_world = base_matrix @ T
        mesh.parent = root_empty
        mesh.matrix_parent_inverse = root_empty.matrix_world.inverted()
        print(f"    {link_name}: {mesh_path.split('meshes/')[-1]}")
    print("\n  Joint positions (world coords, zero config):")
    for j in joints:
        if j["type"] not in ("revolute", "prismatic"):
            continue
        child_T = link_transforms.get(j["child_link"], Matrix.Identity(4))
        world_pos = base_matrix @ child_T
        world_axis = child_T.to_3x3() @ Vector(j["axis"])
        world_axis.normalize()
        bpy.ops.object.empty_add(type="SINGLE_ARROW", location=world_pos.translation, radius=0.04)
        emp = bpy.context.active_object
        emp.name = f"JNT_{j['name']}"
        emp.rotation_euler = world_axis.to_track_quat("Z", "Y").to_euler()
        emp.parent = root_empty
        emp.matrix_parent_inverse = root_empty.matrix_world.inverted()
        emp["joint_type"] = j["type"]
        emp["axis_local"] = list(j["axis"])
        emp["axis_world"] = [round(v, 4) for v in world_axis]
        emp["limit_lower"] = j["lower"]
        emp["limit_upper"] = j["upper"]
        emp["rpy"] = list(j["rpy"])
        emp["child_link"] = j["child_link"]
    tcp_T = link_transforms.get("fr3_hand_tcp", Matrix.Identity(4))
    tcp_world = base_matrix @ tcp_T
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=tcp_world.translation, radius=0.06)
    tcp = bpy.context.active_object
    tcp.name = "JNT_tcp"
    tcp.parent = root_empty
    tcp.matrix_parent_inverse = root_empty.matrix_world.inverted()
    return root_empty


def setup_meshes():
    print("=" * 60)
    print("FRANKA FR3 — MESH IMPORT (no armature)")
    print("=" * 60)
    print("\n[1/5] Cleaning scene...")
    clean_scene()
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start = 1
    scene.frame_end = 300
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    print("[2/5] Importing garment...")
    garment_bbox = None
    if GARMENT_OBJ:
        obj_path = os.path.join(DATASET_ROOT, GARMENT_OBJ)
        cloth = import_garment(obj_path)
        if cloth and "bbox_size" in cloth:
            garment_bbox = cloth["bbox_size"]
    print("[3/5] Creating environment...")
    create_ground()
    create_table(garment_bbox=garment_bbox)
    print("[4/5] Importing Franka meshes...")
    import_franka_meshes()
    print("[5/5] Camera + lighting...")
    bpy.ops.object.camera_add(location=(1.0, -2.0, 1.2))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = (math.radians(65), 0, math.radians(25))
    scene.camera = cam
    bpy.ops.object.light_add(type="SUN", location=(2, -2, 3))
    light = bpy.context.active_object
    light.name = "KeyLight"
    light.data.energy = 3.0
    rx, ry, rz = NEWTON_SCENE["robot_base"]
    print(f"""
{"=" * 60}
MESHES READY — Franka FR3
{"=" * 60}

  FrankaRoot at ({rx}, {ry}, {rz})
  9 mesh objects (link0-7 + hand) at zero-config
  Joint empties (JNT_*) show rotation axis direction

  NEXT STEP: Run blender_franka_armature.py to generate the rig.
""")


if __name__ == "__main__":
    setup_meshes()
