"""
Blender Franka FR3 — Mesh Import
=================================
By David Clabaugh with Claude Code

Imports all Franka FR3 visual meshes at their correct zero-config positions
at world origin. No armature, no scene setup — just the robot meshes.

Run blender_franka_armature.py separately to generate the rig.
"""

import os
import xml.etree.ElementTree as ET

import bpy
from mathutils import Euler, Matrix, Vector

# =============================================================================
# CONFIGURATION
# =============================================================================

FRANKA_ASSET_DIR = (
    r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache"
    r"\newton-assets_franka_emika_panda_c9e36853\franka_emika_panda"
)
FRANKA_URDF = os.path.join(FRANKA_ASSET_DIR, "urdf", "fr3_franka_hand.urdf")


# =============================================================================
# URDF PARSING
# =============================================================================


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


# =============================================================================
# MESH IMPORT
# =============================================================================


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


# =============================================================================
# MAIN
# =============================================================================


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in [bpy.data.meshes, bpy.data.materials, bpy.data.armatures, bpy.data.actions]:
        for item in coll:
            coll.remove(item)


def import_franka_meshes():
    joints, link_meshes = parse_urdf(FRANKA_URDF)
    link_transforms = compute_link_transforms(joints)

    # Root empty at world origin
    bpy.ops.object.empty_add(type="ARROWS", location=(0, 0, 0), radius=0.08)
    root_empty = bpy.context.active_object
    root_empty.name = "FrankaRoot"

    count = 0
    for link_name, mesh_path in link_meshes.items():
        mesh = import_dae(mesh_path, f"{link_name}_visual")
        if mesh is None:
            continue
        T = link_transforms.get(link_name, Matrix.Identity(4))
        mesh.matrix_world = T
        mesh.parent = root_empty
        mesh.matrix_parent_inverse = root_empty.matrix_world.inverted()
        count += 1
        print(f"    {link_name}: {os.path.basename(mesh_path)}")

    return root_empty, count


def setup_meshes():
    print("=" * 60)
    print("FRANKA FR3 — MESH IMPORT")
    print("=" * 60)

    print("\n[1/2] Cleaning scene...")
    clean_scene()

    print("[2/2] Importing Franka meshes at world origin...")
    root, count = import_franka_meshes()

    print(f"""
{"=" * 60}
MESHES READY — Franka FR3
{"=" * 60}

  FrankaRoot at world origin (0, 0, 0)
  {count} mesh objects at zero-config positions

  NEXT: Run blender_franka_armature.py to generate the rig.
""")


if __name__ == "__main__":
    setup_meshes()
