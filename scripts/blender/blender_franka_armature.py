"""
Blender Franka FR3 — Armature Generation Only
===============================================
By David Clabaugh with Claude Code

Generates armature for Franka FR3 already in the scene.
Run blender_franka_mesh.py FIRST.
"""

import math
import os
import sys
import xml.etree.ElementTree as ET

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ""
if not _SCRIPT_DIR:
    import bpy
    _SCRIPT_DIR = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import bpy
from mathutils import Euler, Matrix, Vector

import robot_blender_utils as utils

FRANKA_ASSET_DIR = (
    r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache"
    r"\newton-assets_franka_emika_panda_c9e36853\franka_emika_panda"
)
FRANKA_URDF = os.path.join(FRANKA_ASSET_DIR, "urdf", "fr3_franka_hand.urdf")

NEWTON_SCENE = {
    "robot_base": (-0.5, -0.5, -0.1),
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
    return joints


def compute_link_transforms(joints):
    transforms = {"base": Matrix.Identity(4)}
    for j in joints:
        parent_T = transforms.get(j["parent_link"], Matrix.Identity(4))
        T = Matrix.Translation(Vector(j["xyz"]))
        R = Euler((j["rpy"][0], j["rpy"][1], j["rpy"][2]), "XYZ").to_matrix().to_4x4()
        transforms[j["child_link"]] = parent_T @ T @ R
    return transforms


def generate_armature():
    print("=" * 60)
    print("FRANKA FR3 — ARMATURE GENERATION")
    print("=" * 60)

    root_empty = bpy.data.objects.get("FrankaRoot")
    if root_empty is None:
        print("\nERROR: FrankaRoot not found. Run blender_franka_mesh.py first.")
        return

    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and "Franka" in obj.name:
            bpy.data.objects.remove(obj, do_unlink=True)

    if not os.path.isfile(FRANKA_URDF):
        print(f"\nERROR: Franka URDF not found at:\n  {FRANKA_URDF}")
        return

    joints = parse_urdf(FRANKA_URDF)
    link_transforms = compute_link_transforms(joints)
    base_matrix = Matrix.Translation(Vector(NEWTON_SCENE["robot_base"]))

    utils.create_armature_from_urdf(
        "Franka_Armature", joints, link_transforms, base_matrix, root_empty
    )

    movable = [j for j in joints if j["type"] in ("revolute", "prismatic")]
    print(f"""
{"=" * 60}
ARMATURE READY — Franka FR3
{"=" * 60}

  Franka_Armature with {len(movable)} bones

  Rigging: Select mesh > Shift-select armature > Ctrl+P > Bone
  When done, save .blend — export_trajectories.py reads bone rotations.
""")


if __name__ == "__main__":
    generate_armature()
