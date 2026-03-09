"""
Blender UR5e Arm — Armature Generation Only
=============================================
By David Clabaugh with Claude Code

Generates an armature with 6 bones at each joint position for a standalone
UR5e arm that has already been imported into the scene.

Run blender_ur5e_mesh.py FIRST to import meshes, then run this.
"""

import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ""
if not SCRIPT_DIR:
    import bpy
    SCRIPT_DIR = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import bpy
from mathutils import Matrix, Quaternion, Vector

import robot_blender_utils as utils

_ASSET_PATHS = utils.load_asset_paths()
UR5E_ASSET_DIR = _ASSET_PATHS.get("universal_robots_ur5e", "")
if not UR5E_ASSET_DIR:
    UR5E_ASSET_DIR = r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache\mujoco_menagerie_universal_robots_ur5e_XXXXXXXX\universal_robots_ur5e"
UR5E_MJCF = os.path.join(UR5E_ASSET_DIR, "ur5e.xml")
ROBOT_BASE_POS = utils.NEWTON_SCENE["robot_base"]


def generate_armature():
    print("=" * 65)
    print("UR5e ARM (standalone) — ARMATURE GENERATION")
    print("=" * 65)

    arm_root = bpy.data.objects.get("UR5e_Root")
    if arm_root is None:
        print("\nERROR: UR5e_Root not found in scene.")
        print("Run blender_ur5e_mesh.py first to import meshes.")
        return

    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and obj.name == "UR5e_Armature":
            bpy.data.objects.remove(obj, do_unlink=True)

    if not os.path.isfile(UR5E_MJCF):
        print(f"\nERROR: UR5e MJCF not found at:\n  {UR5E_MJCF}")
        return

    print("\n[1/3] Parsing robot joint data...")
    arm_bodies, _ = utils.parse_mjcf(UR5E_MJCF)
    arm_transforms = utils.compute_mjcf_fk(arm_bodies)

    base_pos = Vector(ROBOT_BASE_POS)
    base_rot = Quaternion((0, 0, 1), math.pi)
    base_matrix = Matrix.Translation(base_pos) @ base_rot.to_matrix().to_4x4()

    print("[2/3] Collecting joint chain...")
    arm_chain = utils._collect_joint_chain_mjcf(arm_bodies, arm_transforms, base_matrix)

    print("[3/3] Creating armature...")
    utils.create_armature("UR5e_Armature", arm_chain, arm_root)

    print(f"""
{"=" * 65}
ARMATURE READY — UR5e Arm (standalone)
{"=" * 65}

  UR5e_Armature with {len(arm_chain)} bones (6 revolute joints)

  Rigging steps:
    1. Select a mesh, then Shift-select the armature
    2. Ctrl+P > Bone (for rigid robot parts)
    3. In Pose Mode, select the matching bone for each mesh
""")


if __name__ == "__main__":
    generate_armature()
