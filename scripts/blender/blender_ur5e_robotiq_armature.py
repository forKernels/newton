"""
Blender UR5e + Robotiq 2F-85 — Armature Generation Only
========================================================
By David Clabaugh with Claude Code

Generates armature for UR5e + Robotiq already in the scene.
Run blender_ur5e_robotiq_mesh.py FIRST.
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
ROBOTIQ_ASSET_DIR = _ASSET_PATHS.get("robotiq_2f85", "")
if not UR5E_ASSET_DIR:
    UR5E_ASSET_DIR = r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache\mujoco_menagerie_universal_robots_ur5e_XXXXXXXX\universal_robots_ur5e"
if not ROBOTIQ_ASSET_DIR:
    ROBOTIQ_ASSET_DIR = r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache\mujoco_menagerie_robotiq_2f85_XXXXXXXX\robotiq_2f85"
UR5E_MJCF = os.path.join(UR5E_ASSET_DIR, "ur5e.xml")
ROBOTIQ_MJCF = os.path.join(ROBOTIQ_ASSET_DIR, "2f85.xml")
HAND_XFORM_POS = [0.0, 0.1, 0.0]
HAND_XFORM_ROT_AXIS = [1.0, 0.0, 0.0]
HAND_XFORM_ROT_ANGLE = -1.5708
EE_BODY_NAME = "wrist_3_link"
ROBOT_BASE_POS = utils.NEWTON_SCENE["robot_base"]


def generate_armature():
    print("=" * 65)
    print("UR5e + ROBOTIQ 2F-85 — ARMATURE GENERATION")
    print("=" * 65)

    arm_root = bpy.data.objects.get("UR5e_Root")
    if arm_root is None:
        print("\nERROR: UR5e_Root not found. Run blender_ur5e_robotiq_mesh.py first.")
        return

    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and "Robotiq" in obj.name:
            bpy.data.objects.remove(obj, do_unlink=True)

    for label, path in [("UR5e MJCF", UR5E_MJCF), ("Robotiq MJCF", ROBOTIQ_MJCF)]:
        if not os.path.isfile(path):
            print(f"\nERROR: {label} not found at:\n  {path}")
            return

    arm_bodies, _ = utils.parse_mjcf(UR5E_MJCF)
    arm_transforms = utils.compute_mjcf_fk(arm_bodies)
    hand_bodies, _ = utils.parse_mjcf(ROBOTIQ_MJCF)

    base_pos = Vector(ROBOT_BASE_POS)
    base_rot = Quaternion((0, 0, 1), math.pi)
    base_matrix = Matrix.Translation(base_pos) @ base_rot.to_matrix().to_4x4()

    hand_quat = Quaternion(Vector(HAND_XFORM_ROT_AXIS), HAND_XFORM_ROT_ANGLE)
    hand_base_world = utils.compute_hand_base_transform(
        arm_transforms, EE_BODY_NAME, HAND_XFORM_POS, hand_quat, base_matrix
    )
    hand_world_transforms = utils.compute_hand_fk_in_world(hand_bodies, hand_base_world)

    arm_chain = utils._collect_joint_chain_mjcf(arm_bodies, arm_transforms, base_matrix)
    hand_chain = utils._collect_joint_chain_mjcf(hand_bodies, hand_world_transforms, Matrix.Identity(4))
    combined = arm_chain + hand_chain

    utils.create_armature("UR5e_Robotiq_Armature", combined, arm_root)

    print(f"""
{"=" * 65}
ARMATURE READY — UR5e + Robotiq 2F-85
{"=" * 65}

  UR5e_Robotiq_Armature with {len(combined)} bones ({len(arm_chain)} arm + {len(hand_chain)} hand)

  Rigging: Select mesh > Shift-select armature > Ctrl+P > Bone
""")


if __name__ == "__main__":
    generate_armature()
